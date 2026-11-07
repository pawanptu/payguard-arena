"""THE one-shot final evaluation: pre-registered in D-022, defenders fixed in D-028, LLM judges added in D-029.

This is the only script that builds TEST_SEEN / TEST_HELDOUT episodes. Run from the repo root:

  python -m scripts.final_eval --plan                   distinct LLM payloads per split and how many each provider has
                                                        not cached yet. No API call, no defender, no metric.
  python -m scripts.final_eval --warm sarvam --cap 600  ask ONE frozen LLM (same model, prompt v1, payload buckets as on
                                                        TRAIN) about payloads it has not answered yet, most frequent first.
                                                        Prints call counts only: never a verdict, a risk or a metric.
  python -m scripts.final_eval --run                    THE one-shot. Every defender on TRAIN, TEST_SEEN and TEST_HELDOUT;
                                                        LLM judges are cache-only here (no call can happen), then the
                                                        pre-registered go/no-go, eval/final/results.json and eval/replay.json.
                                                        Refuses when eval/final/results.json exists: a rerun has to be a
                                                        visible, deliberate act (delete the file, say why in DECISIONS).

Why warm first: an LLM payload is computed from history before the event and does not depend on any verdict, so
the slow, rate-limited calls can be made per distinct payload, ahead of time, without running a defender. The
scoring pass then is a single fast offline run. Calls made here are recorded in eval/final/llm_spend.json and
capped per provider across runs (Sarvam: 600 paid calls on TEST payloads, D-029).
All numbers are SIMULATED replays on synthetic data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from payguard.api.replay import parse_replay
from payguard.baselines.static_rules import StaticRules
from payguard.baselines.trivial import AlwaysHold, AlwaysRelease
from payguard.evolve import replay_export as rx
from payguard.llm.cache import cache_key, canonical
from payguard.llm.judge import PROMPT_VERSION, LLMJudge, make_assessor
from payguard.llm.payload import judge_payload
from payguard.protocol import (HELDOUT_FAMILIES, TEST_HELDOUT_SEEDS, TEST_SEEN_SEEDS, TRAIN_FAMILIES, TRAIN_SEEDS,
                               episode_for, episodes)
from payguard.sim.features import PlatformLog
from payguard.sim.metrics import Metrics
from payguard.sim.runner import run_episode
from payguard.swarm.coordinator import make_swarm
from payguard.swarm.memory import make_memory_swarm
from payguard.swarm.objective import objective
from payguard.swarm.params import SwarmParams, default_params
from payguard.swarm.single import SPECIALIST_NAMES, make_single
from scripts.run_baselines import fmt, run

ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / "eval" / "final"
RESULTS_PATH = FINAL_DIR / "results.json"
SPEND_PATH = FINAL_DIR / "llm_spend.json"
REPLAY_PATH = ROOT / "eval" / "replay.json"
CHAMPION_PATH = ROOT / "eval" / "evolution" / "champion.json"
LINEAGE_PATH = ROOT / "eval" / "evolution" / "lineage.json"

SPLITS: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = {
    "TRAIN": (TRAIN_SEEDS, TRAIN_FAMILIES),
    "TEST_SEEN": (TEST_SEEN_SEEDS, TRAIN_FAMILIES),
    "TEST_HELDOUT": (TEST_HELDOUT_SEEDS, HELDOUT_FAMILIES),
}
PROVIDERS = ("sarvam", "gemini")
# Calls this script may make per provider, summed over every --warm run (the ledger enforces it).
BUDGET = {"sarvam": 600, "gemini": 1500}  # Sarvam: paid, user-approved (D-029). Gemini: free tier, quota-bound.

# Pre-registered (D-022 criterion, D-028 primary defender). Do not edit after a --run.
PRIMARY = "swarm_evolved_memory"
DECISIVE_SPLIT = "TEST_HELDOUT"
MARGIN = 3.0
SWARMS = ("swarm_frozen", "swarm_l1", "swarm_evolved", "swarm_memory", "swarm_evolved_memory")
# The dashboard's episode, played by the primary defender: the first held-out sleeper episode (user's choice
# D-031, made before any TEST number existed, so it is not cherry-picked). Shown whatever the result is.
EPISODE = ("TEST_HELDOUT", TEST_HELDOUT_SEEDS[0])


# ---------------------------------------------------------------- payloads, plan, warm ----------------
def split_episodes(split: str):
    seeds, families = SPLITS[split]
    return episodes(seeds, families)


def distinct_payloads(eps) -> tuple[Counter, dict[str, dict]]:
    """Payload frequency over `eps` (canonical JSON -> count) and one payload per canonical form."""
    freq: Counter[str] = Counter()
    payload_of: dict[str, dict] = {}
    for ep in eps:
        log = PlatformLog(ep.accounts)
        for e in ep.events:
            p = judge_payload(e, log)
            log.add(e)
            c = canonical(p)
            freq[c] += 1
            payload_of[c] = p
    return freq, payload_of


def all_payloads() -> dict[str, tuple[Counter, dict[str, dict]]]:
    out = {}
    for split in SPLITS:
        t = time.time()
        out[split] = distinct_payloads(split_episodes(split))
        print(f"  {split:13s} {len(out[split][0]):4d} distinct payloads over {sum(out[split][0].values()):,} "
              f"payouts ({time.time() - t:.0f}s)", flush=True)
    return out


def is_cached(assessor, payload: dict) -> bool:
    return assessor.cache.get(cache_key(assessor.model, PROMPT_VERSION, payload)) is not None


def plan() -> int:
    print("Distinct LLM payloads (truth-free buckets; no call, no defender, no metric):")
    per_split = all_payloads()
    spend = load_spend()
    for provider in PROVIDERS:
        a = make_assessor(provider, offline=True)
        print(f"{provider} (model {a.model}; spent here {spend_calls(spend, provider)}/{BUDGET[provider]}):")
        union: Counter[str] = Counter()
        for split, (freq, payload_of) in per_split.items():
            missing = [c for c in freq if not is_cached(a, payload_of[c])]
            uncovered = sum(freq[c] for c in missing)
            union.update({c: freq[c] for c in missing})
            print(f"  {split:13s} uncached {len(missing):4d}/{len(freq):4d} distinct = "
                  f"{100 * uncovered / sum(freq.values()):5.1f}% of payouts")
        print(f"  all splits    {len(union)} distinct payloads to ask")
    return 0


def warm(provider: str, cap: int) -> int:
    spend = load_spend()
    left = BUDGET[provider] - spend_calls(spend, provider)
    allowed = min(cap, left)
    if allowed <= 0:
        print(f"{provider}: budget used up ({spend_calls(spend, provider)}/{BUDGET[provider]}); nothing was called.")
        return 1
    assessor = make_assessor(provider, max_calls=allowed, offline=False)
    if assessor.offline or assessor.client is None:
        print(f"{provider}: no API key in the environment; nothing was called.")
        return 1
    print(f"{provider}: collecting payloads (model {assessor.model}) ...")
    union: Counter[str] = Counter()
    payload_of: dict[str, dict] = {}
    for freq, pof in all_payloads().values():
        union.update(freq)
        payload_of.update(pof)
    todo = [payload_of[c] for c in sorted(union, key=lambda c: (-union[c], c)) if not is_cached(assessor, payload_of[c])]
    print(f"{provider}: {len(todo)} uncached distinct payloads; at most {allowed} calls "
          f"(budget left {left}). Counts only are printed.", flush=True)
    t0, asked = time.time(), 0
    try:
        for p in todo:
            if assessor.calls >= allowed or getattr(assessor, "quota_exhausted", False):
                break
            assessor.assess_payload(p)
            asked += 1
            if asked % 25 == 0:
                print(f"  calls={assessor.calls} errors={assessor.fallbacks.get('error', 0)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        errors = assessor.fallbacks.get("error", 0)
        record_spend(provider, assessor.calls, errors)
        still = sum(1 for p in todo if not is_cached(assessor, p))
        print(f"{provider}: calls={assessor.calls} errors={errors} still uncached={still}/{len(todo)} "
              f"quota_exhausted={bool(getattr(assessor, 'quota_exhausted', False))} ({time.time() - t0:.0f}s)")
    return 0


# ---------------------------------------------------------------- spend ledger -------------------------
def load_spend() -> dict:
    return json.loads(SPEND_PATH.read_text(encoding="utf-8")) if SPEND_PATH.is_file() else {}


def spend_calls(spend: dict, provider: str) -> int:
    return int(spend.get(provider, {}).get("calls", 0))


def record_spend(provider: str, calls: int, errors: int) -> None:
    """Re-reads the ledger right before writing, so warm runs of two providers at once keep each other's entries."""
    spend = load_spend()
    entry = spend.setdefault(provider, {"calls": 0, "budget": BUDGET[provider], "runs": []})
    entry["calls"] = int(entry["calls"]) + int(calls)
    entry["runs"].append({"at": now_iso(), "calls": int(calls), "errors": int(errors)})
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    SPEND_PATH.write_text(json.dumps(spend, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- the one-shot -------------------------
def load_champion() -> tuple[SwarmParams, dict]:
    raw = CHAMPION_PATH.read_bytes()
    doc = json.loads(raw)
    return SwarmParams(**doc["params"]), {"id": doc.get("id"), "sha256": hashlib.sha256(raw).hexdigest()}


def defender_factories(champion: SwarmParams) -> list[tuple[str, object]]:
    d = default_params()
    return [
        ("always_release", AlwaysRelease),
        ("always_hold", AlwaysHold),
        ("static_rules", StaticRules),
        *[(f"single_{n}", lambda n=n: make_single(n, d)) for n in SPECIALIST_NAMES],
        ("swarm_frozen", lambda: make_swarm(d, learn_weights=False)),
        ("swarm_l1", lambda: make_swarm(d, learn_weights=True)),
        ("swarm_evolved", lambda: make_swarm(champion, learn_weights=True)),
        ("swarm_memory", lambda: make_memory_swarm(d, None, name="swarm_memory")),
        ("swarm_evolved_memory", lambda: make_memory_swarm(champion, None, name="swarm_evolved_memory")),
    ]


def check(metrics: dict[tuple[str, str], Metrics], split: str, candidate: str, best_single: str) -> dict:
    """D-022 on one split: objective >= MARGIN below BOTH static_rules and the TRAIN-chosen best single specialist,
    and honest block rate no higher than static_rules' on that split."""
    c, s, b = (metrics[(n, split)] for n in (candidate, "static_rules", best_single))
    oc, os_, ob = objective(c), objective(s), objective(b)
    out = {"split": split, "candidate": candidate, "objective": oc, "static_rules": os_, best_single: ob,
           "margin_vs_static": os_ - oc, "margin_vs_best_single": ob - oc,
           "honest_block_rate": c.honest_block_rate, "static_honest_block_rate": s.honest_block_rate}
    out["pass"] = bool(out["margin_vs_static"] >= MARGIN and out["margin_vs_best_single"] >= MARGIN
                       and c.honest_block_rate <= s.honest_block_rate)
    return out


def go_no_go(metrics: dict[tuple[str, str], Metrics], best_single: str) -> dict:
    tests = [s for s in SPLITS if s != "TRAIN"]
    primary = {s: check(metrics, s, PRIMARY, best_single) for s in tests}
    secondary = {sw: {s: check(metrics, s, sw, best_single)["pass"] for s in tests} for sw in SWARMS if sw != PRIMARY}
    return {"criterion": f"D-022: on {DECISIVE_SPLIT}, {PRIMARY} objective >= {MARGIN} below both static_rules and "
                         f"{best_single} (best single specialist on TRAIN), honest block <= static_rules'",
            "primary": PRIMARY, "best_single_on_train": best_single, "checks": primary,
            "verdict": "GO" if primary[DECISIVE_SPLIT]["pass"] else "NO-GO",
            "secondary_pass_not_decisive": secondary}


class _Recorder:
    """Stores the memory swarm's specialist scores (+ memory) and L1 weights at decision time."""

    def __init__(self, swarm):
        self.swarm, self.name, self.at = swarm, swarm.name, {}

    def decide(self, event, log):
        v = self.swarm.decide(event, log)
        s = self.swarm.last_signals
        self.at[event.id] = ({**s["scores"], "memory": float(s.get("memory", 0.0))}, dict(s["weights"]))
        return v

    def learn(self, event, is_attack):
        self.swarm.learn(event, is_attack)


def episode_doc(champion: SwarmParams, episode: tuple[str, int] = EPISODE) -> dict:
    split, seed = episode
    ep = episode_for(seed, SPLITS[split][1])
    rec = _Recorder(make_memory_swarm(champion, None, name=PRIMARY))
    res = run_episode(ep, rec)
    events = []
    for e in ep.events:
        oc, tr = res.outcomes[e.id], ep.truth[e.id]
        sig, w = rec.at[e.id]
        events.append({"id": e.id, "t": e.t, "worker_id": e.worker_id, "requester_id": e.requester_id,
                       "amount": e.amount, "verdict": res.verdicts[e.id].value, "paid": bool(oc.paid),
                       "paid_at": oc.paid_at, "is_attack": bool(tr.is_attack), "family": tr.family,
                       "signals": sig, "weights": w})
    return {"seed": seed, "split": split, "defender": PRIMARY, "events": events}


def display_names(best_single: str) -> list[str]:
    return ["always_release", "always_hold", "static_rules", best_single,
            *[f"llm_judge_{p}" for p in PROVIDERS], *SWARMS]


def build_final_replay(metrics: dict[tuple[str, str], Metrics], best_single: str, champion: SwarmParams,
                       generated_at: str, episode: tuple[str, int] = EPISODE) -> dict:
    rows = [rx.result_row(n, metrics[(n, s)], s) for s in SPLITS for n in display_names(best_single)
            if (n, s) in metrics]
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    doc = rx.clean({"schema_version": 1, "simulated": True, "synthetic_fixture": False, "generated_at": generated_at,
                    "objective": rx.OBJECTIVE_TEXT, "results": rows, "evolution": rx.evolution_rows(lineage),
                    "episode": episode_doc(champion, episode), "payouts": []})
    parse_replay(doc, "final replay")  # the dashboard's contract is the final word
    size = len(json.dumps(doc, separators=(",", ":")))
    if size > rx.MAX_BYTES:
        raise SystemExit(f"final replay would be {size:,} bytes (> {rx.MAX_BYTES:,}); nothing written")
    return doc


def run_once(providers: tuple[str, ...]) -> int:
    if RESULTS_PATH.exists():
        print(f"{RESULTS_PATH.relative_to(ROOT)} exists: the one-shot already ran. A rerun must be deliberate: "
              "delete the file and record why in docs/DECISIONS.md first.")
        return 2
    champion, champ_meta = load_champion()
    generated_at = now_iso()
    metrics: dict[tuple[str, str], Metrics] = {}
    llm: dict[str, dict] = {p: {} for p in providers}
    t0 = time.time()
    for split in SPLITS:
        eps = split_episodes(split)
        print(f"== {split}: {len(eps)} episodes, {sum(len(e.events) for e in eps):,} payouts", flush=True)
        for name, make in defender_factories(champion):
            metrics[(name, split)] = m = run(make, eps)
            print(f"  {name:21s} obj={objective(m):6.2f}  {fmt(m)}", flush=True)
        for p in providers:
            a = make_assessor(p, offline=True)  # cache-only: no API call can happen in the scoring pass
            name = f"llm_judge_{p}"
            metrics[(name, split)] = m = run(lambda a=a, name=name: LLMJudge(a, name=name), eps)
            llm[p][split] = {"model": a.model, "cache_hits": a.cache_hits, "fallbacks": a.fallback_count,
                             "fallback_share_pct": 100 * a.fallback_count / max(1, a.cache_hits + a.fallback_count)}
            print(f"  {name:21s} obj={objective(m):6.2f}  {fmt(m)}  fallback={llm[p][split]['fallback_share_pct']:.1f}%",
                  flush=True)
    best_single = rx.best_single({n: m for (n, s), m in metrics.items() if s == "TRAIN"})
    decision = go_no_go(metrics, best_single)
    results = {
        "generated_at": generated_at, "git_rev": git_rev(), "simulated": True, "champion": champ_meta,
        "objective": rx.OBJECTIVE_TEXT, "splits": {s: {"seeds": [v[0][0], v[0][-1]], "families": list(v[1])}
                                                    for s, v in SPLITS.items()},
        "go_no_go": decision, "llm": llm,
        "rows": [{"defender": n, "split": s, "objective": objective(m), **asdict(m)} for (n, s), m in metrics.items()],
    }
    replay = build_final_replay(metrics, best_single, champion, generated_at)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(rx.clean(results), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    size = rx.write_replay(REPLAY_PATH, replay)
    for s, c in decision["checks"].items():
        print(f"{s:13s} {PRIMARY} obj={c['objective']:.2f}  vs static {c['margin_vs_static']:+.2f}  "
              f"vs {best_single} {c['margin_vs_best_single']:+.2f}  block {c['honest_block_rate']:.1f}% "
              f"(static {c['static_honest_block_rate']:.1f}%)  pass={c['pass']}")
    print(f"VERDICT ({DECISIVE_SPLIT} decides): {decision['verdict']}   secondary: "
          f"{decision['secondary_pass_not_decisive']}")
    print(f"wrote {RESULTS_PATH.relative_to(ROOT)} and {REPLAY_PATH.relative_to(ROOT)} ({size:,} bytes) "
          f"in {time.time() - t0:.0f}s")
    return 0


# ---------------------------------------------------------------- helpers / CLI ------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="count uncached payloads per split and provider (no calls)")
    g.add_argument("--warm", choices=PROVIDERS, help="ask this provider about uncached payloads (needs --cap)")
    g.add_argument("--run", action="store_true", help="THE one-shot evaluation (refuses if results exist)")
    ap.add_argument("--cap", type=int, help="calls allowed in this --warm run (also bounded by the ledger budget)")
    ap.add_argument("--providers", default=",".join(PROVIDERS),
                    help="LLM judges scored by --run, comma-separated (default: %(default)s)")
    args = ap.parse_args(argv)
    if args.plan:
        return plan()
    if args.warm:
        if args.cap is None or args.cap < 1:
            ap.error("--warm needs --cap N (N >= 1)")
        return warm(args.warm, args.cap)
    providers = tuple(p for p in args.providers.split(",") if p)
    if not set(providers) <= set(PROVIDERS):
        ap.error(f"--providers must be a subset of {PROVIDERS}")
    return run_once(providers)


if __name__ == "__main__":
    sys.exit(main())
