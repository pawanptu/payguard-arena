"""THE round-2 one-shot evaluation: pre-registered in D-034, amended in D-035 (Gemini row added later).

This is the only script that builds TEST_HELDOUT2 episodes (seeds 4000-4019, the sealed family in
payguard/heldout2/). Every command except --freeze-defender refuses to run until swarm_r2 is frozen. Run from the
repo root:

  python -m scripts.final_eval_r2 --freeze-defender      record sha256 of eval/r2/champion.json, payguard/robust/*.py
                                                         and payguard/swarm/*.py in eval/final_r2/defender_freeze.json.
                                                         Once. Nothing below runs before this file exists and matches.
  python -m scripts.final_eval_r2 --plan                 distinct LLM payloads on TEST_HELDOUT2 and how many each
                                                         provider has not cached. No call, no defender, no metric.
  python -m scripts.final_eval_r2 --warm sarvam --cap N  ask ONE frozen LLM about uncached TEST_HELDOUT2 payloads.
                                                         Same shared ledger and budget as round 1
                                                         (eval/final/llm_spend.json; Sarvam 600 paid calls in total).
  python -m scripts.final_eval_r2 --run                  THE one-shot (D-035: without llm_judge_gemini). Every display
                                                         defender + swarm_r2 on TEST_HELDOUT2, swarm_r2 also on TRAIN /
                                                         TEST_SEEN / TEST_HELDOUT (reported only), the pre-registered
                                                         verdict, eval/final_r2/results.json and eval/replay.json.
                                                         Refuses when results exist.
  python -m scripts.final_eval_r2 --gemini-row           after Gemini's quota reset (D-035): llm_judge_gemini on
                                                         TEST_HELDOUT2 only, cache-only, run once, written to
                                                         eval/final_r2/gemini_row.json; the replay is rebuilt. It can
                                                         never change results.json or the verdict.

All numbers are SIMULATED replays on synthetic data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import payguard.heldout2  # noqa: F401  registers the sealed family before any episode is built
from payguard.api.replay import parse_replay
from payguard.evolve import replay_export as rx
from payguard.heldout2.protocol import HELDOUT2_FAMILIES, TEST_HELDOUT2_SEEDS
from payguard.heldout2.protocol import episode_for as sealed_episode_for
from payguard.heldout2.protocol import episodes as sealed_episodes
from payguard.llm.judge import LLMJudge, make_assessor
from payguard.sim.metrics import Metrics
from payguard.sim.runner import run_episode
from payguard.swarm.objective import objective
from scripts import final_eval as r1
from scripts.run_baselines import fmt, run

ROOT = r1.ROOT
OUT_DIR = ROOT / "eval" / "final_r2"
RESULTS_PATH = OUT_DIR / "results.json"
GEMINI_PATH = OUT_DIR / "gemini_row.json"
FREEZE_PATH = OUT_DIR / "defender_freeze.json"
R2_CHAMPION_PATH = ROOT / "eval" / "r2" / "champion.json"
DEFENDER_DIRS = (ROOT / "payguard" / "robust", ROOT / "payguard" / "swarm")

# Pre-registered (D-034). Do not edit after a --run.
PRIMARY = "swarm_r2"
DECISIVE_SPLIT = "TEST_HELDOUT2"
BEST_SINGLE = "single_policy"  # best single specialist on TRAIN, fixed in round 1 (D-033); re-checked below
MARGIN = r1.MARGIN
SEALED_SEEDS: tuple[int, ...] = TEST_HELDOUT2_SEEDS
SEALED_FAMILIES: tuple[str, ...] = HELDOUT2_FAMILIES
REPORT_SPLITS = ("TRAIN", "TEST_SEEN", "TEST_HELDOUT")  # swarm_r2 is scored here too; never decisive
NOT_CLEAN = {"TEST_HELDOUT": "the round-1 failure on this split was known while swarm_r2 was designed (D-033)"}
RUN_PROVIDERS = ("sarvam",)  # D-035: Gemini comes later, via --gemini-row
LATE_PROVIDER = "gemini"
EPISODE = (DECISIVE_SPLIT, SEALED_SEEDS[0])  # dashboard episode, chosen before any round-2 number existed


# ---------------------------------------------------------------- freeze of swarm_r2 ----------------------
def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def defender_hashes() -> dict[str, str]:
    files = [R2_CHAMPION_PATH, *sorted(p for d in DEFENDER_DIRS for p in d.glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): _digest(p) for p in files}


def freeze_defender() -> int:
    if FREEZE_PATH.exists():
        print(f"{FREEZE_PATH.relative_to(ROOT)} exists: swarm_r2 is already frozen.")
        return 2
    if not R2_CHAMPION_PATH.is_file():
        print(f"{R2_CHAMPION_PATH.relative_to(ROOT)} is missing: run the TRAIN-only selection first.")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"frozen_at": r1.now_iso(), "git_rev": r1.git_rev(), "files": defender_hashes()}
    FREEZE_PATH.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"froze swarm_r2: {len(doc['files'])} files -> {FREEZE_PATH.relative_to(ROOT)}  "
          f"champion sha256 {doc['files'][R2_CHAMPION_PATH.relative_to(ROOT).as_posix()]}")
    return 0


def require_frozen() -> dict:
    """D-034: no TEST_HELDOUT2 episode before swarm_r2 is frozen, and nothing about it may move afterwards."""
    if not FREEZE_PATH.is_file():
        raise SystemExit("swarm_r2 is not frozen (python -m scripts.final_eval_r2 --freeze-defender); "
                         "no sealed episode is built before that (D-034).")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))["files"]
    now = defender_hashes()
    problems = [f"changed: {k}" for k in now if k in frozen and now[k] != frozen[k]]
    problems += [f"not in freeze: {k}" for k in now if k not in frozen]
    problems += [f"missing: {k}" for k in frozen if k not in now]
    if problems:
        raise SystemExit("swarm_r2 moved after its freeze:\n  " + "\n  ".join(problems))
    return frozen


def make_r2():
    from payguard.robust.r2 import make_r2_defender  # imported late: the package lands with feat/robust
    return make_r2_defender(R2_CHAMPION_PATH)


def sealed_split_episodes():
    require_frozen()
    return sealed_episodes(SEALED_SEEDS, SEALED_FAMILIES)


# ---------------------------------------------------------------- payloads, plan, warm ----------------
def plan() -> int:
    freq, payload_of = r1.distinct_payloads(sealed_split_episodes())
    spend = r1.load_spend()
    print(f"{DECISIVE_SPLIT}: {len(freq)} distinct payloads over {sum(freq.values()):,} payouts (no call, no metric)")
    for provider in r1.PROVIDERS:
        a = make_assessor(provider, offline=True)
        missing = [c for c in freq if not r1.is_cached(a, payload_of[c])]
        share = 100 * sum(freq[c] for c in missing) / max(1, sum(freq.values()))
        print(f"  {provider:7s} (model {a.model}; ledger {r1.spend_calls(spend, provider)}/{r1.BUDGET[provider]}): "
              f"uncached {len(missing)}/{len(freq)} distinct = {share:.1f}% of payouts")
    return 0


def warm(provider: str, cap: int) -> int:
    spend = r1.load_spend()
    left = r1.BUDGET[provider] - r1.spend_calls(spend, provider)
    allowed = min(cap, left)
    if allowed <= 0:
        print(f"{provider}: budget used up ({r1.spend_calls(spend, provider)}/{r1.BUDGET[provider]}); nothing was called.")
        return 1
    freq, payload_of = r1.distinct_payloads(sealed_split_episodes())
    assessor = make_assessor(provider, max_calls=allowed, offline=False)
    if assessor.offline or assessor.client is None:
        print(f"{provider}: no API key in the environment; nothing was called.")
        return 1
    todo = [payload_of[c] for c in sorted(freq, key=lambda c: (-freq[c], c)) if not r1.is_cached(assessor, payload_of[c])]
    print(f"{provider}: {len(todo)} uncached distinct {DECISIVE_SPLIT} payloads; at most {allowed} calls "
          f"(ledger budget left {left}). Counts only are printed.", flush=True)
    t0 = time.time()
    try:
        for p in todo:
            if assessor.calls >= allowed or getattr(assessor, "quota_exhausted", False):
                break
            assessor.assess_payload(p)
            if assessor.calls and assessor.calls % 25 == 0:
                print(f"  calls={assessor.calls} errors={assessor.fallbacks.get('error', 0)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        errors = assessor.fallbacks.get("error", 0)
        r1.record_spend(provider, assessor.calls, errors)
        still = sum(1 for p in todo if not r1.is_cached(assessor, p))
        print(f"{provider}: calls={assessor.calls} errors={errors} still uncached={still}/{len(todo)} "
              f"quota_exhausted={bool(getattr(assessor, 'quota_exhausted', False))} ({time.time() - t0:.0f}s)")
    return 0


# ---------------------------------------------------------------- rows ---------------------------------
Rows = dict[tuple[str, str], Metrics]


def metrics_from_row(row: dict) -> Metrics:
    return Metrics(**{f.name: row[f.name] for f in fields(Metrics) if f.name in row})


def load_rows(path: Path) -> Rows:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {(r["defender"], r["split"]): metrics_from_row(r) for r in doc["rows"]}


def merge_rows(*sources: tuple[str, Rows]) -> Rows:
    """Combine the result rows of round 1, the round-2 one-shot and the late Gemini run into one table.

    `sources` are (label, rows) pairs in the order they were produced: ("round1", ...), ("round2", ...) and,
    once it exists, (LATE_PROVIDER, ...). The returned dict is what the dashboard shows. A later source never
    overwrites an earlier row, and the late source may add only its own judge on the decisive split (D-035).
    """
    allowed = {(f"llm_judge_{LATE_PROVIDER}", DECISIVE_SPLIT)}
    for label, rows in sources:
        if label == LATE_PROVIDER and (bad := set(rows) - allowed):
            raise ValueError(f"{label} may not add {sorted(bad)}")
    table = {}
    for label, rows in sources:
        for key, metrics in rows.items():
            if key in table:
                raise ValueError(f"{label} would overwrite {key}")
            table[key] = metrics
    return table


def result_row_dicts(rows: Rows) -> list[dict]:
    return [{"defender": n, "split": s, "objective": objective(m), **asdict(m)} for (n, s), m in rows.items()]


# ---------------------------------------------------------------- verdict ------------------------------
def go_no_go(metrics: Rows) -> dict:
    c = r1.check(metrics, DECISIVE_SPLIT, PRIMARY, BEST_SINGLE)
    secondary = {sw: r1.check(metrics, DECISIVE_SPLIT, sw, BEST_SINGLE)["pass"] for sw in r1.SWARMS}
    return {"criterion": f"D-022 via D-034: on {DECISIVE_SPLIT}, {PRIMARY} objective >= {MARGIN} below both "
                         f"static_rules and {BEST_SINGLE} (best single specialist on TRAIN), honest block <= "
                         f"static_rules'",
            "primary": PRIMARY, "best_single_on_train": BEST_SINGLE, "checks": {DECISIVE_SPLIT: c},
            "verdict": "GO" if c["pass"] else "NO-GO", "secondary_pass_not_decisive": secondary,
            "amendment": "D-035: llm_judge_gemini is added later by --gemini-row and cannot change this verdict"}


# ---------------------------------------------------------------- replay -------------------------------
_SEVERITY = {"RELEASE": 0.0, "STEP_UP": 0.5, "HOLD": 1.0}


class _Recorder:
    """swarm_r2's specialist scores (+ the layer's verdict as 0 / 0.5 / 1) and L1 weights at decision time."""

    def __init__(self, defender):
        self.defender, self.name, self.at = defender, defender.name, {}

    def decide(self, event, log):
        v = self.defender.decide(event, log)
        s = self.defender.last_signals
        sig = dict(s["scores"])
        if "layer" in s:
            sig["layer"] = _SEVERITY[s["layer"]["verdict"]]
        self.at[event.id] = (sig, dict(s["weights"]))
        return v

    def learn(self, event, is_attack):
        self.defender.learn(event, is_attack)


def episode_doc(episode: tuple[str, int] = EPISODE) -> dict:
    split, seed = episode
    assert split == DECISIVE_SPLIT
    require_frozen()
    ep = sealed_episode_for(seed)
    rec = _Recorder(make_r2())
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


def split_best_single(table: Rows, split: str) -> str | None:
    """The single specialist with the lowest objective ON `split`. Report-only (D-038): the rule's single is the
    one fixed on TRAIN (BEST_SINGLE), but a reader should see when another single comes close on the test itself."""
    singles = {n: m for (n, s), m in table.items() if s == split and n.startswith("single_")}
    return min(singles, key=lambda n: objective(singles[n])) if singles else None


def build_replay(table: Rows, generated_at: str, episode: dict) -> dict:
    shown = set(r1.display_names(BEST_SINGLE)) | {PRIMARY}
    extra = split_best_single(table, DECISIVE_SPLIT)  # shown on the decisive split only, next to BEST_SINGLE
    order = {s: i for i, s in enumerate(("TRAIN", "TEST_SEEN", "TEST_HELDOUT", DECISIVE_SPLIT))}
    rows = [rx.result_row(n, m, s) for (n, s), m in sorted(table.items(), key=lambda kv: order[kv[0][1]])
            if n in shown or (s == DECISIVE_SPLIT and n == extra)]
    lineage = json.loads(r1.LINEAGE_PATH.read_text(encoding="utf-8"))
    doc = rx.clean({"schema_version": 1, "simulated": True, "synthetic_fixture": False, "generated_at": generated_at,
                    "objective": rx.OBJECTIVE_TEXT, "results": rows, "evolution": rx.evolution_rows(lineage),
                    "episode": episode, "payouts": []})
    parse_replay(doc, "round-2 replay")  # the dashboard's contract is the final word
    size = len(json.dumps(doc, separators=(",", ":")))
    if size > rx.MAX_BYTES:
        raise SystemExit(f"replay would be {size:,} bytes (> {rx.MAX_BYTES:,}); nothing written")
    return doc


# ---------------------------------------------------------------- the one-shot -------------------------
def run_once() -> int:
    if RESULTS_PATH.exists():
        print(f"{RESULTS_PATH.relative_to(ROOT)} exists: the round-2 one-shot already ran. A rerun must be "
              "deliberate: delete the file and record why in docs/DECISIONS.md first.")
        return 2
    freeze = require_frozen()
    round1 = load_rows(r1.RESULTS_PATH)
    train = {n: m for (n, s), m in round1.items() if s == "TRAIN"}
    if rx.best_single(train) != BEST_SINGLE:
        raise SystemExit(f"best single on TRAIN is {rx.best_single(train)}, not the pre-registered {BEST_SINGLE}")
    r1_champion, _ = r1.load_champion()
    generated_at = r1.now_iso()
    metrics: Rows = {}
    llm: dict[str, dict] = {}
    t0 = time.time()

    eps = sealed_split_episodes()
    print(f"== {DECISIVE_SPLIT}: {len(eps)} episodes, {sum(len(e.events) for e in eps):,} payouts", flush=True)
    for name, make in [*r1.defender_factories(r1_champion), (PRIMARY, make_r2)]:
        metrics[(name, DECISIVE_SPLIT)] = m = run(make, eps)
        print(f"  {name:21s} obj={objective(m):6.2f}  {fmt(m)}", flush=True)
    for p in RUN_PROVIDERS:
        a = make_assessor(p, offline=True)  # cache-only: no API call can happen in the scoring pass
        name = f"llm_judge_{p}"
        metrics[(name, DECISIVE_SPLIT)] = m = run(lambda a=a, name=name: LLMJudge(a, name=name), eps)
        llm[p] = {DECISIVE_SPLIT: _llm_stats(a)}
        print(f"  {name:21s} obj={objective(m):6.2f}  {fmt(m)}  "
              f"fallback={llm[p][DECISIVE_SPLIT]['fallback_share_pct']:.2f}%", flush=True)
    for split in REPORT_SPLITS:
        metrics[(PRIMARY, split)] = m = run(make_r2, r1.split_episodes(split))
        note = f"  [not clean: {NOT_CLEAN[split]}]" if split in NOT_CLEAN else ""
        print(f"  {split:13s} {PRIMARY} obj={objective(m):6.2f}  {fmt(m)}{note}", flush=True)

    decision = go_no_go(metrics)
    results = {
        "generated_at": generated_at, "git_rev": r1.git_rev(), "simulated": True, "defender_freeze": freeze,
        "objective": rx.OBJECTIVE_TEXT, "pre_registration": ["D-034", "D-035"],
        "splits": {DECISIVE_SPLIT: {"seeds": [SEALED_SEEDS[0], SEALED_SEEDS[-1]], "families": list(SEALED_FAMILIES)}},
        "report_only": {s: NOT_CLEAN.get(s, "reported, not decisive") for s in REPORT_SPLITS},
        "go_no_go": decision, "llm": llm, "rows": result_row_dicts(metrics),
    }
    table = merge_rows(("round1", round1), ("round2", metrics))
    replay = build_replay(table, generated_at, episode_doc())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(rx.clean(results), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    size = rx.write_replay(r1.REPLAY_PATH, replay)
    c = decision["checks"][DECISIVE_SPLIT]
    print(f"{DECISIVE_SPLIT} {PRIMARY} obj={c['objective']:.2f}  vs static {c['margin_vs_static']:+.2f}  "
          f"vs {BEST_SINGLE} {c['margin_vs_best_single']:+.2f}  block {c['honest_block_rate']:.1f}% "
          f"(static {c['static_honest_block_rate']:.1f}%)  pass={c['pass']}")
    print(f"VERDICT ({DECISIVE_SPLIT} decides): {decision['verdict']}   secondary: "
          f"{decision['secondary_pass_not_decisive']}")
    print(f"wrote {RESULTS_PATH.relative_to(ROOT)} and {r1.REPLAY_PATH.relative_to(ROOT)} ({size:,} bytes) "
          f"in {time.time() - t0:.0f}s")
    return 0


def gemini_row() -> int:
    if not RESULTS_PATH.exists():
        print("run --run first: the Gemini row is added to an existing round-2 result (D-035).")
        return 2
    if GEMINI_PATH.exists():
        print(f"{GEMINI_PATH.relative_to(ROOT)} exists: the Gemini row was already added (run once).")
        return 2
    require_frozen()
    eps = sealed_split_episodes()
    a = make_assessor(LATE_PROVIDER, offline=True)  # cache-only, like every scoring pass
    name = f"llm_judge_{LATE_PROVIDER}"
    m = run(lambda: LLMJudge(a, name=name), eps)
    stats = _llm_stats(a)
    print(f"  {name:21s} obj={objective(m):6.2f}  {fmt(m)}  fallback={stats['fallback_share_pct']:.2f}%")
    late = {(name, DECISIVE_SPLIT): m}
    table = merge_rows(("round1", load_rows(r1.RESULTS_PATH)), ("round2", load_rows(RESULTS_PATH)),
                       (LATE_PROVIDER, late))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    replay = build_replay(table, results["generated_at"], episode_doc())
    GEMINI_PATH.write_text(json.dumps(rx.clean({"generated_at": r1.now_iso(), "git_rev": r1.git_rev(),
                                                "llm": {LATE_PROVIDER: {DECISIVE_SPLIT: stats}},
                                                "rows": result_row_dicts(late)}), indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    size = rx.write_replay(r1.REPLAY_PATH, replay)
    print(f"wrote {GEMINI_PATH.relative_to(ROOT)} and {r1.REPLAY_PATH.relative_to(ROOT)} ({size:,} bytes); "
          f"verdict unchanged: {results['go_no_go']['verdict']}")
    return 0


def _llm_stats(a) -> dict:
    return {"model": a.model, "cache_hits": a.cache_hits, "fallbacks": a.fallback_count,
            "fallback_share_pct": 100 * a.fallback_count / max(1, a.cache_hits + a.fallback_count)}


# ---------------------------------------------------------------- CLI ----------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze-defender", action="store_true", help="freeze swarm_r2 (once, before anything else)")
    g.add_argument("--plan", action="store_true", help="count uncached TEST_HELDOUT2 payloads (no calls)")
    g.add_argument("--warm", choices=r1.PROVIDERS, help="ask this provider about uncached payloads (needs --cap)")
    g.add_argument("--run", action="store_true", help="THE round-2 one-shot (refuses if results exist)")
    g.add_argument("--gemini-row", action="store_true", help="add the Gemini row after its quota reset (D-035)")
    ap.add_argument("--cap", type=int, help="calls allowed in this --warm run (also bounded by the ledger budget)")
    args = ap.parse_args(argv)
    if args.freeze_defender:
        return freeze_defender()
    if args.plan:
        return plan()
    if args.warm:
        if args.cap is None or args.cap < 1:
            ap.error("--warm needs --cap N (N >= 1)")
        return warm(args.warm, args.cap)
    if args.gemini_row:
        return gemini_row()
    return run_once()


if __name__ == "__main__":
    sys.exit(main())
