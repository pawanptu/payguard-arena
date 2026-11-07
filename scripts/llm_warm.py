"""Paid Sarvam (or free-tier Gemini) calls for the LLM defender, always under an explicit cap (D-025). TRAIN only.

Run from the repo root:
  python -m scripts.llm_warm --smoke 20     # 20 varied, not yet cached TRAIN payloads -> table of verdicts
  python -m scripts.llm_warm --warm 500     # replay the LLM judge over every TRAIN episode, <= 500 paid calls
  python -m scripts.llm_warm --provider gemini --warm 500   # same with the Gemini judge (own cache file)

Every validated reply goes to the cache (Sarvam: data/llm_cache.jsonl unless PAYGUARD_LLM_CACHE says otherwise;
Gemini: data/llm_cache_gemini.jsonl, env PAYGUARD_LLM_CACHE_GEMINI), so the warm-up reuses the smoke calls and
later runs replay offline. With --provider gemini progress is printed every 25 calls and the run stops cleanly
when the daily quota is exhausted (everything so far is cached; re-run later to resume). TEST_SEEN /
TEST_HELDOUT are never built here. Credentials are read by the client from the environment / .env and never
printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict

from payguard.llm.cache import cache_key
from payguard.llm.judge import PROMPT_VERSION, LLMAssessor, LLMJudge
from payguard.llm.judge import make_assessor as make_provider_assessor
from payguard.llm.payload import judge_payload
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episodes
from payguard.sim.features import PlatformLog
from payguard.swarm.objective import objective
from scripts.estimate_llm_cost import n_elevated
from scripts.run_baselines import fmt, run

PROGRESS_EVERY = 25
KEY_NAMES = {"sarvam": "SARVAM_API_KEY", "gemini": "GEMINI_API_KEY"}


class QuotaStop(Exception):
    """Raised inside a warm-up when the client reported an exhausted daily/project quota."""


def make_assessor(cap: int, provider: str = "sarvam") -> LLMAssessor:
    if provider == "sarvam":
        a = LLMAssessor(max_calls=cap, offline=False)
    else:
        a = make_provider_assessor(provider, max_calls=cap, offline=False)
    if a.offline or a.client is None:
        sys.exit(f"No {KEY_NAMES[provider]} in the environment: nothing was called.")
    return a


def scan_payloads(eps, assessor: LLMAssessor) -> tuple[Counter, dict]:
    """(frequency by cache key, payload by cache key) over the given episodes."""
    freq: Counter[str] = Counter()
    payload_of: dict[str, dict] = {}
    for ep in eps:
        log = PlatformLog(ep.accounts)
        for e in ep.events:
            p = judge_payload(e, log)
            log.add(e)
            k = cache_key(assessor.model, PROMPT_VERSION, p)
            freq[k] += 1
            payload_of[k] = p
    return freq, payload_of


def watch(assessor: LLMAssessor, total: int, *, every: int = PROGRESS_EVERY, clock=time.time, out=print) -> None:
    """Instrument `assessor` in place: print progress every `every` real calls (calls, cache hits, elapsed time,
    estimated time left for `total` expected calls) and raise QuotaStop as soon as the quota is exhausted."""
    inner = assessor.assess_payload
    t0 = clock()

    def assess_payload(payload):
        got = inner(payload)
        if assessor.quota_exhausted:
            raise QuotaStop
        if got.called and assessor.calls % every == 0:
            elapsed = clock() - t0
            left = max(total - assessor.calls, 0) * elapsed / assessor.calls
            out(f"progress: {assessor.calls}/{total} calls, {assessor.cache_hits} cache hits, "
                f"elapsed {elapsed / 60:.1f} min, ~{left / 60:.1f} min left")
        return got

    assessor.assess_payload = assess_payload  # type: ignore[method-assign]


def quota_message(assessor: LLMAssessor, provider: str) -> str:
    return (f"{provider} daily/project quota exhausted after {assessor.calls} calls. Everything made so far is "
            f"cached; re-run the same command later (after the quota resets) and it resumes from the cache.")


def varied_uncached(eps, assessor: LLMAssessor, n: int) -> list[dict]:
    """Up to `n` distinct payloads, spread over 0..5 elevated signals, most frequent first in each group."""
    freq, payload_of = scan_payloads(eps, assessor)
    groups: dict[int, list[str]] = defaultdict(list)
    for k in sorted(payload_of, key=lambda k: (-freq[k], k)):
        if assessor.cache.get(k) is None:
            groups[n_elevated(payload_of[k])].append(k)
    picked: list[str] = []
    while len(picked) < n and any(groups.values()):
        for lvl in sorted(groups):
            if groups[lvl] and len(picked) < n:
                picked.append(groups[lvl].pop(0))
    return [payload_of[k] for k in picked]


def smoke(n: int, provider: str = "sarvam", assessor: LLMAssessor | None = None) -> int:
    eps = episodes(TRAIN_SEEDS, TRAIN_FAMILIES)
    assessor = assessor if assessor is not None else make_assessor(n, provider)
    payloads = varied_uncached(eps, assessor, n)
    print(f"smoke: {len(payloads)} uncached TRAIN payloads, cap {n} paid calls, model {assessor.model}")
    if provider != "sarvam":
        watch(assessor, len(payloads))
    verdicts: Counter[str] = Counter()
    t0 = time.time()
    for i, p in enumerate(payloads, 1):
        try:
            a = assessor.assess_payload(p)
        except QuotaStop:
            print(quota_message(assessor, provider))
            return 0
        verdicts[a.verdict.value] += 1
        short = " ".join(f"{k.split('_')[0]}={v}" for k, v in p.items())
        print(f"{i:2d} elev={n_elevated(p)} {a.source:8s} {a.verdict.value:8s} risk={a.risk:.2f}  {short}")
        print(f"     why: {a.rationale[:160]}")
    print(f"verdicts: {dict(verdicts)}   stats: {assessor.stats()}   ({time.time() - t0:.0f}s)")
    return 0


def warm(cap: int, provider: str = "sarvam", assessor: LLMAssessor | None = None) -> int:
    eps = episodes(TRAIN_SEEDS, TRAIN_FAMILIES)
    assessor = assessor if assessor is not None else make_assessor(cap, provider)
    t0 = time.time()
    name = "llm_judge" if provider == "sarvam" else f"llm_judge_{provider}"
    if provider != "sarvam":
        freq, _ = scan_payloads(eps, assessor)
        todo = sum(1 for k in freq if assessor.cache.get(k) is None)
        total = min(cap, todo)
        print(f"warm: {len(freq)} distinct TRAIN payloads, {todo} not cached, cap {cap}, "
              f"model {assessor.model}; expect <= {total} calls")
        watch(assessor, total)
    try:
        m = run(lambda: LLMJudge(assessor, name=name), eps)
    except QuotaStop:
        print(quota_message(assessor, provider))
        print(f"stats: {assessor.stats()}   ({time.time() - t0:.0f}s)")
        return 0
    print(f"{name} TRAIN obj={objective(m):6.2f}  {fmt(m)}")
    print(f"stats: {assessor.stats()}   ({time.time() - t0:.0f}s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", type=int, metavar="N", help="N varied uncached payloads (cap = N)")
    g.add_argument("--warm", type=int, metavar="CAP", help="judge every TRAIN payout, at most CAP paid calls")
    ap.add_argument("--provider", choices=sorted(KEY_NAMES), default="sarvam",
                    help="which LLM judge to call (default sarvam: unchanged behaviour)")
    args = ap.parse_args()
    if args.smoke is not None:
        return smoke(args.smoke, args.provider)
    return warm(args.warm, args.provider)


if __name__ == "__main__":
    sys.exit(main())
