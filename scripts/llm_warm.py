"""Paid Sarvam calls for the LLM defender, always under an explicit cap (D-025). TRAIN split only.

Run from the repo root:
  python -m scripts.llm_warm --smoke 20     # 20 varied, not yet cached TRAIN payloads -> table of verdicts
  python -m scripts.llm_warm --warm 500     # replay the LLM judge over every TRAIN episode, <= 500 paid calls

Every validated reply goes to the cache (data/llm_cache.jsonl unless PAYGUARD_LLM_CACHE says otherwise), so
the warm-up reuses the smoke calls and later runs replay offline. TEST_SEEN / TEST_HELDOUT are never built
here. Credentials are read by the client from the environment / .env and never printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict

from payguard.llm.cache import cache_key
from payguard.llm.judge import PROMPT_VERSION, LLMAssessor, LLMJudge
from payguard.llm.payload import judge_payload
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episodes
from payguard.sim.features import PlatformLog
from payguard.swarm.objective import objective
from scripts.estimate_llm_cost import n_elevated
from scripts.run_baselines import fmt, run


def make_assessor(cap: int) -> LLMAssessor:
    a = LLMAssessor(max_calls=cap, offline=False)
    if a.offline or a.client is None:
        sys.exit("No SARVAM_API_KEY in the environment: nothing was called.")
    return a


def varied_uncached(eps, assessor: LLMAssessor, n: int) -> list[dict]:
    """Up to `n` distinct payloads, spread over 0..5 elevated signals, most frequent first in each group."""
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


def smoke(n: int) -> int:
    eps = episodes(TRAIN_SEEDS, TRAIN_FAMILIES)
    assessor = make_assessor(n)
    payloads = varied_uncached(eps, assessor, n)
    print(f"smoke: {len(payloads)} uncached TRAIN payloads, cap {n} paid calls, model {assessor.model}")
    verdicts: Counter[str] = Counter()
    t0 = time.time()
    for i, p in enumerate(payloads, 1):
        a = assessor.assess_payload(p)
        verdicts[a.verdict.value] += 1
        short = " ".join(f"{k.split('_')[0]}={v}" for k, v in p.items())
        print(f"{i:2d} elev={n_elevated(p)} {a.source:8s} {a.verdict.value:8s} risk={a.risk:.2f}  {short}")
        print(f"     why: {a.rationale[:160]}")
    print(f"verdicts: {dict(verdicts)}   stats: {assessor.stats()}   ({time.time() - t0:.0f}s)")
    return 0


def warm(cap: int) -> int:
    eps = episodes(TRAIN_SEEDS, TRAIN_FAMILIES)
    assessor = make_assessor(cap)
    t0 = time.time()
    m = run(lambda: LLMJudge(assessor), eps)
    print(f"llm_judge TRAIN obj={objective(m):6.2f}  {fmt(m)}")
    print(f"stats: {assessor.stats()}   ({time.time() - t0:.0f}s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", type=int, metavar="N", help="N varied uncached payloads (cap = N)")
    g.add_argument("--warm", type=int, metavar="CAP", help="judge every TRAIN payout, at most CAP paid calls")
    args = ap.parse_args()
    if args.smoke is not None:
        return smoke(args.smoke)
    return warm(args.warm)


if __name__ == "__main__":
    sys.exit(main())
