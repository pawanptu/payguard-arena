"""Offline LLM cost estimate. Makes NO API calls and reads no credentials.

Run from the repo root:  python -m scripts.estimate_llm_cost

Replays the TRAIN episodes only (TRAIN_SEEDS x TRAIN_FAMILIES; TEST_SEEN / TEST_HELDOUT are never touched),
builds the LLM payload for every payout and counts how many distinct cache keys (= paid API calls on a
cold cache) each use would need:
  judge       the single LLM judge decides every payout       -> unique keys over all payouts
  borderline  the swarm consults the LLM only on borderline    -> unique keys over the borderline subset
A payout is "borderline" here if ANY per-account bucket sits in its ELEVATED range (young account, large
amount, many payouts in 24 h, far above own median, shared device/destination).
The real swarm will pick its own borderline cases; this is a simple upper-bound proxy.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from payguard.llm.cache import cache_key
from payguard.llm.client import DEFAULT_MODEL
from payguard.llm.judge import PROMPT_VERSION, SYSTEM_PROMPT, fallback_risk
from payguard.llm.payload import LABELS, PER_ACCOUNT_FIELDS, judge_payload, level
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episodes
from payguard.sim.features import PlatformLog

# Per-account / sharing buckets counted as elevated (ordinal positions in payload.LABELS).
ELEVATED = (
    ("account_age", lambda lv: lv == 0),            # < 7 days
    ("amount_usd", lambda lv: lv >= 2),             # >= $250
    ("payouts_24h", lambda lv: lv >= 2),            # 5+ in 24 h
    ("amount_vs_own_median", lambda lv: lv >= 1),   # >= 2x own median
    ("workers_sharing_device_or_destination", lambda lv: lv >= 1),
)
PRICE_NOTE = "Multiply calls by your Sarvam per-token price; no price is assumed here."


def n_elevated(payload: dict) -> int:
    return sum(bool(pred(level(payload, f))) for f, pred in ELEVATED)


def is_borderline(payload: dict) -> bool:
    return n_elevated(payload) >= 1


def main() -> int:
    eps = episodes(TRAIN_SEEDS, TRAIN_FAMILIES)
    total = 0
    keys_all: Counter[str] = Counter()
    keys_border: Counter[str] = Counter()
    n_border = 0
    keys_multi: Counter[str] = Counter()
    n_multi = 0
    would_fallback = Counter()
    sample_chars = len(SYSTEM_PROMPT)
    for ep in eps:
        log = PlatformLog(ep.accounts)
        for e in ep.events:
            p = judge_payload(e, log)  # before log.add(e)
            log.add(e)
            k = cache_key(DEFAULT_MODEL, PROMPT_VERSION, p)
            total += 1
            keys_all[k] += 1
            if is_borderline(p):
                n_border += 1
                keys_border[k] += 1
            if n_elevated(p) >= 2:
                n_multi += 1
                keys_multi[k] += 1
            would_fallback[fallback_risk(p) >= 0.3] += 1
            sample_chars = len(SYSTEM_PROMPT) + len(json.dumps(p, sort_keys=True))
    print(f"TRAIN episodes: {len(eps)} (seeds {TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}, families {', '.join(TRAIN_FAMILIES)})")
    print(f"model={DEFAULT_MODEL} prompt_version={PROMPT_VERSION}")
    print(f"payouts                         : {total}")
    print(f"unique cache keys, judge (all)  : {len(keys_all)}  ({100 * len(keys_all) / total:.2f}% of payouts)")
    print(f"borderline payouts              : {n_border}  ({100 * n_border / total:.1f}% of payouts)")
    print(f"unique cache keys, borderline   : {len(keys_border)}")
    print(f"2+ elevated signals             : {n_multi}  ({100 * n_multi / total:.1f}% of payouts)")
    print(f"unique cache keys, 2+ elevated  : {len(keys_multi)}")
    print(f"payload fields                  : {len(LABELS)} ({len(PER_ACCOUNT_FIELDS)} per-account, "
          f"{len(LABELS) - len(PER_ACCOUNT_FIELDS)} graph)")
    print(f"approx prompt size per call     : {sample_chars} chars, ~{sample_chars // 4} input tokens "
          f"(+ ~80 output tokens)")
    top = keys_all.most_common(1)[0][1]
    print(f"most shared key                 : {top} payouts; median payouts per key "
          f"{sorted(keys_all.values())[len(keys_all) // 2]}")
    print("Cold-cache API calls needed: judge = unique keys (all); swarm Quality = unique keys (borderline).")
    print("Paid calls are opt-in: PAYGUARD_LLM_MAX_CALLS defaults to 0; beyond the cap the deterministic fallback decides.")
    print(PRICE_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
