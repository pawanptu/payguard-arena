"""Run the pre-committed baselines on the frozen protocol and print the comparison table.

Run from the repo root:  python -m scripts.run_baselines
All numbers are SIMULATED replays on synthetic data, never real fraud statistics.
"""
from __future__ import annotations

import sys

from payguard.baselines.static_rules import StaticRules
from payguard.baselines.trivial import AlwaysHold, AlwaysRelease
from payguard.protocol import (
    HELDOUT_FAMILIES,
    TEST_HELDOUT_SEEDS,
    TEST_SEEN_SEEDS,
    TRAIN_FAMILIES,
    TRAIN_SEEDS,
    episodes,
)
from payguard.sim.metrics import Metrics, evaluate
from payguard.sim.runner import run_episode

SPLITS = {
    "TRAIN (ring+structuring)": (TRAIN_SEEDS, TRAIN_FAMILIES),
    "TEST_SEEN (ring+structuring)": (TEST_SEEN_SEEDS, TRAIN_FAMILIES),
    "TEST_HELDOUT (sleeper, unseen family)": (TEST_HELDOUT_SEEDS, HELDOUT_FAMILIES),
}


def run(defender_factory, eps) -> Metrics:
    return evaluate([run_episode(ep, defender_factory()) for ep in eps])


def fmt(m: Metrics) -> str:
    fam = " ".join(f"{k}={v:5.1f}%" for k, v in m.leakage_pct_by_family.items())
    return (f"leak={m.leakage_pct:5.1f}% (${m.leaked_usd:9,.0f}/${m.attack_usd:9,.0f})  "
            f"honest_block={m.honest_block_rate:4.1f}% unpaid={m.honest_unpaid_rate:4.1f}% "
            f"ttp_med={m.honest_median_ttp_h:4.1f}h  immunity={m.immunity_attempts:5.1f} "
            f"never_contained={m.never_contained_rate:5.1f}%  [{fam}]")


def main() -> int:
    factories = [AlwaysRelease, AlwaysHold, StaticRules]
    for split, (seeds, families) in SPLITS.items():
        eps = episodes(seeds, families)
        print(f"\n== {split}: {len(eps)} episodes, {sum(len(e.events) for e in eps):,} payouts")
        for make in factories:
            print(f"  {make.name:15s} {fmt(run(make, eps))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
