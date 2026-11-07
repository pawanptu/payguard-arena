"""Evaluate the swarm against the pre-committed baselines on the TRAIN split ONLY.

Run from the repo root:  python -m scripts.run_swarm
TEST_SEEN and TEST_HELDOUT are deliberately not imported here: they are touched once, at the end, by a
separate script, never while the swarm is being designed or tuned. All numbers are SIMULATED replays.

Rows: always_release / always_hold (bounds), static_rules (a), the four single specialists (b), the frozen
swarm (d: same specialists, initial weights never updated) and the L1 swarm (online reputation learning).
`obj` is the pre-registered scalar (lower is better); its three components are printed next to it.
"""
from __future__ import annotations

import sys
import time

from payguard.baselines.static_rules import StaticRules
from payguard.baselines.trivial import AlwaysHold, AlwaysRelease
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episodes
from payguard.swarm.coordinator import make_swarm
from payguard.swarm.objective import objective
from payguard.swarm.params import default_params
from payguard.swarm.single import SPECIALIST_NAMES, make_single
from scripts.run_baselines import fmt, run


def main() -> int:
    params = default_params()
    rows = [
        ("always_release", AlwaysRelease),
        ("always_hold", AlwaysHold),
        ("static_rules", StaticRules),
        *[(f"single_{n}", lambda n=n: make_single(n, params)) for n in SPECIALIST_NAMES],
        ("swarm_frozen", lambda: make_swarm(params, learn_weights=False)),
        ("swarm_l1", lambda: make_swarm(params, learn_weights=True)),
    ]
    eps = episodes(TRAIN_SEEDS, TRAIN_FAMILIES)
    print(f"== TRAIN (ring+structuring): {len(eps)} episodes, {sum(len(e.events) for e in eps):,} payouts")
    print("   obj = leak + 0.5*honest_block + 1.0*honest_unpaid   (lower is better)")
    t0 = time.time()
    for label, make in rows:
        m = run(make, eps)
        print(f"  {label:15s} obj={objective(m):6.2f}  {fmt(m)}")
    print(f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
