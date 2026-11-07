"""Evaluate baseline (f), the swarm with shared memory, against the plain L1 swarm on the TRAIN split ONLY.

Run from the repo root:  python -m scripts.run_memory [--procs N]
Only TRAIN seeds are used. The memory parameters were chosen on the FIRST half of TRAIN (seeds 1000-1009);
the second half (1010-1019) is reported separately to show whether the result generalises within TRAIN.
All numbers are SIMULATED replays.

Rows: swarm_l1 (default params), swarm_memory (chosen MemoryParams), and one ablation row per entity kind
(that kind's memory alone; the other kinds' weights are set to 0, nothing else changes) plus
"memory minus requester" (the three account-bound kinds together).
`obj` = leak + 0.5*honest_block + 1.0*honest_unpaid (lower is better); `immunity` = mean attack payouts that
succeed in a campaign before it is contained; `never_contained` = share of campaigns never contained.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from payguard.sim.metrics import evaluate
from payguard.sim.runner import run_episode
from payguard.swarm.coordinator import make_swarm
from payguard.swarm.memory import KINDS, MemoryParams, default_memory_params, make_memory_swarm
from payguard.swarm.objective import objective
from scripts.run_baselines import fmt

HALVES = {
    "first half of TRAIN, seeds 1000-1009 (parameters were chosen here)": TRAIN_SEEDS[:10],
    "second half of TRAIN, seeds 1010-1019 (NOT used for choosing parameters)": TRAIN_SEEDS[10:],
    "full TRAIN, seeds 1000-1019": TRAIN_SEEDS,
}


def rows() -> list[tuple[str, MemoryParams | None]]:
    """(label, MemoryParams or None for the plain swarm)."""
    base = default_memory_params()
    off = {f"weight_{k}": 0.0 for k in KINDS}
    out: list[tuple[str, MemoryParams | None]] = [("swarm_l1", None), ("swarm_memory", base)]
    for k in KINDS:
        out.append((f"only_{k}", replace(base, **{**off, f"weight_{k}": base.__dict__[f"weight_{k}"]})))
    out.append(("memory_minus_requester", replace(base, weight_requester=0.0)))
    return out


def _job(args):
    idx, seed, mp = args
    ep = episode_for(seed, TRAIN_FAMILIES)
    return idx, seed, run_episode(ep, make_swarm() if mp is None else make_memory_swarm(None, mp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4, help="worker processes (default 4)")
    procs = max(1, min(ap.parse_args().procs, 4))
    table = rows()
    t0 = time.time()
    jobs = [(i, s, mp) for i, (_, mp) in enumerate(table) for s in TRAIN_SEEDS]
    with Pool(procs) as pool:
        done = pool.map(_job, jobs, chunksize=1)
    by_row: dict[int, dict[int, object]] = {}
    for i, seed, res in done:
        by_row.setdefault(i, {})[seed] = res
    print("   obj = leak + 0.5*honest_block + 1.0*honest_unpaid   (lower is better)")
    for title, seeds in HALVES.items():
        n_pay = sum(len(by_row[0][s].episode.events) for s in seeds)
        print(f"\n== TRAIN {title}: {len(seeds)} episodes, {n_pay:,} payouts")
        for i, (label, _) in enumerate(table):
            m = evaluate([by_row[i][s] for s in seeds])
            print(f"  {label:23s} obj={objective(m):6.2f}  {fmt(m)}")
    print(f"({time.time() - t0:.0f}s, {procs} processes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
