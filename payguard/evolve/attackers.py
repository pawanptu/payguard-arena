"""Attacker co-evolution: a small pool of attack parameters per TRAIN family, mutated every generation.

Each generation, per family: the pool (parents) + one mutant per parent + `fresh` new uniform draws are each
played alone against the CURRENT champion (L1 swarm) on `seeds` TRAIN_A seeds, as arena episodes
`build_episode(seed, [(family, params)])`. An attacker's score is the USD it gets paid out (leaked_usd). The
strongest `pool_size` survive. The same seed with other attacker params gives the same honest traffic.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from ..sim.attacks import mutate as mutate_attacker
from ..sim.attacks import sample_params as sample_attacker
from ..swarm.params import SwarmParams
from .evaluation import Spec, arena_spec, swarm_job
from .splits import FAMILIES

Pools = dict[str, list[Any]]   # family -> attack params (dataclasses), strongest first once scored


def initial_pools(rng: np.random.Generator, pool_size: int, families: tuple[str, ...] = FAMILIES) -> Pools:
    return {fam: [sample_attacker(fam, rng) for _ in range(pool_size)] for fam in families}


def candidate_attackers(pools: Pools, rng: np.random.Generator, scale: float, fresh: int,
                        families: tuple[str, ...] = FAMILIES) -> list[tuple[str, Any]]:
    """Parents first, then their mutants, then fresh draws; deterministic order. Only `families` are attacked."""
    out: list[tuple[str, Any]] = []
    for fam in families:
        parents = pools[fam]
        out += [(fam, p) for p in parents]
        out += [(fam, mutate_attacker(p, fam, rng, scale)) for p in parents]
        out += [(fam, sample_attacker(fam, rng)) for _ in range(fresh)]
    return out


def scoring_jobs(champion: SwarmParams, cands: list[tuple[str, Any]], seeds: list[int]) -> list[tuple]:
    return [swarm_job(champion, [arena_spec(s, [(fam, p)]) for s in seeds], True) for fam, p in cands]


def select_strongest(cands: list[tuple[str, Any]], results: list, pool_size: int,
                     families: tuple[str, ...] = FAMILIES) -> tuple[Pools, dict[str, list[dict]]]:
    """Keep the `pool_size` highest-leaking attackers of each family (stable: ties keep candidate order).
    Returns the new pools and, per family, the survivors' score records (strongest first)."""
    pools: Pools = {}
    records: dict[str, list[dict]] = {}
    for fam in families:
        rows = [(p, m) for (f, p), (m, _w) in zip(cands, results) if f == fam]
        rows.sort(key=lambda r: -r[1].leaked_usd)
        rows = rows[:pool_size]
        pools[fam] = [p for p, _ in rows]
        records[fam] = [attacker_record(fam, p, m) for p, m in rows]
    return pools, records


def attacker_record(family: str, params: Any, m) -> dict:
    return {"family": family, "params": asdict(params), "leaked_usd": round(m.leaked_usd, 2),
            "attack_usd": round(m.attack_usd, 2), "leakage_pct": round(m.leakage_pct, 4)}


def arena_specs(pools: Pools, seeds: tuple[int, ...], generation: int, n: int,
                families: tuple[str, ...] = FAMILIES) -> list[Spec]:
    """`n` arena episodes: episode i has one campaign per family (ring and structuring by default), taken from the
    i-th strongest attacker of each pool (wrapping), on seed seeds[(generation * n + i) mod len(seeds)]."""
    return [arena_spec(seeds[(generation * n + i) % len(seeds)],
                       [(fam, pools[fam][i % len(pools[fam])]) for fam in families]) for i in range(n)]
