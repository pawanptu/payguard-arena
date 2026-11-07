"""Post-run evidence for the report. TRAIN seeds only.

  tables      protocol TRAIN_A / TRAIN_B objectives of the champion, the default swarm_l1 (and variants), static
              rules and the best single specialist
  attackers   did evolution make the attackers stronger? default swarm_l1 against arena episodes built from the
              INITIAL attacker pool (random draws) versus the FINAL pool, on TRAIN_A (selection seeds) and
              TRAIN_B (never used to select attackers); same for the champion
  robustness  champion vs default swarm_l1 on arena episodes from the FINAL pool on TRAIN_B seeds
"""
from __future__ import annotations

from dataclasses import asdict, replace

from ..swarm.params import SwarmParams, default_params
from . import attackers as atk
from .evaluation import Evaluator, arena_spec, proto_spec, swarm_job
from .loop import EvolveConfig, r, score_dict
from .replay_export import best_single
from .splits import FAMILIES

ARENA_SEEDS_PER_ATTACKER = 4


def pool_arena_specs(pools: atk.Pools, seeds: tuple[int, ...], per_attacker: int = ARENA_SEEDS_PER_ATTACKER) -> list[tuple]:
    """For each pool slot i and each of the first `per_attacker` seeds: one ring + one structuring campaign."""
    n = min(len(pools[f]) for f in FAMILIES)
    return [arena_spec(s, [(f, pools[f][i]) for f in FAMILIES]) for i in range(n) for s in seeds[:per_attacker]]


def changed_genes(params: dict, base: dict | None = None) -> dict:
    base = base or asdict(default_params())
    return {k: {"default": base[k], "champion": v} for k, v in params.items() if v != base[k]}


def final_analysis(cfg: EvolveConfig, ev: Evaluator, result: dict, metrics_all: dict) -> dict:
    champ = SwarmParams(**result["champion"]["params"])
    d = default_params()
    d_abstain = replace(d, l1_abstain=1)
    A, B = cfg.fitness_seeds, cfg.gate_seeds
    bs = best_single(metrics_all)
    single_name = bs.removeprefix("single_")

    # --- protocol TRAIN_A / TRAIN_B tables ---
    table_jobs: list[tuple[str, str, tuple]] = []
    for split, seeds in (("TRAIN_A", A), ("TRAIN_B", B)):
        specs = [proto_spec(s) for s in seeds]
        table_jobs += [
            (split, "swarm_evolved", swarm_job(champ, specs, True)),
            (split, "swarm_l1_default", swarm_job(d, specs, True)),
            (split, "swarm_frozen_default", swarm_job(d, specs, False)),
            (split, "swarm_l1_default+abstain", swarm_job(d_abstain, specs, True)),
            (split, "static_rules", ("static", tuple(specs))),
            (split, bs, ("single", single_name, asdict(d), tuple(specs))),
        ]

    # --- attacker evolution and robustness (arena episodes from the pools) ---
    init, final = result["initial_pools"], result["final_pools"]
    arena_jobs: list[tuple[str, str, tuple]] = []
    for split, seeds in (("TRAIN_A", A), ("TRAIN_B", B)):
        for pool_name, pools in (("initial_pool", init), ("final_pool", final)):
            specs = pool_arena_specs(pools, seeds)
            arena_jobs += [(f"{split}/{pool_name}", "swarm_l1_default", swarm_job(d, specs, True)),
                           (f"{split}/{pool_name}", "swarm_evolved", swarm_job(champ, specs, True))]

    res = ev.run([j for *_, j in table_jobs] + [j for *_, j in arena_jobs])
    out_tables: dict[str, dict] = {}
    for (split, name, _), (m, _w) in zip(table_jobs, res):
        out_tables.setdefault(split, {})[name] = score_dict(m)
    out_arena: dict[str, dict] = {}
    for (key, name, _), (m, _w) in zip(arena_jobs, res[len(table_jobs):]):
        out_arena.setdefault(key, {})[name] = score_dict(m)

    def gain(split: str, name: str) -> float:
        return r(out_arena[f"{split}/final_pool"][name]["leakage_pct"] - out_arena[f"{split}/initial_pool"][name]["leakage_pct"])

    lin = result["lineage"]
    first, last = lin[0]["strongest_attacker"], lin[-1]["strongest_attacker"]
    return {
        "champion": {"id": result["champion"]["id"], "generation_promoted": result["champion"]["generation"],
                     "changed_genes": changed_genes(result["champion"]["params"]),
                     "uses_l1_abstain": bool(result["champion"]["params"]["l1_abstain"])},
        "best_single": bs,
        "tables": out_tables,
        "all_train": {n: score_dict(m) for n, m in metrics_all.items()},
        "attacker_evolution": {
            "arena_episodes_per_pool": len(pool_arena_specs(final, A)),
            "arena": out_arena,
            "leakage_gain_vs_default_swarm_l1_pct_points": {s: gain(s, "swarm_l1_default") for s in ("TRAIN_A", "TRAIN_B")},
            "leakage_gain_vs_champion_pct_points": {s: gain(s, "swarm_evolved") for s in ("TRAIN_A", "TRAIN_B")},
            "strongest_attacker_leakage_gen0": {f: first[f]["leakage_pct"] for f in FAMILIES},
            "strongest_attacker_leakage_final_gen": {f: last[f]["leakage_pct"] for f in FAMILIES},
        },
        "robustness_final_pool_train_b": out_arena["TRAIN_B/final_pool"],
        "promotions": [e["generation"] for e in lin if e["promoted"]],
        "rollbacks": [e["generation"] for e in lin if e["rolled_back"]],
    }
