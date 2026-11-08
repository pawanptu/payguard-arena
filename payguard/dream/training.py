"""Round-3 training: the dreamer defender, evolved by the L2 loop with the dream engine switched on (P5).

`train(cfg, families, evo_seed) -> SwarmParams` is the function `scripts/r3_dev_check.py` injects (default
`payguard.dream.training:train`). It evolves a defender on `families` ONLY (fitness TRAIN_A, gate TRAIN_B, both TRAIN
seeds) with `DreamEngine` as the loop's `hooks`, and returns the champion's parameters. `train_run` returns the whole
result (lineage, per-generation calibration gap EMA for `build_freeze_doc`, timing).

Knobs:
  * `EVOLVE_CONFIG`: the evolution settings (module constant, D-028 full config by default: population 28,
    22 generations). A reduced configuration for a dev run is set by assigning this constant, or passed as
    `evolve_cfg=`; the director applies any reduction to every arm alike.
  * `workers`: evaluation processes. Default: env `PAYGUARD_WORKERS`, else `os.cpu_count() - 2` (at least 1).
    Results do not depend on it (every job is a pure function of its spec).

Leave-one-family-out: `families` restricts the attacker pools, the arena episodes, the protocol fitness/gate
episodes and the dream campaigns (`DreamConfig.families` is forced to `families`), so a held-out family is never
attacked, observed or dreamed during training.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from functools import partial
from typing import Any

from ..evolve.evaluation import Evaluator
from ..evolve.loop import DreamLoopConfig, EvolveConfig, run_evolution
from ..evolve.splits import TRAIN_A
from ..protocol import TRAIN_FAMILIES
from ..swarm.params import SwarmParams
from .contract import DreamConfig, assert_seed_allowed
from .engine import DreamEngine
from .skeptic import HoldoutSkeptic
from .worldmodel import AbcWorldModel

WORKERS_ENV = "PAYGUARD_WORKERS"
EVOLVE_CONFIG: EvolveConfig = EvolveConfig(pop=28, gens=22, name="full")   # D-028 full config (pop 28, 22 generations)


def default_workers() -> int:
    env = os.environ.get(WORKERS_ENV, "").strip()
    if env:
        return max(1, int(env))
    return max(1, (os.cpu_count() or 4) - 2)


def train_run(cfg: DreamConfig, families: Sequence[str], evo_seed: int, *, evolve_cfg: EvolveConfig | None = None,
              workers: int | None = None, evaluator: Evaluator | None = None,
              loop_cfg: DreamLoopConfig = DreamLoopConfig(), log: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
    """Full result of one dream-trained evolution on `families` with evolution seed `evo_seed`."""
    fams = tuple(families)
    if not fams or any(f not in TRAIN_FAMILIES for f in fams) or len(set(fams)) != len(fams):
        raise ValueError(f"families must be distinct TRAIN families {TRAIN_FAMILIES}, got {fams}")
    base = EVOLVE_CONFIG if evolve_cfg is None else evolve_cfg
    ecfg = replace(base, seed=int(evo_seed), families=fams)
    dcfg = replace(cfg, families=fams)                       # dreams can never be drawn from a held-out family
    for s in ecfg.fitness_seeds + ecfg.gate_seeds:
        assert_seed_allowed(s)
    engine = DreamEngine(dcfg, make_model=partial(AbcWorldModel, families=fams), skeptic=HoldoutSkeptic(),
                         seeds=TRAIN_A, base_seed=int(evo_seed))
    preload = tuple(ecfg.fitness_seeds) + tuple(ecfg.gate_seeds)

    def go(ev: Evaluator) -> dict:
        return run_evolution(ecfg, ev, log=log, hooks=engine, loop_cfg=loop_cfg)

    if evaluator is not None:
        result, n_workers = go(evaluator), evaluator.workers
    else:
        n_workers = workers if workers is not None else default_workers()
        with Evaluator(n_workers, preload) as ev:
            result = go(ev)

    lin = result["lineage"]
    champion = dict(result["champion"])
    return {
        "families": list(fams),
        "evo_seed": int(evo_seed),
        "workers": int(n_workers),
        "evolve_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(ecfg).items()},
        "dream_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(dcfg).items()},
        "champion": champion,
        "champion_params": SwarmParams(**champion["params"]),
        "calibration_gap_ema": [float(e["dream"]["gap_ema"]) for e in lin],   # one per generation, for H3
        "lineage": lin,
        "generations_run": result["generations_run"],
        "stopped_early": result["stopped_early"],
        "seconds": result["seconds"],
        "gen_seconds": result["gen_seconds"],
    }


def train(cfg: DreamConfig, families: tuple[str, ...], evo_seed: int) -> SwarmParams:
    """The champion's SwarmParams after a dream-trained evolution on `families` (see module docstring)."""
    return train_run(cfg, families, evo_seed)["champion_params"]
