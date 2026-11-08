"""Leave-one-family-out (LOFO) folds on TRAIN, an episode cache and a small process pool.

Hard guard: the only seeds and families this module can ever build are the protocol's TRAIN ones
(`payguard.protocol.TRAIN_SEEDS`, `TRAIN_FAMILIES`). `get_episode` raises for anything else, so a typo cannot
quietly pull a test split into a fit.

Fold layout (see eval/r2/SELECTION_RULE.md): the honest traffic of an episode depends only on its seed, so a fold
is out-of-family AND out-of-seed:  fit on family A with the seeds of one TRAIN half, evaluate on family B with the
seeds of the other half.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from ..protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from ..sim.metrics import Metrics
from ..sim.metrics import evaluate as aggregate
from ..sim.runner import run_episode
from ..sim.types import Episode

HALVES: tuple[tuple[int, ...], tuple[int, ...]] = (tuple(TRAIN_SEEDS[:10]), tuple(TRAIN_SEEDS[10:]))
DIRECTIONS: tuple[tuple[str, str], ...] = (("ring", "structuring"), ("structuring", "ring"))

Spec = tuple  # ("fam", seed, (family, ...))
_EPISODES: dict[Spec, Episode] = {}


def fam_spec(seed: int, families: tuple[str, ...]) -> Spec:
    return ("fam", int(seed), tuple(families))


def check_spec(spec: Spec) -> None:
    _, seed, families = spec
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"seed {seed} is not a TRAIN seed; swarm_r2 never builds other splits")
    if not families or not set(families) <= set(TRAIN_FAMILIES):
        raise ValueError(f"families {families} are not all TRAIN families {TRAIN_FAMILIES}")


def get_episode(spec: Spec) -> Episode:
    ep = _EPISODES.get(spec)
    if ep is None:
        check_spec(spec)
        ep = _EPISODES[spec] = episode_for(spec[1], spec[2])
    return ep


@dataclass(frozen=True)
class Fold:
    direction: str            # "ring->structuring"
    half: int                 # the half the fit seeds come from
    fit: tuple[Spec, ...]
    evaluate: tuple[Spec, ...]

    @property
    def tag(self) -> str:
        return f"{self.direction}@h{self.half}"


def lofo_folds() -> list[Fold]:
    folds = []
    for a, b in DIRECTIONS:
        for h in (0, 1):
            folds.append(Fold(f"{a}->{b}", h,
                              tuple(fam_spec(s, (a,)) for s in HALVES[h]),
                              tuple(fam_spec(s, (b,)) for s in HALVES[1 - h])))
    return folds


def full_train_specs() -> tuple[Spec, ...]:
    """The protocol TRAIN episodes: 20 seeds, both families mixed (same as `protocol.episodes`)."""
    return tuple(fam_spec(s, tuple(TRAIN_FAMILIES)) for s in TRAIN_SEEDS)


# ---------------------------------------------------------------- defenders from plain-dict specs --------
def build_defender(dspec: dict):
    """dspec: {"kind": "static"} | {"kind": "single", "name"} | {"kind": "swarm", "params", "learn"} |
    {"kind": "evo_mem", "params"} | {"kind": "r2", "params", "layer"}  (params = SwarmParams as dict)."""
    from ..baselines.static_rules import StaticRules
    from ..swarm.coordinator import make_swarm
    from ..swarm.memory import make_memory_swarm
    from ..swarm.params import SwarmParams, default_params
    from ..swarm.single import make_single
    from .layer import make_r2

    kind = dspec["kind"]
    if kind == "static":
        return StaticRules()
    if kind == "single":
        return make_single(dspec["name"], default_params())
    if kind == "swarm":
        return make_swarm(SwarmParams(**dspec["params"]) if dspec.get("params") else default_params(),
                          dspec.get("learn", True))
    if kind == "evo_mem":
        return make_memory_swarm(SwarmParams(**dspec["params"]), None, name="swarm_evolved_memory")
    if kind == "r2":
        return make_r2(SwarmParams(**dspec["params"]) if dspec.get("params") else default_params(),
                       dspec.get("layer"))
    raise ValueError(kind)


_STATS: dict[Spec, tuple] = {}


def run_job(job: tuple):
    """Top-level (picklable) entry of a worker process.
    ("metrics", dspec, specs) -> Metrics over those episodes;   ("stats", specs) -> (X, is_attack) layer statistics."""
    if job[0] == "metrics":
        _, dspec, specs = job
        return aggregate([run_episode(get_episode(s), build_defender(dspec)) for s in specs])
    if job[0] == "stats":
        import numpy as np

        from .layer import collect_stats
        xs, ys = [], []
        for s in job[1]:
            if s not in _STATS:
                _STATS[s] = collect_stats(get_episode(s))
            x, y = _STATS[s]
            xs.append(x)
            ys.append(y)
        return np.vstack(xs), np.concatenate(ys)
    raise ValueError(job[0])


class Runner:
    """Process pool (or inline for workers <= 1); results come back in job order. Deterministic: every random draw
    of the benchmark is keyed by (seed, event id), not by process."""

    def __init__(self, workers: int = 1):
        self.workers = max(1, int(workers))
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "Runner":
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(self.workers)
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def map(self, jobs: list[tuple]) -> list:
        if self._pool is None:
            return [run_job(j) for j in jobs]
        return list(self._pool.map(run_job, jobs, chunksize=1))

    def metrics(self, dspec: dict, specs) -> Metrics:
        return self.map([("metrics", dspec, tuple(specs))])[0]
