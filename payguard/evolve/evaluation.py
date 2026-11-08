"""Evaluation of candidate swarms, in parallel and deterministically.

`evaluate(params, episodes, learn_weights)` aggregates exactly the way `scripts.run_baselines.run` does: one
fresh defender per episode, then the frozen `payguard.sim.metrics.evaluate` over all runs of the set.

Episodes are described by hashable SPECS and built once per process (cache), so jobs shipped to worker
processes are tiny plain tuples/dicts (Windows `spawn` needs top-level functions and picklable arguments):

    ("proto", seed[, families])              protocol TRAIN episode (`protocol.episode_for`, both TRAIN families by default)
    ("arena", seed, ((family, items), ...))  `build_episode(seed, campaigns)`: the honest traffic of `seed`
                                             plus the given attack campaigns (items = sorted attack-param pairs)

A job is evaluated by one process from start to finish, and every random draw in the benchmark is keyed by
(seed, event id), never by process or call order, so results do not depend on the number of workers.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any

from ..baselines.static_rules import StaticRules
from ..baselines.trivial import AlwaysHold, AlwaysRelease
from ..protocol import TRAIN_FAMILIES, episode_for
from ..sim.attacks import PARAMS
from ..sim.engine import build_episode
from ..sim.metrics import Metrics
from ..sim.metrics import evaluate as aggregate
from ..sim.runner import run_episode
from ..sim.types import Episode
from ..swarm.coordinator import make_swarm
from ..swarm.params import SwarmParams
from ..swarm.single import make_single

Spec = tuple
MAX_ARENA_CACHE = 400

_EPISODES: dict[Spec, Episode] = {}


# --- specs ---
def proto_spec(seed: int, families: tuple[str, ...] | None = None) -> Spec:
    """Protocol TRAIN episode. The default families (ring + structuring) give the plain spec ("proto", seed), which
    keeps every existing lineage byte-for-byte; any other family tuple adds it as a third item."""
    if families is None or tuple(families) == TRAIN_FAMILIES:
        return ("proto", int(seed))
    return ("proto", int(seed), tuple(families))


def arena_spec(seed: int, campaigns: list[tuple[str, Any]]) -> Spec:
    """`campaigns` = [(family, attack params dataclass or dict)]."""
    cs = tuple((fam, tuple(sorted((asdict(p) if not isinstance(p, dict) else p).items()))) for fam, p in campaigns)
    return ("arena", int(seed), cs)


def get_episode(spec: Spec) -> Episode:
    """Build once per process, then serve from the cache."""
    ep = _EPISODES.get(spec)
    if ep is None:
        if spec[0] == "proto":
            ep = episode_for(spec[1], spec[2] if len(spec) > 2 else TRAIN_FAMILIES)
        else:
            ep = build_episode(spec[1], [(fam, PARAMS[fam](**dict(items))) for fam, items in spec[2]])
            arena = [k for k in _EPISODES if k[0] == "arena"]
            for k in arena[: max(0, len(arena) - MAX_ARENA_CACHE + 1)]:
                del _EPISODES[k]
        _EPISODES[spec] = ep
    return ep


def _init_worker(preload_seeds: tuple[int, ...]) -> None:
    for s in preload_seeds:
        get_episode(proto_spec(s))


# --- evaluation ---
def run_swarm_on(params: SwarmParams, episodes: list[Episode], learn_weights: bool = True) -> tuple[Metrics, dict]:
    """Metrics over the set, and the coordinator weights at the end of the LAST episode's run."""
    results, weights = [], {}
    for ep in episodes:
        swarm = make_swarm(params, learn_weights)
        results.append(run_episode(ep, swarm))
        weights = dict(swarm.weights)
    return aggregate(results), weights


def evaluate(params: SwarmParams, episodes: list[Episode], learn_weights: bool = True) -> Metrics:
    """Same aggregation as `scripts.run_baselines.run(lambda: make_swarm(params, learn_weights), episodes)`."""
    return run_swarm_on(params, episodes, learn_weights)[0]


def run_job(job: tuple) -> tuple[Metrics, dict | None]:
    """Top-level (picklable) entry point of a worker process. Job kinds:
    ("swarm", params_dict, learn, specs) | ("single", name, params_dict, specs) | ("static"|"always_release"|
    "always_hold", specs)."""
    kind = job[0]
    if kind == "swarm":
        _, pd, learn, specs = job
        return run_swarm_on(SwarmParams(**pd), [get_episode(s) for s in specs], learn)
    if kind == "single":
        _, name, pd, specs = job
        make = lambda: make_single(name, SwarmParams(**pd))  # noqa: E731
    else:
        specs = job[1]
        make = {"static": StaticRules, "always_release": AlwaysRelease, "always_hold": AlwaysHold}[kind]
    return aggregate([run_episode(get_episode(s), make()) for s in specs]), None


def swarm_job(params: SwarmParams, specs: list[Spec] | tuple[Spec, ...], learn_weights: bool = True) -> tuple:
    return ("swarm", asdict(params), bool(learn_weights), tuple(specs))


class Evaluator:
    """Runs batches of jobs on a process pool (or inline with workers <= 1); results come back in job order."""

    def __init__(self, workers: int = 1, preload_seeds: tuple[int, ...] = ()):
        self.workers = max(1, int(workers))
        self.preload = tuple(preload_seeds)
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "Evaluator":
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(self.workers, initializer=_init_worker, initargs=(self.preload,))
        else:
            _init_worker(self.preload)
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def run(self, jobs: list[tuple]) -> list[tuple[Metrics, dict | None]]:
        if self._pool is None:
            return [run_job(j) for j in jobs]
        return list(self._pool.map(run_job, jobs, chunksize=1))
