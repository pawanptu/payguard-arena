"""DREAM EVALUATOR: predicted objective of a defender on imagined attackers (P3).

Dream i is one imagined episode: the honest traffic of seed `seeds[i % len(seeds)]` (TRAIN_A in the loop) plus the
dream's campaigns, described by `arena_spec(seed, list(dream))`. The defender runs on it with a fresh swarm
(`swarm_job`, learned weights on, exactly like a real fitness episode). Per-dream objective is
`payguard.swarm.objective.objective`; the dream score is the reliability-weighted MEAN of those:

    objective = sum_i w_i * obj_i / sum_i w_i,       w_i = reliability.weight(dream_i, tau)  (all 1 if not use_skeptic)

`leakage_pct` and `honest_block_rate` of the score are the same weighted means of the per-dream components, so the
objective is a linear function of them (plus the unpaid-honest term, which `DreamScore` does not carry).

BATCHING DESIGN. `run_swarm_on` aggregates over a list of episodes and `Metrics` has no per-episode breakdown, so a
single job per candidate would lose the per-dream objectives the weights need. Instead `predict_many` sends ONE
`evaluator.run` batch with ONE JOB PER (distinct candidate, dream) pair, each job holding a single arena spec. Cost
versus one job per candidate: the same number of simulated episodes and the same episode builds (every worker has to
build the dream episodes it sees in either design; they are cached per process), but N_cand x N_dream pickled jobs
(each a ~1 KB tuple, results a small Metrics) instead of N_cand, i.e. a few hundred microseconds of IPC per job,
negligible next to a swarm replay. Candidates with identical parameters are evaluated once.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from ..evolve.evaluation import Evaluator, arena_spec, swarm_job
from ..swarm.objective import objective
from ..swarm.params import SwarmParams
from .contract import Dream, DreamScore, ReliabilityMap, assert_seed_allowed


def dream_weights(dreams: Sequence[Dream], reliability: ReliabilityMap, tau: float, use_skeptic: bool) -> tuple[float, ...]:
    if not use_skeptic:
        return tuple(1.0 for _ in dreams)
    return tuple(float(reliability.weight(d, tau)) for d in dreams)


def dream_specs(dreams: Sequence[Dream], seeds: Sequence[int]) -> list[tuple]:
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("need at least one seed")
    for s in seeds:
        assert_seed_allowed(s)
    return [arena_spec(seeds[i % len(seeds)], list(d)) for i, d in enumerate(dreams)]


def _score(ms: Sequence[Any], weights: tuple[float, ...]) -> DreamScore:
    per = tuple(float(objective(m)) for m in ms)
    tot = sum(weights)
    if not per or tot <= 0:
        return DreamScore(0.0, per, weights, 0.0, 0.0)
    wm = lambda xs: float(sum(w * x for w, x in zip(weights, xs)) / tot)  # noqa: E731
    return DreamScore(wm(per), per, weights, wm([m.leakage_pct for m in ms]), wm([m.honest_block_rate for m in ms]))


def predict_many(candidates: Sequence[SwarmParams], dreams: Sequence[Dream], reliability: ReliabilityMap, *,
                 seeds: Sequence[int], evaluator: Evaluator, tau: float = 1.0, use_skeptic: bool = True) -> list[DreamScore]:
    """DreamScore of every candidate (same order), from a single batch of jobs on `evaluator`."""
    specs = dream_specs(dreams, seeds)
    weights = dream_weights(dreams, reliability, tau, use_skeptic)
    keys = [tuple(sorted(asdict(c).items())) for c in candidates]
    distinct = list(dict.fromkeys(keys))
    first = {k: candidates[keys.index(k)] for k in distinct}
    jobs = [swarm_job(first[k], [sp], True) for k in distinct for sp in specs]
    results = evaluator.run(jobs) if jobs else []
    n = len(specs)
    by_key = {k: _score([results[j * n + i][0] for i in range(n)], weights) for j, k in enumerate(distinct)}
    return [by_key[k] for k in keys]


def predict(params: SwarmParams, dreams: Sequence[Dream], reliability: ReliabilityMap, *, seeds: Sequence[int],
            evaluator: Evaluator, tau: float = 1.0, use_skeptic: bool = True) -> DreamScore:
    return predict_many([params], dreams, reliability, seeds=seeds, evaluator=evaluator, tau=tau,
                        use_skeptic=use_skeptic)[0]
