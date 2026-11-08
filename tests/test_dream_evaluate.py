"""Dream evaluator (P3): weighting, determinism, seed guard, batching. Dev seeds 9300-9399 only."""
import math
from dataclasses import replace

import numpy as np
import pytest

from payguard.dream import DreamScore, FlatReliability, uniform_dreams
from payguard.dream.evaluate import dream_specs, predict, predict_many
from payguard.evolve.evaluation import Evaluator, arena_spec, run_job, swarm_job
from payguard.sim.attacks import RingParams, StructuringParams
from payguard.swarm.objective import objective
from payguard.swarm.params import SwarmParams

SEEDS = (9300, 9301)
DEFAULT = SwarmParams()
STRICT = replace(DEFAULT, step_up_threshold=0.1, hold_threshold=0.55)
DREAMS = uniform_dreams(4, np.random.default_rng(7)) + [
    (("ring", RingParams()), ("structuring", StructuringParams()))]


class HalfRel(FlatReliability):
    """Weight 1 for dreams starting with a ring campaign, 0.25 otherwise."""

    def weight(self, dream, tau):
        return 1.0 if dream[0][0] == "ring" else 0.25


@pytest.fixture(scope="module")
def ev():
    with Evaluator(workers=1) as e:
        yield e


def test_matches_manual_jobs_and_weighting(ev):
    score = predict(DEFAULT, DREAMS, HalfRel(), seeds=SEEDS, evaluator=ev, tau=1.0, use_skeptic=True)
    assert isinstance(score, DreamScore) and len(score.per_dream) == len(DREAMS)
    for i, d in enumerate(DREAMS):  # per-dream objective = objective of that one swarm_job
        m, _ = run_job(swarm_job(DEFAULT, [arena_spec(SEEDS[i % 2], list(d))], True))
        assert math.isclose(score.per_dream[i], objective(m))
    w = tuple(1.0 if d[0][0] == "ring" else 0.25 for d in DREAMS)
    assert score.weights == w
    assert math.isclose(score.objective, sum(a * b for a, b in zip(w, score.per_dream)) / sum(w))
    assert 0.0 <= score.leakage_pct <= 100.0 and 0.0 <= score.honest_block_rate <= 100.0


def test_noskeptic_ignores_reliability(ev):
    flat = predict(DEFAULT, DREAMS, HalfRel(), seeds=SEEDS, evaluator=ev, use_skeptic=False)
    assert flat.weights == (1.0,) * len(DREAMS)
    assert math.isclose(flat.objective, float(np.mean(flat.per_dream)))
    skep = predict(DEFAULT, DREAMS, HalfRel(), seeds=SEEDS, evaluator=ev, use_skeptic=True)
    assert skep.per_dream == flat.per_dream and not math.isclose(skep.objective, flat.objective)


def test_predict_many_equals_predict_dedupes_and_is_deterministic(ev):
    cands = [DEFAULT, STRICT, DEFAULT]
    many = predict_many(cands, DREAMS[:3], FlatReliability(), seeds=SEEDS, evaluator=ev)
    assert len(many) == 3 and many[0] == many[2]
    single = predict(STRICT, DREAMS[:3], FlatReliability(), seeds=SEEDS, evaluator=ev)
    assert single == many[1]
    assert many[0].per_dream != many[1].per_dream  # different defenders score differently
    assert predict_many(cands, DREAMS[:3], FlatReliability(), seeds=SEEDS, evaluator=ev) == many


def test_one_batch_of_candidates_times_dreams():
    calls = []

    class Spy:
        def run(self, jobs):
            calls.append(len(jobs))
            return Evaluator(1).run(jobs)

    predict_many([DEFAULT, STRICT, DEFAULT], DREAMS[:2], FlatReliability(), seeds=SEEDS, evaluator=Spy())
    assert calls == [2 * 2]  # ONE call, distinct candidates x dreams


def test_seed_guard_and_specs():
    with pytest.raises(AssertionError):
        dream_specs(DREAMS, (9300, 2000))
    with pytest.raises(ValueError):
        dream_specs(DREAMS, ())
    sp = dream_specs(DREAMS, SEEDS)
    assert [s[1] for s in sp] == [9300, 9301, 9300, 9301, 9300]
    assert sp[4] == arena_spec(9300, list(DREAMS[4]))


def test_empty_dreams_and_candidates(ev):
    s = predict(DEFAULT, [], FlatReliability(), seeds=SEEDS, evaluator=ev)
    assert s.objective == 0.0 and s.per_dream == ()
    assert predict_many([], DREAMS, FlatReliability(), seeds=SEEDS, evaluator=ev) == []
