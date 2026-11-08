"""Dream engine (P3): protocol, arms, blend, determinism, end-to-end with the stand-ins. Dev seeds 9300-9399 only."""
import json
import math

import numpy as np
import pytest

from payguard.dream import (DreamConfig, FlatReliability, FlatSkeptic, Observation, UniformWorldModel, observe,
                            uniform_dreams)
from payguard.dream.contract import DreamHooks
from payguard.dream.engine import DreamEngine, predicted_gain
from payguard.evolve.evaluation import Evaluator
from payguard.protocol import TRAIN_FAMILIES, episode_for
from payguard.swarm.params import SwarmParams

SEEDS = (9300, 9301)
SMALL = dict(n_dreams=4)


def engine(arm="dreamer_full", **kw):
    return DreamEngine(DreamConfig.for_arm(arm, **(SMALL | kw)), UniformWorldModel, FlatSkeptic(), seeds=SEEDS, base_seed=3)


@pytest.fixture(scope="module")
def obs():
    return [observe(episode_for(9310, TRAIN_FAMILIES))]


def test_protocol_and_blend_math():
    e = engine()
    assert isinstance(e, DreamHooks)
    assert e.blend([10.0, 20.0], [0.0, 10.0]) == [pytest.approx(7.0), pytest.approx(17.0)]  # lam = 0.3
    e.end_generation(5.0, 0.0)  # gap 5 -> ema 2.5 -> lam 0.3 / 3.5
    lam = 0.3 / 3.5
    assert e.blend([10.0], [0.0]) == [pytest.approx(10.0 * (1 - lam))]
    with pytest.raises(ValueError):
        e.blend([1.0], [1.0, 2.0])
    with pytest.raises(RuntimeError):
        e.dream_fitness([SwarmParams()], None)


def test_nocalib_blend_stays_fixed():
    e = engine("dreamer_nocalib")
    e.end_generation(50.0, 0.0)
    assert e.blend([10.0], [0.0]) == [pytest.approx(7.0)]


def test_none_gains_keep_state():
    e = engine()
    e.end_generation(2.0, 1.0)
    state = (e.calibrator.gap_ema, e.calibrator.lam())
    e.end_generation(None, None)
    assert (e.calibrator.gap_ema, e.calibrator.lam()) == state
    assert "gap" not in e.record() or e.record()["gap"] == 1.0


def test_predicted_gain_semantics():
    assert predicted_gain(10.0, 7.5) == 2.5      # challenger better -> positive
    assert predicted_gain(7.5, 10.0) == -2.5
    assert DreamEngine.predicted_gain(3.0, 1.0) == 2.0


def test_dreams_deterministic_per_generation(obs):
    a, b = engine(), engine()
    a.begin_generation(2, obs)
    b.begin_generation(2, obs)
    assert a.dreams == b.dreams and len(a.dreams) == 4
    b.begin_generation(3, obs)
    assert a.dreams != b.dreams
    first = list(b.dreams)
    b.begin_generation(2, obs)  # re-drawing generation 2 after generation 3 gives the generation-2 dreams again
    assert b.dreams == a.dreams and b.dreams != first


def test_nowm_arm_dreams_are_uniform(obs):
    e = engine("dreamer_nowm")
    e.begin_generation(0, obs)
    assert e.dreams == uniform_dreams(4, np.random.default_rng([3, 0, 3]), TRAIN_FAMILIES)
    assert e.record()["share_counts"] == {"uniform": 4}


def test_noskeptic_weights_are_one(obs):
    class Rel(FlatReliability):
        def weight(self, dream, tau):
            return 0.5

    class Skep:
        def assess(self, observations, make_model, rng):
            return Rel()

    on = DreamEngine(DreamConfig(**SMALL), UniformWorldModel, Skep(), seeds=SEEDS)
    off = DreamEngine(DreamConfig(use_skeptic=False, **SMALL), UniformWorldModel, Skep(), seeds=SEEDS)
    on.begin_generation(0, [])
    off.begin_generation(0, [])
    assert on.weights == (0.5,) * 4 and off.weights == (1.0,) * 4
    assert on.record()["mean_weight"] == 0.5 and off.record()["min_weight"] == 1.0


def test_end_to_end_two_candidates_four_dreams(obs):
    e = engine()
    cands = [SwarmParams(), SwarmParams(step_up_threshold=0.1, hold_threshold=0.55)]
    with Evaluator(workers=1) as ev:
        e.begin_generation(0, obs)
        fits = e.dream_fitness(cands, ev)
        assert len(fits) == 2 and all(math.isfinite(f) for f in fits) and fits[0] != fits[1]
        assert fits == [s.objective for s in e.last_scores]
        again = e.dream_fitness(cands, ev)
        assert again == fits
    blended = e.blend([5.0, 6.0], fits)
    assert len(blended) == 2
    e.end_generation(predicted_gain(fits[0], fits[1]), 0.5)
    rec = e.record()
    json.dumps(rec)
    for k in ("g", "lam", "gap_ema", "n_dreams", "mean_weight", "min_weight", "share_counts", "n_observations",
              "fit_seconds"):
        assert k in rec
    assert rec["g"] == 0 and rec["lam"] == 0.3 and rec["n_dreams"] == 4 and rec["n_observations"] == 1
    assert rec["gap_ema"] > 0 and rec["lam_next"] < 0.3 and sum(rec["share_counts"].values()) == 4
    assert rec["gap"] == pytest.approx(abs(predicted_gain(fits[0], fits[1]) - 0.5))


def test_record_before_begin_is_serializable():
    json.dumps(engine().record())
