"""Round-3 contract (P0): seed guard, observation stripping, regions, config, stand-ins. Dev seeds 9100-9199 only."""
import numpy as np
import pytest

from payguard.dream import (DreamConfig, FlatReliability, FlatSkeptic, UniformWorldModel, assert_seed_allowed, observe,
                            region_of, uniform_dreams)
from payguard.dream.contract import ReliabilityMap, Skeptic, WorldModel
from payguard.protocol import TRAIN_FAMILIES, episode_for
from payguard.sim.attacks import GENERATORS, sample_params


def test_seed_guard():
    for ok in (1000, 1019, 9000, 9999):
        assert_seed_allowed(ok)
    for bad in (999, 1020, 2000, 3005, 4010, 5000, 5019, 10000):
        with pytest.raises(AssertionError):
            assert_seed_allowed(bad)
    assert_seed_allowed(5000, allow_test=True)


def test_observe_keeps_only_is_attack():
    ep = episode_for(9100, TRAIN_FAMILIES)
    ob = observe(ep)
    assert ob.confirmed_attack_ids == frozenset(i for i, t in ep.truth.items() if t.is_attack)
    assert len(ob.events) == len(ep.events)
    assert not hasattr(ob, "truth")
    with pytest.raises(AssertionError):
        observe(type(ep)(2000, ep.hours, ep.events, ep.truth, ep.accounts))


def test_region_bins_in_range_and_deterministic():
    rng = np.random.default_rng(0)
    for fam in TRAIN_FAMILIES:
        for _ in range(50):
            fam_r, idx = region_of((fam, sample_params(fam, rng)))
            assert fam_r == fam and all(0 <= k < 4 for k in idx)
    p = sample_params("ring", np.random.default_rng(1))
    assert region_of(("ring", p)) == region_of(("ring", p))


def test_config_arms_and_validation():
    assert DreamConfig.for_arm("dreamer_nowm").use_wm is False
    assert DreamConfig.for_arm("dreamer_noskeptic").use_skeptic is False
    assert DreamConfig.for_arm("dreamer_nocalib").use_calibration is False
    full = DreamConfig.for_arm("dreamer_full")
    assert full.use_wm and full.use_skeptic and full.use_calibration
    with pytest.raises(ValueError):
        DreamConfig(share_posterior=0.9)
    with pytest.raises(ValueError):
        DreamConfig(lambda0=0.0)
    with pytest.raises(KeyError):
        DreamConfig.for_arm("other")


def test_stand_ins_satisfy_protocols_and_are_deterministic():
    assert isinstance(UniformWorldModel(), WorldModel)
    assert isinstance(FlatReliability(), ReliabilityMap)
    assert isinstance(FlatSkeptic(), Skeptic)
    a = uniform_dreams(6, np.random.default_rng(3))
    b = UniformWorldModel().fit([], np.random.default_rng(9)).sample(6, np.random.default_rng(3))
    assert a == b
    assert [d[0][0] for d in a] == ["ring", "structuring"] * 3
    assert all(d[0][0] in GENERATORS for d in a)
    assert FlatReliability().weight(a[0], 1.0) == 1.0
