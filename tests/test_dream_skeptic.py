"""Round-3 skeptic (P2): protocol, reliability weights, coverage rule, hold-out discrimination, determinism, timings."""
import ast
import functools
import inspect
import time

import numpy as np
import pytest

from payguard.dream import (FlatSkeptic, Observation, ReliabilityMap, Skeptic, UniformWorldModel, observe, region_of,
                            uniform_dreams)
from payguard.dream import skeptic as sk
from payguard.dream.skeptic import HoldoutSkeptic, RegionReliability
from payguard.dream.worldmodel import AbcWorldModel
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from payguard.sim.attacks import RingParams, StructuringParams, mutate, sample_params
from payguard.sim.engine import build_episode

BASE = [("ring", RingParams(ring_size=10, n_requesters=2, rate_per_day=0.8, amount_median=60, amount_sigma=0.5,
                            account_age_days=200, start_day=1)),
        ("structuring", StructuringParams(n_accounts=2, eps=20, k=3, burst_window_hours=6, n_bursts=2,
                                          account_age_days=15, start_day=1))]


@functools.lru_cache(maxsize=None)
def narrow_obs():
    """Attackers that stay near one configuration (small mutations): a world model CAN learn this."""
    rng = np.random.default_rng(11)
    return tuple(observe(build_episode(s, [(f, mutate(p, f, rng, 0.05)) for f, p in BASE])) for s in range(9200, 9212))


@functools.lru_cache(maxsize=None)
def train_obs():
    return tuple(observe(episode_for(s, TRAIN_FAMILIES)) for s in TRAIN_SEEDS)


@functools.lru_cache(maxsize=None)
def abc_map():
    return HoldoutSkeptic().assess(list(narrow_obs()), AbcWorldModel, np.random.default_rng(2))


@functools.lru_cache(maxsize=None)
def uniform_map():
    return HoldoutSkeptic().assess(list(narrow_obs()), UniformWorldModel, np.random.default_rng(2))


def test_protocol_conformance():
    assert isinstance(HoldoutSkeptic(), Skeptic)
    assert isinstance(abc_map(), ReliabilityMap)
    assert isinstance(RegionReliability({}, 1.0), ReliabilityMap)


# ---------------------------------------------------------------------------------------------------- RegionReliability unit
def _ring_region(*idx):
    return ("ring", tuple(idx))


def test_region_reliability_rules():
    e1, e2, e3 = _ring_region(0, 0, 0, 0, 0, 0, 0), _ring_region(3, 3, 3, 3, 3, 3, 3), _ring_region(1, 3, 1, 3, 1, 3, 1)
    rr = RegionReliability({e1: 0.2, e2: 0.9, e3: 0.9}, max_error=1.7)
    assert rr.error(e1) == 0.2 and rr.error(e2) == 0.9
    # a neighbouring cell of a checked cell inherits the coarse-cell error, an unrelated region gets max_error
    assert rr.error(_ring_region(0, 0, 0, 0, 0, 0, 1)) == pytest.approx(0.2)
    assert rr.error(_ring_region(3, 0, 3, 0, 3, 0, 3)) == 1.7
    assert rr.error(("structuring", (0,) * 7)) == 1.7
    assert rr.error(("sleeper", (0,) * 9)) == 1.7
    # ordering: highest error first, ties broken by the region tuple, deterministic
    assert rr.uncertain_regions(2) == [e3, e2]
    assert rr.uncertain_regions(10) == [e3, e2, e1]
    assert rr.uncertain_regions(0) == []
    # weight formula
    p = sample_params("ring", np.random.default_rng(0))
    dream = (("ring", p),)
    err = rr.error(region_of(dream[0]))
    assert rr.weight(dream, 1.0) == pytest.approx(1 / (1 + err))
    assert rr.weight(dream, 0.5) == pytest.approx(1 / (1 + err / 0.5))
    two = (("ring", p), ("ring", sample_params("ring", np.random.default_rng(1))))
    ws = [1 / (1 + rr.error(region_of(c)) / 2.0) for c in two]
    assert rr.weight(two, 2.0) == pytest.approx(np.mean(ws))
    with pytest.raises(ValueError):
        rr.weight(dream, 0.0)


# ---------------------------------------------------------------------------------------------------- assess
def test_assess_reliability_is_sane():
    rm = abc_map()
    regions = rm.regions
    assert regions and all(np.isfinite(e) and e >= 0 for e in regions.values())
    assert rm.max_error >= 1.0 and rm.max_error >= max(regions.values())
    dreams = AbcWorldModel().fit(list(narrow_obs()), np.random.default_rng(0)).sample(64, np.random.default_rng(3))
    for tau in (0.5, 1.0, 3.0):
        for d in dreams + uniform_dreams(64, np.random.default_rng(3)):
            assert 0.0 < rm.weight(d, tau) <= 1.0
    top = rm.uncertain_regions(5)
    assert len(top) == 5 and top == rm.uncertain_regions(5)
    errs = [rm.error(r) for r in top]
    assert errs == sorted(errs, reverse=True)


def test_uncovered_regions_get_max_error():
    rm = abc_map()
    sleeper = ("sleeper", (1,) * 9)               # family the model never predicted
    assert rm.error(sleeper) == rm.max_error
    covered = set(rm.regions)
    # build a ring and a structuring cell that is neither checked nor has a checked coarse parent
    def coarse(r):
        return r[0], tuple(i * 2 // rm.n_bins for i in r[1])
    cc = {coarse(r) for r in covered}
    found = 0
    rng = np.random.default_rng(0)
    for fam in TRAIN_FAMILIES:
        for _ in range(500):
            r = region_of((fam, sample_params(fam, rng)))
            if r not in covered and coarse(r) not in cc:
                assert rm.error(r) == rm.max_error
                found += 1
                break
    assert found >= 1


def test_holdout_discriminates_learnable_attackers():
    """Attackers near a fixed configuration: the ABC model beats a uniform prior on held-back campaigns."""
    abc, uni = np.median(abc_map().held_back_errors), np.median(uniform_map().held_back_errors)
    print(f"median held-back error (relative to prior): abc {abc:.2f}, uniform {uni:.2f}")
    assert abc < 0.85 and abc < uni - 0.15
    assert 0.7 < uni < 1.4                          # uniform predictive is, by definition, about as good as the prior
    # dreams from the posterior sit in regions the skeptic trusts more than uniform dreams do
    rm = abc_map()
    post = AbcWorldModel().fit(list(narrow_obs()), np.random.default_rng(0)).sample(64, np.random.default_rng(3))
    uni_d = uniform_dreams(64, np.random.default_rng(3))
    assert np.mean([rm.weight(d, 1.0) for d in post]) > np.mean([rm.weight(d, 1.0) for d in uni_d]) + 0.05


def test_uniform_model_fallback_rule():
    """No predict_stats on the model: the skeptic simulates sampled dreams itself; every checked region ~ 'as good as prior'."""
    um = UniformWorldModel()
    assert not hasattr(um, "predict_stats")
    rm = uniform_map()
    vals = np.array(list(rm.regions.values()))
    assert 0.6 < vals.mean() < 1.5
    assert rm.max_error >= 1.0


def test_fewer_than_two_observations_all_uncovered():
    ob = narrow_obs()[:1]
    rm = HoldoutSkeptic().assess(list(ob), AbcWorldModel, np.random.default_rng(0))
    assert rm.max_error == 1.0 and rm.regions == {} and rm.uncertain_regions(3) == []
    assert rm.error(("ring", (0,) * 7)) == 1.0
    assert rm.weight((("ring", sample_params("ring", np.random.default_rng(0))),), 1.0) == pytest.approx(0.5)
    assert HoldoutSkeptic().assess([], AbcWorldModel, np.random.default_rng(0)).max_error == 1.0


def test_determinism_and_fold_split_independent_of_input_order():
    obs = list(narrow_obs()[:6])
    a = HoldoutSkeptic(k_folds=2).assess(obs, AbcWorldModel, np.random.default_rng(9))
    b = HoldoutSkeptic(k_folds=2).assess(list(reversed(obs)), AbcWorldModel, np.random.default_rng(9))
    assert a.regions == b.regions and a.max_error == b.max_error
    c = HoldoutSkeptic(k_folds=2).assess(obs, AbcWorldModel, np.random.default_rng(10))
    assert c.regions != a.regions or c.max_error != a.max_error
    assert abc_map().regions == HoldoutSkeptic().assess(list(narrow_obs()), AbcWorldModel, np.random.default_rng(2)).regions


def test_does_not_mutate_model_factory_state_or_observations():
    obs = list(narrow_obs()[:4])
    before = [(o.seed, len(o.events), o.confirmed_attack_ids) for o in obs]
    HoldoutSkeptic(k_folds=2).assess(obs, AbcWorldModel, np.random.default_rng(1))
    assert before == [(o.seed, len(o.events), o.confirmed_attack_ids) for o in obs]


def test_no_truth_access_in_source():
    tree = ast.parse(inspect.getsource(sk))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"truth", "campaign_id", "family"}, (node.attr, node.lineno)
        if isinstance(node, ast.Name):
            assert node.id not in {"Truth", "Episode"}, (node.id, node.lineno)
        if isinstance(node, ast.ImportFrom):
            assert not {a.name for a in node.names} & {"Truth", "Episode", "build_episode", "episode_for"}


def test_works_on_train_observations_and_timing():
    obs = list(train_obs())
    t0 = time.perf_counter()
    rm = HoldoutSkeptic().assess(obs, AbcWorldModel, np.random.default_rng(2))
    dt = time.perf_counter() - t0
    print(f"assess(20 TRAIN obs, K=3): {dt:.2f}s, {len(rm.regions)} checked regions, max_error {rm.max_error:.2f}")
    assert dt < 30.0
    assert rm.max_error >= 1.0 and len(rm.regions) > 10
    assert isinstance(FlatSkeptic().assess(obs, AbcWorldModel, np.random.default_rng(0)), ReliabilityMap)
