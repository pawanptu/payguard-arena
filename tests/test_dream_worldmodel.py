"""Round-3 world model (P2): protocol, determinism, validity, recovery, no truth leak, timings. Dev seeds 9200-9299 + TRAIN."""
import ast
import dataclasses
import functools
import inspect
import time
from dataclasses import fields

import numpy as np
import pytest

from payguard.dream import Observation, UniformWorldModel, WorldModel, observe, uniform_dreams
from payguard.dream import worldmodel as wm
from payguard.dream.worldmodel import AbcWorldModel, pseudo_campaigns
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from payguard.sim.attacks import BOUNDS, INT_FIELDS, PARAMS, sample_params
from payguard.sim.engine import build_episode
from payguard.sim.types import Truth


def nd(family, p, q):
    """Mean over fields of |p - q| / range: normalized parameter distance in [0, 1]."""
    return float(np.mean([abs(getattr(p, f.name) - getattr(q, f.name)) / (BOUNDS[family][f.name][1] - BOUNDS[family][f.name][0])
                          for f in fields(p)]))


@functools.lru_cache(maxsize=None)
def train_obs():
    return tuple(observe(episode_for(s, TRAIN_FAMILIES)) for s in TRAIN_SEEDS)


@functools.lru_cache(maxsize=None)
def fitted_train_model():
    return AbcWorldModel().fit(list(train_obs()), np.random.default_rng(0))


def assert_valid_dream(dream, families=TRAIN_FAMILIES):
    assert isinstance(dream, tuple) and len(dream) == 1
    fam, p = dream[0]
    assert fam in families and isinstance(p, PARAMS[fam])
    for f in fields(p):
        lo, hi = BOUNDS[fam][f.name]
        v = getattr(p, f.name)
        assert lo <= v <= hi, (fam, f.name, v)
        if f.name in INT_FIELDS:
            assert isinstance(v, int) and not isinstance(v, bool)
        else:
            assert isinstance(v, float)


# ---------------------------------------------------------------------------------------------------- conformance
def test_protocol_conformance():
    assert isinstance(AbcWorldModel(), WorldModel)
    assert isinstance(fitted_train_model(), WorldModel)
    assert isinstance(UniformWorldModel(), WorldModel)


def test_unfitted_falls_back_to_uniform():
    m = AbcWorldModel()
    assert not m.fitted and m.posterior_particles() == []
    got = m.sample(10, np.random.default_rng(3))
    assert got == uniform_dreams(10, np.random.default_rng(3))
    # fitting on nothing / on observations without confirmed attacks stays unfitted
    assert not m.fit([], np.random.default_rng(0)).fitted
    ob = train_obs()[0]
    quiet = Observation(ob.seed, ob.hours, ob.events, frozenset(), ob.accounts)
    assert not m.fit([quiet], np.random.default_rng(0)).fitted


def test_fit_returns_new_model_and_does_not_mutate():
    m = AbcWorldModel()
    f = m.fit(list(train_obs()[:3]), np.random.default_rng(0))
    assert f is not m and f.fitted and not m.fitted


def test_seed_guard_on_fit():
    ob = train_obs()[0]
    bad = Observation(2000, ob.hours, ob.events, ob.confirmed_attack_ids, ob.accounts)
    with pytest.raises(AssertionError):
        AbcWorldModel().fit([bad], np.random.default_rng(0))


# ---------------------------------------------------------------------------------------------------- validity / determinism
def test_samples_valid_and_in_bounds():
    m = fitted_train_model()
    ds = m.sample(200, np.random.default_rng(1))
    assert len(ds) == 200
    for d in ds:
        assert_valid_dream(d)
    # family restriction honoured
    only = m.sample(30, np.random.default_rng(2), ("structuring",))
    assert all(d[0][0] == "structuring" for d in only)
    for d in only:
        assert_valid_dream(d, ("structuring",))


def test_unknown_family_falls_back_to_uniform():
    ds = fitted_train_model().sample(6, np.random.default_rng(4), ("sleeper",))
    assert ds == uniform_dreams(6, np.random.default_rng(4), ("sleeper",))


def test_particles_are_a_distribution_over_valid_campaigns():
    parts = fitted_train_model().posterior_particles()
    assert parts and abs(sum(w for _, _, w in parts) - 1.0) < 1e-9
    for fam, p, w in parts:
        assert w > 0
        assert_valid_dream(((fam, p),))


def test_determinism():
    obs = list(train_obs()[:8])
    a = AbcWorldModel().fit(obs, np.random.default_rng(5))
    b = AbcWorldModel().fit(obs, np.random.default_rng(5))
    assert a.posterior_particles() == b.posterior_particles()
    assert a.sample(20, np.random.default_rng(6)) == b.sample(20, np.random.default_rng(6))
    assert a.sample(20, np.random.default_rng(6)) != a.sample(20, np.random.default_rng(7))
    s1 = a.predict_stats(12, np.random.default_rng(8))
    s2 = b.predict_stats(12, np.random.default_rng(8))
    assert s1.shape == (12, wm.N_STATS) and np.array_equal(s1, s2, equal_nan=True)


# ---------------------------------------------------------------------------------------------------- clustering
def test_pseudo_campaigns_recover_the_campaign_structure():
    for s in range(9200, 9230):
        ep = episode_for(s, TRAIN_FAMILIES)
        true_c = {t.campaign_id for t in ep.truth.values() if t.is_attack}   # truth used only HERE, to grade the clustering
        pcs = pseudo_campaigns(observe(ep))
        assert len(pcs) == len(true_c), s


# ---------------------------------------------------------------------------------------------------- recovery
@functools.lru_cache(maxsize=None)
def recovery_numbers():
    rng = np.random.default_rng(5)
    out = {}
    for fam in TRAIN_FAMILIES:
        post, prior, share = [], [], []
        for s in range(9200, 9212):
            p = sample_params(fam, rng)
            ob = observe(build_episode(s, [(fam, p)]))
            m = AbcWorldModel().fit([ob], np.random.default_rng(0))
            parts = m.posterior_particles()
            sh = sum(w for f, _, w in parts if f == fam)
            share.append(sh)
            post.append(sum(nd(fam, p, pp) * w for f, pp, w in parts if f == fam) / sh)
            prior.append(float(np.mean([nd(fam, p, sample_params(fam, rng)) for _ in range(200)])))
        out[fam] = (float(np.mean(post)), float(np.mean(prior)), float(np.mean(share)))
    return out


@pytest.mark.parametrize("fam,max_post", [("ring", 0.15), ("structuring", 0.22)])
def test_posterior_concentrates_near_truth(fam, max_post):
    post, prior, share = recovery_numbers()[fam]
    print(f"{fam}: posterior distance {post:.3f} vs prior {prior:.3f}, true-family share {share:.3f}")
    assert post < max_post
    assert post < 0.6 * prior
    assert share > 0.9


def test_posterior_mean_beats_prior_mean():
    """Distance of the posterior MEAN (weighted-average parameters) vs the prior mean (mid-range)."""
    rng = np.random.default_rng(21)
    for fam in TRAIN_FAMILIES:
        better = 0
        for s in range(9220, 9228):
            p = sample_params(fam, rng)
            m = AbcWorldModel().fit([observe(build_episode(s, [(fam, p)]))], np.random.default_rng(0))
            parts = [(pp, w) for f, pp, w in m.posterior_particles() if f == fam]
            w = np.array([x[1] for x in parts])
            kw = {f.name: float(np.average([getattr(pp, f.name) for pp, _ in parts], weights=w)) for f in fields(p)}
            kw = {k: (int(round(v)) if k in INT_FIELDS else v) for k, v in kw.items()}
            mid = {k: (int(round((lo + hi) / 2)) if k in INT_FIELDS else (lo + hi) / 2) for k, (lo, hi) in BOUNDS[fam].items()}
            better += nd(fam, p, PARAMS[fam](**kw)) < nd(fam, p, PARAMS[fam](**mid))
        assert better >= 6, (fam, better)


def test_mixed_observation_keeps_both_families():
    """One episode with a ring AND a structuring campaign: the unlabelled mix is split and both families get mass."""
    rng = np.random.default_rng(31)
    shares, dists = [], []
    for s in range(9230, 9236):
        pr, ps = sample_params("ring", rng), sample_params("structuring", rng)
        ob = observe(build_episode(s, [("ring", pr), ("structuring", ps)]))
        assert len(pseudo_campaigns(ob)) == 2
        m = AbcWorldModel().fit([ob], np.random.default_rng(0))
        fs = m.family_share()
        shares.append(min(fs.values()))
        assert abs(sum(fs.values()) - 1) < 1e-9
        for fam, p in (("ring", pr), ("structuring", ps)):
            parts = [(pp, w) for f, pp, w in m.posterior_particles() if f == fam]
            dists.append(sum(nd(fam, p, pp) * w for pp, w in parts) / sum(w for _, w in parts))
    assert min(shares) > 0.35          # each campaign carries half of the posterior mass
    assert np.mean(dists) < 0.2        # prior-vs-truth distance is ~0.35


# ---------------------------------------------------------------------------------------------------- no truth leak
def test_no_truth_access_in_source():
    """The world model's code never reads Truth/Episode/family/campaign_id (AST check, comments/docstrings ignored)."""
    tree = ast.parse(inspect.getsource(wm))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"truth", "campaign_id", "family"}, (node.attr, node.lineno)
        if isinstance(node, ast.Name):
            assert node.id not in {"Truth", "Episode"}, (node.id, node.lineno)
        if isinstance(node, ast.ImportFrom):
            assert not {a.name for a in node.names} & {"Truth", "Episode", "build_episode", "episode_for"}


def test_shuffled_family_labels_change_nothing():
    """Same episodes with scrambled Truth.family / campaign_id (is_attack untouched): observations, fit, samples identical."""
    rng = np.random.default_rng(0)
    fams = ["ring", "structuring", "sleeper", "honest"]
    for s in (1000, 1001, 1002, 1003):
        ep = episode_for(s, TRAIN_FAMILIES)
        scrambled = dataclasses.replace(ep, truth={i: Truth(t.is_attack, fams[int(rng.integers(0, 4))], int(rng.integers(-1, 9)))
                                                   for i, t in ep.truth.items()})
        assert observe(ep) == observe(scrambled)
    ep_a = [episode_for(s, TRAIN_FAMILIES) for s in (1000, 1001, 1002)]
    ep_b = [dataclasses.replace(e, truth={i: Truth(t.is_attack, "x", 0) for i, t in e.truth.items()}) for e in ep_a]
    ma = AbcWorldModel().fit([observe(e) for e in ep_a], np.random.default_rng(1))
    mb = AbcWorldModel().fit([observe(e) for e in ep_b], np.random.default_rng(1))
    assert ma.posterior_particles() == mb.posterior_particles()
    assert ma.sample(10, np.random.default_rng(2)) == mb.sample(10, np.random.default_rng(2))
    assert not hasattr(observe(ep_a[0]), "truth")


# ---------------------------------------------------------------------------------------------------- timings
def test_timings():
    obs = list(train_obs())
    wm._TABLES.clear()                    # cold: includes the one-off reference table
    t0 = time.perf_counter()
    m = AbcWorldModel().fit(obs, np.random.default_rng(0))
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    AbcWorldModel().fit(obs, np.random.default_rng(0))
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    ds = m.sample(64, np.random.default_rng(1))
    samp = time.perf_counter() - t0
    print(f"fit(20 TRAIN obs) cold {cold:.2f}s warm {warm:.2f}s, sample(64) {samp * 1000:.1f}ms")
    assert len(ds) == 64
    assert cold < 30.0 and warm < 6.0 and samp < 1.0   # cold = one-off 1000-row table per process; warm ~2.5 s (D-043)
