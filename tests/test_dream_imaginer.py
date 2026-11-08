"""Imaginer (P3): shares, determinism, validity, nowm arm. No episodes are built here."""
from dataclasses import fields

import numpy as np
import pytest

from payguard.dream import DreamConfig, FlatReliability, UniformWorldModel, region_of, uniform_dreams
from payguard.dream.imaginer import Imaginer, allocate, crossover, sample_in_region
from payguard.protocol import TRAIN_FAMILIES
from payguard.sim.attacks import BOUNDS, INT_FIELDS, PARAMS, RingParams, StructuringParams, sample_params


class Rel:
    """Reliability map with fixed uncertain cells."""
    max_error = 3.0

    def __init__(self, regions):
        self.regions = regions

    def error(self, region):
        return 3.0 if region in self.regions else 0.0

    def weight(self, dream, tau):
        return 1.0

    def uncertain_regions(self, k):
        return self.regions[:k]


class RingOnlyModel:
    """Posterior that only dreams ring campaigns with ring_size 10 (tests fill-in of the missing family)."""

    def fit(self, observations, rng):
        return self

    def sample(self, n, rng, families=TRAIN_FAMILIES):
        return [(("ring", RingParams(ring_size=10, amount_median=float(rng.uniform(30, 120)))),) for _ in range(n)]


def valid(dream):
    assert len(dream) >= 1
    for fam, p in dream:
        assert fam in TRAIN_FAMILIES and isinstance(p, PARAMS[fam])
        for f in fields(p):
            lo, hi = BOUNDS[fam][f.name]
            v = getattr(p, f.name)
            assert lo <= v <= hi, (fam, f.name, v)
            if f.name in INT_FIELDS:
                assert isinstance(v, int)


REGIONS = [("ring", (0, 1, 2, 3, 0, 1, 2)), ("structuring", (3, 0, 1, 2, 0, 1, 2)), ("ring", (1, 1, 1, 1, 1, 1, 1))]


def test_allocate_largest_remainder():
    assert allocate(32, (0.5, 0.25, 0.25)) == [16, 8, 8]
    assert allocate(32, (0.34, 0.33, 0.33)) == [11, 11, 10]
    assert allocate(10, (1 / 3, 1 / 3, 1 / 3)) == [4, 3, 3]
    for n in range(0, 40):
        for sh in [(0.5, 0.25, 0.25), (0.34, 0.33, 0.33), (0.2, 0.7, 0.1)]:
            a = allocate(n, sh)
            assert sum(a) == n and all(abs(x - n * s) < 1 for x, s in zip(a, sh))


def test_counts_and_validity():
    im = Imaginer(DreamConfig())
    ds = im.dream(UniformWorldModel(), Rel(REGIONS), 32, np.random.default_rng(1), TRAIN_FAMILIES)
    assert len(ds) == 32
    assert im.last_counts == {"posterior": 16, "uncertain": 8, "recombined": 8}
    for d in ds:
        valid(d)
    for j, d in enumerate(ds[16:24]):  # uncertain dreams: single campaign, round-robin over the cells' families
        assert len(d) == 1 and d[0][0] == REGIONS[j % 3][0]
    rec = ds[24:]
    assert all(len(d) == 1 for d in rec[0::2])  # crossovers
    for d in rec[1::2]:                         # multi-campaign mixes
        assert len(d) == 2 and d[0][0] != d[1][0]


def test_uncertain_samples_stay_in_cell():
    rng = np.random.default_rng(0)
    for reg in REGIONS:
        for _ in range(30):
            c = sample_in_region(reg, rng, 4)
            valid((c,))
            got = region_of(c)
            assert got[0] == reg[0]
            names = [f.name for f in fields(c[1])]
            for nm, a, b in zip(names, got[1], reg[1]):
                if a != b:  # only integer fields with an integer-free bin may deviate
                    assert nm in INT_FIELDS


def test_determinism_and_rng_sensitivity():
    im = Imaginer(DreamConfig())
    a = im.dream(UniformWorldModel(), Rel(REGIONS), 16, np.random.default_rng(5))
    b = im.dream(UniformWorldModel(), Rel(REGIONS), 16, np.random.default_rng(5))
    c = im.dream(UniformWorldModel(), Rel(REGIONS), 16, np.random.default_rng(6))
    assert a == b and a != c


def test_flat_reliability_falls_back_to_posterior():
    im = Imaginer(DreamConfig())
    ds = im.dream(UniformWorldModel(), FlatReliability(), 32, np.random.default_rng(2))
    assert im.last_counts == {"posterior": 24, "uncertain": 0, "recombined": 8}
    assert len(ds) == 32
    for d in ds:
        valid(d)


def test_nowm_gives_only_uniform_dreams():
    im = Imaginer(DreamConfig.for_arm("dreamer_nowm"))
    ds = im.dream(RingOnlyModel(), Rel(REGIONS), 12, np.random.default_rng(3), TRAIN_FAMILIES)
    assert ds == uniform_dreams(12, np.random.default_rng(3), TRAIN_FAMILIES)
    assert im.last_counts == {"uniform": 12}
    assert all(len(d) == 1 for d in ds)


def test_recombination_fills_missing_family_and_crosses_over():
    im = Imaginer(DreamConfig(share_posterior=0.0, share_uncertain=0.0, share_recombined=1.0))
    ds = im.dream(RingOnlyModel(), FlatReliability(), 20, np.random.default_rng(4))
    assert len(ds) == 20 and im.last_counts["recombined"] == 20
    multi = [d for d in ds if len(d) == 2]
    assert multi and all({c[0] for c in d} == {"ring", "structuring"} for d in multi)
    for d in ds:
        valid(d)
    singles = [c for d in ds if len(d) == 1 for c in d]
    assert singles and all(c[1].ring_size == 10 for c in singles if c[0] == "ring")


def test_crossover_takes_fields_from_parents():
    rng = np.random.default_rng(0)
    a, b = ("ring", sample_params("ring", rng)), ("ring", sample_params("ring", rng))
    for _ in range(20):
        _, child = crossover(a, b, rng)
        for f in fields(child):
            assert getattr(child, f.name) in (getattr(a[1], f.name), getattr(b[1], f.name))
    with pytest.raises(ValueError):
        crossover(a, ("structuring", StructuringParams()), rng)


def test_sanitizes_out_of_bounds_posterior_and_refuses_heldout_family():
    class BadModel(RingOnlyModel):
        def sample(self, n, rng, families=TRAIN_FAMILIES):
            return [(("ring", RingParams(ring_size=500, amount_median=-3.0)),)] * n

    im = Imaginer(DreamConfig(share_posterior=1.0, share_uncertain=0.0, share_recombined=0.0))
    for d in im.dream(BadModel(), FlatReliability(), 4, np.random.default_rng(0)):
        valid(d)
        assert d[0][1].ring_size == 30
    with pytest.raises(ValueError):
        im.dream(UniformWorldModel(), FlatReliability(), 4, np.random.default_rng(0), ("ring", "sleeper"))
