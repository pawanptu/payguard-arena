"""IMAGINER: turns a world model + a reliability map into this generation's dreams (P3).

Three sources, shares from `DreamConfig` (posterior / uncertain / recombined), counts by largest-remainder rounding:

  posterior   `model.sample`: what the world model believes the attackers look like.
  uncertain   parameters drawn UNIFORMLY inside the grid cells `reliability.uncertain_regions(k)` (cells where the
              world model was wrong or never checked). Cells are visited round-robin, most uncertain first. Inside a
              cell a float is uniform on the bin's [lo, hi) interval; an integer is uniform over the integers whose
              `region_of` bin is that bin (if the bin holds no integer, e.g. n_bursts in 1..3 with 4 bins, the integer
              nearest the bin centre). No uncertain cells (flat map) -> those dreams come from the posterior instead.
  recombined  built from a pool of posterior campaigns, TRAIN families only, half of each kind:
                * CROSSOVER: two posterior campaigns of the SAME family, each field taken from either parent (uniform
                  crossover): parameter combinations the posterior has not produced itself.
                * MULTI-CAMPAIGN: one campaign each of two DIFFERENT families in one episode (two distinct random
                  families; with one family only, two campaigns of it): tactic mixes, which is also what the real
                  arena episodes look like (ring + structuring together).
              A family the pool lacks is filled with a uniform draw so recombination never fails.

`use_wm=False` (arm dreamer_nowm): all n dreams are `uniform_dreams`, same ranges, no model, no reliability.

Held-out safety: families outside `TRAIN_FAMILIES` are refused (ValueError), so the sleeper can never be dreamed.
Everything is a pure function of (model, reliability, n, rng, families, cfg); child generators for each source are
seeded from `rng` so the three sources do not shift each other's draws.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Sequence

import numpy as np

from ..protocol import TRAIN_FAMILIES
from ..sim.attacks import BOUNDS, INT_FIELDS, PARAMS, sample_params
from .contract import Campaign, Dream, DreamConfig, ReliabilityMap, Region, WorldModel, region_of, uniform_dreams


def allocate(n: int, shares: Sequence[float]) -> list[int]:
    """Largest-remainder rounding of n * shares to integers summing to n (ties: earlier share first)."""
    raw = [n * s for s in shares]
    base = [int(np.floor(r + 1e-12)) for r in raw]
    order = sorted(range(len(shares)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[: n - sum(base)]:
        base[i] += 1
    return base


def sanitize(campaign: Campaign) -> Campaign:
    """Clip every field into BOUNDS (ints rounded) so a model's output is always a valid campaign."""
    family, p = campaign
    if family not in PARAMS or not isinstance(p, PARAMS[family]):
        raise TypeError(f"campaign {family!r} does not carry {PARAMS.get(family)} parameters: {type(p)}")
    kw = {}
    for f in fields(p):
        lo, hi = BOUNDS[family][f.name]
        v = float(getattr(p, f.name))
        kw[f.name] = int(min(max(round(v), lo), hi)) if f.name in INT_FIELDS else float(min(max(v, lo), hi))
    return family, PARAMS[family](**kw)


def _bin_edges(family: str, name: str, k: int, n_bins: int) -> tuple[float, float]:
    lo, hi = BOUNDS[family][name]
    w = (hi - lo) / n_bins
    return lo + k * w, lo + (k + 1) * w


def sample_in_region(region: Region, rng: np.random.Generator, n_bins: int) -> Campaign:
    """One campaign inside the grid cell `region` (see module docstring)."""
    family, idx = region
    names = [f.name for f in fields(PARAMS[family])]
    if len(idx) != len(names):
        raise ValueError(f"region {region} does not match {family} parameters")
    kw = {}
    for name, k in zip(names, idx):
        a, b = _bin_edges(family, name, int(k), n_bins)
        if name in INT_FIELDS:
            lo, hi = BOUNDS[family][name]
            cands = [v for v in range(int(lo), int(hi) + 1)
                     if min(max(int((v - lo) / (hi - lo) * n_bins), 0), n_bins - 1) == k]
            if cands:
                kw[name] = int(cands[int(rng.integers(0, len(cands)))])
            else:
                kw[name] = int(min(max(round((a + b) / 2), lo), hi))
        else:
            kw[name] = float(rng.uniform(a, b))
    return family, PARAMS[family](**kw)


def crossover(a: Campaign, b: Campaign, rng: np.random.Generator) -> Campaign:
    """Uniform crossover of two campaigns of the same family."""
    if a[0] != b[0]:
        raise ValueError("crossover needs two campaigns of the same family")
    family = a[0]
    kw = {f.name: getattr(a[1] if rng.random() < 0.5 else b[1], f.name) for f in fields(a[1])}
    return family, PARAMS[family](**kw)


class Imaginer:
    def __init__(self, cfg: DreamConfig):
        self.cfg = cfg
        self.last_counts: dict[str, int] = {}   # source -> number of dreams in the last call (after fallbacks)

    # ---------------------------------------------------------------------------------------------- public
    def dream(self, model: WorldModel, reliability: ReliabilityMap, n: int, rng: np.random.Generator,
              families: tuple[str, ...] | None = None) -> list[Dream]:
        families = tuple(self.cfg.families if families is None else families)
        bad = [f for f in families if f not in TRAIN_FAMILIES]
        if bad or not families:
            raise ValueError(f"imaginer works on TRAIN families only, got {families}")
        if n <= 0:
            self.last_counts = {}
            return []
        if not self.cfg.use_wm:
            self.last_counts = {"uniform": n}
            return uniform_dreams(n, rng, families)
        cfg = self.cfg
        n_post, n_unc, n_rec = allocate(n, (cfg.share_posterior, cfg.share_uncertain, cfg.share_recombined))
        r_post, r_unc, r_rec = (np.random.default_rng(int(s)) for s in rng.integers(0, 2**62, size=3))
        regions = [r for r in reliability.uncertain_regions(max(n_unc, 1)) if r[0] in families] if n_unc else []
        if n_unc and not regions:  # nothing uncertain known: those dreams come from the posterior instead
            n_post, n_unc = n_post + n_unc, 0
        out = self._posterior(model, n_post, r_post, families)
        out += self._uncertain(regions, n_unc, r_unc)
        out += self._recombined(model, n_rec, r_rec, families)
        assert len(out) == n
        self.last_counts = {"posterior": n_post, "uncertain": n_unc, "recombined": n_rec}
        return out

    # ---------------------------------------------------------------------------------------------- sources
    def _posterior(self, model: WorldModel, n: int, rng: np.random.Generator, families: tuple[str, ...]) -> list[Dream]:
        if n == 0:
            return []
        raw = list(model.sample(n, rng, families))
        if len(raw) < n:  # a short model is topped up uniformly rather than silently dropping dreams
            raw += uniform_dreams(n - len(raw), rng, families)
        return [tuple(sanitize(c) for c in d) for d in raw[:n]]

    def _uncertain(self, regions: list[Region], n: int, rng: np.random.Generator) -> list[Dream]:
        return [(sample_in_region(regions[i % len(regions)], rng, self.cfg.n_bins),) for i in range(n)]

    def _recombined(self, model: WorldModel, n: int, rng: np.random.Generator, families: tuple[str, ...]) -> list[Dream]:
        if n == 0:
            return []
        pool: dict[str, list[Campaign]] = {f: [] for f in families}
        for d in model.sample(max(4, 2 * n), rng, families):
            for c in d:
                if c[0] in pool:
                    pool[c[0]].append(sanitize(c))

        def pick(fam: str) -> Campaign:
            if not pool[fam]:
                return fam, sample_params(fam, rng)
            return pool[fam][int(rng.integers(0, len(pool[fam])))]

        out: list[Dream] = []
        for i in range(n):
            if i % 2 == 0:  # crossover
                fam = families[int(rng.integers(0, len(families)))]
                out.append((crossover(pick(fam), pick(fam), rng),))
            else:           # multi-campaign
                if len(families) >= 2:
                    a, b = rng.choice(len(families), size=2, replace=False)
                    fams = (families[int(a)], families[int(b)])
                else:
                    fams = (families[0], families[0])
                out.append(tuple(pick(f) for f in fams))
        return out


def dream_regions(dream: Dream, n_bins: int) -> list[Region]:
    """Grid cells touched by a dream (helper for logging and tests)."""
    return [region_of(c, n_bins) for c in dream]
