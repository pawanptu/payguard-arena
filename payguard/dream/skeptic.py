"""Round-3 skeptic (P2): where can the world model NOT be trusted? A hold-out check, attributed to parameter regions.

`HoldoutSkeptic.assess(observations, make_model, rng)`:
  1. Split the observations deterministically into K folds (sorted by seed, fold i = every K-th observation; K =
     `k_folds`, capped at the number of observations).
  2. For each fold: `make_model().fit(other folds)`, then ask the fitted model for `n_pred` posterior-predictive
     campaigns and their summary statistics (`predict_campaigns` if the model has it; otherwise the documented
     fallback: take `model.sample(n_pred, ...)` and simulate each sampled campaign with the same generator and the
     same statistics, so ANY `WorldModel`, e.g. `UniformWorldModel`, can be assessed).
  3. Held-back pseudo-campaigns (clustered with observable links, `worldmodel.pseudo_campaigns`) are compared with the
     predictions in the robustly standardized statistic space of `worldmodel`. The prediction error of held-back
     campaign j is `e_j` = mean distance to its `k_near` nearest predicted campaigns, expressed RELATIVE to the same
     number for a uniform prior predictive of equal size: err_j = e_j / e_prior_j, clipped to [0, 3]. 1 means "no better
     than guessing from the prior", 0 perfect. This makes `tau` (default 1) scale-free.
  4. Attribution: err_j is charged to the regions (`region_of`) of the `k_near` predicted campaigns that best explain
     j, i.e. the posterior particles responsible for covering j. A region's error is the mean of the charges it
     received. A region the model's predictions never explained is "uncovered".
  5. Lookup `error(region)`: exact cell if charged; else the coarse parent cell (every bin index halved, so a
     neighbour cell of a checked cell is also checked); else `max_error`.

`max_error` = max(1.0, 95th percentile of the err_j, largest charged cell error). Reason: an unchecked region must
never look better than a checked one, and 1.0 is "as bad as the prior" so it is the least we assume about the
unknown; the 95th percentile keeps it tied to how wrong the model was on real held-back data. With fewer than 2
observations there is nothing to hold back: every region is uncovered (max_error = 1.0).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from ..protocol import TRAIN_FAMILIES
from .contract import N_BINS, Campaign, Dream, Observation, Region, WorldModel, region_of, uniform_dreams
from .worldmodel import DEFAULT_DAYS, DEFAULT_HONEST_REQUESTERS, observation_stats, simulate_stats, stat_scale

ERR_CLIP = 3.0


def _coarse(region: Region, n_bins: int) -> Region:
    fam, idx = region
    return fam, tuple(int(i) * 2 // n_bins for i in idx)


class RegionReliability:
    """Satisfies `contract.ReliabilityMap`. Immutable after construction."""

    def __init__(self, errors: dict[Region, float], max_error: float, n_bins: int = N_BINS,
                 charges: dict[Region, int] | None = None, held_back_errors: Sequence[float] = ()):
        self._err = dict(errors)
        self.max_error = float(max_error)
        self.n_bins = n_bins
        self.charges = dict(charges or {})
        self.held_back_errors = tuple(float(e) for e in held_back_errors)
        acc: dict[Region, list[float]] = defaultdict(list)
        for r, e in self._err.items():
            acc[_coarse(r, n_bins)].append(e * self.charges.get(r, 1))
        cnt: dict[Region, int] = defaultdict(int)
        for r in self._err:
            cnt[_coarse(r, n_bins)] += self.charges.get(r, 1)
        self._coarse_err = {c: float(sum(v) / cnt[c]) for c, v in acc.items()}

    def error(self, region: Region) -> float:
        if region in self._err:
            return self._err[region]
        c = _coarse(region, self.n_bins)
        return self._coarse_err.get(c, self.max_error)

    def weight(self, dream: Dream, tau: float) -> float:
        if tau <= 0:
            raise ValueError("tau must be > 0")
        ws = [1.0 / (1.0 + self.error(region_of(c, self.n_bins)) / tau) for c in dream]
        return float(sum(ws) / len(ws))

    def uncertain_regions(self, k: int) -> list[Region]:
        """The `k` checked regions with the highest error (ties: smaller region tuple first)."""
        if k <= 0:
            return []
        return [r for r, _ in sorted(self._err.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

    @property
    def regions(self) -> dict[Region, float]:
        return dict(self._err)


def _stats_of(campaigns: Sequence[Campaign], rng: np.random.Generator, days: int) -> np.ndarray:
    return np.array([simulate_stats(f, p, np.random.default_rng(rng.integers(0, 2 ** 62)), days,
                                    DEFAULT_HONEST_REQUESTERS) for f, p in campaigns])


def _predict(model: WorldModel, n: int, rng: np.random.Generator, families: tuple[str, ...],
             days: int) -> tuple[list[Campaign], np.ndarray]:
    if hasattr(model, "predict_campaigns"):
        return model.predict_campaigns(n, rng)           # type: ignore[attr-defined]
    camps = [c for d in model.sample(n, rng, families) for c in d]   # fallback: simulate the sampled dreams ourselves
    return camps, _stats_of(camps, rng, days)


class HoldoutSkeptic:
    """Satisfies `contract.Skeptic`."""

    def __init__(self, k_folds: int = 3, n_pred: int = 96, k_near: int = 8, n_bins: int = N_BINS,
                 families: tuple[str, ...] = TRAIN_FAMILIES):
        self.k_folds, self.n_pred, self.k_near, self.n_bins, self.families = k_folds, n_pred, k_near, n_bins, families

    def assess(self, observations: Sequence[Observation], make_model: Callable[[], WorldModel],
               rng: np.random.Generator) -> RegionReliability:
        obs = sorted(observations, key=lambda o: o.seed)
        K = min(self.k_folds, len(obs))
        if K < 2:
            return RegionReliability({}, 1.0, self.n_bins)
        days = int(round(float(np.median([o.hours for o in obs])) / 24.0)) or DEFAULT_DAYS
        center, scale = stat_scale(self.families, days)
        k = self.k_near
        # prior predictive baseline (one set shared by all folds)
        prior = [d[0] for d in uniform_dreams(self.n_pred, np.random.default_rng(rng.integers(0, 2 ** 62)), self.families)]
        zp = (_stats_of(prior, np.random.default_rng(rng.integers(0, 2 ** 62)), days) - center) / scale
        zp = zp[~np.isnan(zp).any(axis=1)]
        charges: dict[Region, list[float]] = defaultdict(list)
        held: list[float] = []
        for f in range(K):
            test = [o for i, o in enumerate(obs) if i % K == f]
            train = [o for i, o in enumerate(obs) if i % K != f]
            frng = np.random.default_rng(rng.integers(0, 2 ** 62))
            model = make_model().fit(train, frng)
            camps, st = _predict(model, self.n_pred, frng, self.families, days)
            z = (st - center) / scale
            ok = ~np.isnan(z).any(axis=1)
            z = z[ok]
            camps = [c for c, o in zip(camps, ok) if o]
            for zo in (observation_stats(test) - center) / scale:
                base = np.sort(np.sqrt(((zp - zo) ** 2).sum(axis=1)))[:k].mean() if len(zp) else 1.0
                if not len(z):
                    held.append(ERR_CLIP)
                    continue
                d = np.sqrt(((z - zo) ** 2).sum(axis=1))
                near = np.argsort(d, kind="stable")[:k]
                err = float(np.clip(d[near].mean() / max(base, 1e-9), 0.0, ERR_CLIP))
                held.append(err)
                for i in near:
                    charges[region_of(camps[int(i)], self.n_bins)].append(err)
        errors = {r: float(np.mean(v)) for r, v in charges.items()}
        counts = {r: len(v) for r, v in charges.items()}
        p95 = float(np.quantile(held, 0.95)) if held else 1.0
        max_error = max(1.0, p95, max(errors.values(), default=0.0))
        return RegionReliability(errors, max_error, self.n_bins, counts, held)
