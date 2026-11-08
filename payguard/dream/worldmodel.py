"""Round-3 world model (P2): an ABC (approximate Bayesian computation) posterior over attacker campaigns.

What it learns. A distribution over campaigns `(family, params)` for the families it is given (default: the TRAIN
families). It learns it from `Observation`s ONLY: events, `confirmed_attack_ids`, `accounts`. It never sees
`Truth.family` or campaign ids (the module imports neither `Truth` nor `Episode`; a test greps the source).

Method: ABC rejection with a fixed reference table, Epanechnikov kernel weights, Beaumont local-linear regression
adjustment, and a perturbation kernel at sampling time.
  * Reference table: `n_sims` prior draws per family (uniform inside BOUNDS, via `sample_params`), each simulated ONCE
    with the real generator `GENERATORS[family](params, rng, IdAlloc(...), days, cid, n_honest_requesters)` (no honest
    traffic needed) and reduced to a summary-statistic vector. The table depends only on the simulator, the family
    set, `days`, and a fixed `pool_seed`, never on the data, so it is built once per process and cached; `fit` is then
    only a distance computation (milliseconds) and the skeptic can refit on folds cheaply. Why rejection and not SMC:
    the table is reused across every fit (20 observations, K folds, every generation), SMC would re-simulate per fit
    and cost K times more for no gain at this dimension; rejection with quantile tolerance has no tuning schedule.
  * Pseudo-campaigns. The confirmed attack events of an observation are an unlabelled mix of campaigns and families.
    We split them with OBSERVABLE links only (`pseudo_campaigns`): two attacking workers are linked when (a) their
    accounts were created within `link_hours` of each other (batch registration, from `Observation.accounts`) or (b)
    they both used a requester that is attack-concentrated (>= 80% of that requester's events are confirmed attacks,
    >= 6 of them, from >= 3 distinct workers; computed from all events and the labels; deliberately strict because
    unpopular honest requesters can look concentrated by chance and would wrongly merge small campaigns). Connected components are the pseudo-campaigns. This is a
    clustering, not a mixture model: a mixture over an unknown number of campaigns of unknown families would need a
    per-observation latent assignment; the links recover the single campaigns exactly in this simulator and degrade to
    "merged" campaigns (a coarser pseudo-campaign) when the links collide, which ABC then treats as one observation.
  * Per pseudo-campaign `m` we take the `accept_frac` nearest table rows (Euclidean on robustly standardized
    statistics), weight them with an Epanechnikov kernel, normalize to 1, and move their parameters toward the observed
    statistics with a ridge-shrunk weighted linear regression per family (Beaumont et al. 2002; params scaled to [0,1]
    of BOUNDS, clipped, ints rounded). Measured on dev seeds the adjustment cuts the normalized parameter error from
    0.21/0.25 (plain rejection, ring/structuring) to 0.05/0.10 (prior: 0.35). The population posterior is the average
    over the observed pseudo-campaigns: a mixture of per-campaign ABC posteriors (the predictive of "the next
    attacker campaign looks like one of those we have seen"). Particles are table rows, so params are always valid.
  * `sample` draws particles by weight and perturbs them (gaussian kernel, `jitter` x range, ints rounded, clipped to
    BOUNDS), so repeated draws are not literal copies. Before any fit (or for families the model has no knowledge of)
    it falls back to `uniform_dreams`.

Summary statistics (`STAT_NAMES`): workers, payouts per worker, amount level and spread, distance of amounts below
REVIEW_THRESHOLD_USD, share near it, distinct requesters, requester concentration (HHI) and reach across workers,
devices+destinations per worker, within-worker inter-payout gaps (median, q25, share < 12 h), mean account age, timing
of first attack and span, payout-count dispersion across workers. They are computed by the same function
(`campaign_stats`) from observed pseudo-campaigns and from simulated campaigns.

Extra public API for the skeptic: `posterior_particles()`, `predict_stats(n, rng)`, `predict_campaigns(n, rng)`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np

from ..protocol import TRAIN_FAMILIES
from ..sim.attacks import BOUNDS, GENERATORS, INT_FIELDS, PARAMS, REVIEW_THRESHOLD_USD, sample_params
from ..sim.workers import IdAlloc
from .contract import Campaign, Dream, Observation, assert_seed_allowed, uniform_dreams

STAT_NAMES = ("log_workers", "log_events_per_worker", "log_median_amount", "amount_sigma", "log_threshold_gap",
              "near_threshold_share", "log_requesters", "requester_hhi", "requester_reach", "devdst_per_worker",
              "log_gap_median", "log_gap_q25", "short_gap_share", "age_days", "first_day", "span_days", "count_cv")
N_STATS = len(STAT_NAMES)
DEFAULT_DAYS = 14
DEFAULT_N_SIMS = 250             # reference-table rows per family (400 and 800 recovered no better, 2x the cold cost)
DEFAULT_HONEST_REQUESTERS = 60   # SimConfig.n_requesters: popular requesters structuring picks from
_SIM_WORKERS, _SIM_REQ = 10_000, 100   # id offsets for simulated campaigns (never collide with anything we compute)


# ---------------------------------------------------------------------------------------------------- statistics
def campaign_stats(t: np.ndarray, w: np.ndarray, r: np.ndarray, dev: np.ndarray, dst: np.ndarray, amt: np.ndarray,
                   ages_days: Mapping[int, float], hours: float) -> np.ndarray:
    """Summary-statistic vector (len N_STATS) of one campaign's attack events. Returns NaNs if there are no events."""
    n = len(t)
    if n == 0:
        return np.full(N_STATS, np.nan)
    uw, cnt = np.unique(w, return_counts=True)
    nw = len(uw)
    # requesters: how concentrated, and how many attacking workers share each one
    ur, rinv, rc = np.unique(r, return_inverse=True, return_counts=True)
    share = rc / n
    key = np.unique(rinv.astype(np.int64) * 10_000_000 + w.astype(np.int64))
    workers_per_req = np.bincount(key // 10_000_000, minlength=len(ur))
    reach = float(np.sum(share * workers_per_req / nw))
    # gaps between consecutive payouts of the same worker
    order = np.lexsort((t, w))
    ws, ts = w[order], t[order]
    same = ws[1:] == ws[:-1]
    gaps = (ts[1:] - ts[:-1])[same]
    if len(gaps):
        g_med, g_q25 = float(np.median(gaps)), float(np.quantile(gaps, 0.25))
        short = float(np.mean(gaps < 12.0))
    else:
        g_med = g_q25 = float(hours)
        short = 0.0
    la = np.log(amt)
    gap_thr = np.clip(REVIEW_THRESHOLD_USD - amt, 0.0, 1000.0)
    ages = [ages_days[int(x)] for x in uw if int(x) in ages_days]
    mean_cnt = float(cnt.mean())
    return np.array([
        np.log(nw), np.log(n / nw), float(np.median(la)), float(la.std()), np.log1p(float(np.median(gap_thr))),
        float(np.mean((REVIEW_THRESHOLD_USD - amt >= 0) & (REVIEW_THRESHOLD_USD - amt <= 75.0))),
        np.log(len(ur)), float(np.sum(share ** 2)), reach,
        (len(np.unique(dev)) + len(np.unique(dst))) / (2.0 * nw),
        np.log1p(g_med), np.log1p(g_q25), short,
        float(np.mean(ages)) if ages else 0.0, float(t.min()) / 24.0, float(t.max() - t.min()) / 24.0,
        float(cnt.std() / mean_cnt) if mean_cnt > 0 else 0.0,
    ])


@dataclass(frozen=True)
class PseudoCampaign:
    """A cluster of confirmed attack events linked by observable evidence only. `stats` has len N_STATS."""

    event_ids: tuple[int, ...]
    workers: tuple[int, ...]
    stats: np.ndarray


def pseudo_campaigns(obs: Observation, link_hours: float = 1.0, conc: float = 0.8,
                     min_conc_events: int = 6, min_conc_workers: int = 3) -> list[PseudoCampaign]:
    """Cluster the confirmed attack events of `obs` into pseudo-campaigns (see module docstring). Deterministic."""
    attack = [e for e in obs.events if e.id in obs.confirmed_attack_ids]
    if not attack:
        return []
    workers = sorted({e.worker_id for e in attack})
    parent = {w: w for w in workers}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # (a) batch registration: creation times within link_hours (chained over the sorted creation times)
    dated = sorted((obs.accounts[w], w) for w in workers if w in obs.accounts)
    for (t0, w0), (t1, w1) in zip(dated, dated[1:]):
        if t1 - t0 <= link_hours:
            union(w0, w1)
    # (b) shared attack-concentrated requester
    tot: dict[int, int] = {}
    att: dict[int, int] = {}
    users: dict[int, set[int]] = {}
    for e in obs.events:
        tot[e.requester_id] = tot.get(e.requester_id, 0) + 1
        if e.id in obs.confirmed_attack_ids:
            att[e.requester_id] = att.get(e.requester_id, 0) + 1
            users.setdefault(e.requester_id, set()).add(e.worker_id)
    for rid, a in att.items():
        if a >= min_conc_events and a / tot[rid] >= conc and len(users[rid]) >= min_conc_workers:
            ws = sorted(users[rid])
            for w in ws[1:]:
                union(ws[0], w)
    groups: dict[int, list] = {}
    for e in attack:
        groups.setdefault(find(e.worker_id), []).append(e)
    out = []
    for root in sorted(groups):
        ev = groups[root]
        ages = {w: -obs.accounts[w] / 24.0 for w in {e.worker_id for e in ev} if w in obs.accounts}
        st = campaign_stats(np.array([e.t for e in ev]), np.array([e.worker_id for e in ev]),
                            np.array([e.requester_id for e in ev]), np.array([e.device_id for e in ev]),
                            np.array([e.destination_id for e in ev]), np.array([e.amount for e in ev]), ages, obs.hours)
        out.append(PseudoCampaign(tuple(e.id for e in ev), tuple(sorted({e.worker_id for e in ev})), st))
    return out


def simulate_stats(family: str, params: Any, rng: np.random.Generator, days: int = DEFAULT_DAYS,
                   n_honest_requesters: int = DEFAULT_HONEST_REQUESTERS) -> np.ndarray:
    """Run the real generator once (campaign alone, no honest traffic) and reduce its attack payouts to statistics."""
    ids = IdAlloc(_SIM_WORKERS, _SIM_REQ, _SIM_WORKERS, _SIM_WORKERS)
    raws, accounts = GENERATORS[family](params, rng, ids, days, 0, n_honest_requesters)
    rows = []
    for raw in raws:
        t, w, req, dv, ds, a, tr = raw
        if tr.is_attack:
            rows.append((t, w, req, dv, ds, a))
    if not rows:
        return np.full(N_STATS, np.nan)
    arr = np.array(rows, dtype=float)
    ages = {w: -v / 24.0 for w, v in accounts.items()}
    return campaign_stats(arr[:, 0], arr[:, 1].astype(np.int64), arr[:, 2].astype(np.int64), arr[:, 3].astype(np.int64),
                          arr[:, 4].astype(np.int64), arr[:, 5], ages, days * 24.0)


def observation_stats(observations: Sequence[Observation]) -> np.ndarray:
    """(M, N_STATS) stats of every pseudo-campaign across the observations (order: observation, then cluster)."""
    rows = [pc.stats for ob in observations for pc in pseudo_campaigns(ob)]
    return np.array(rows) if rows else np.empty((0, N_STATS))


# ---------------------------------------------------------------------------------------------------- reference table
@dataclass(frozen=True)
class _Table:
    campaigns: tuple[Campaign, ...]
    stats: np.ndarray      # (N, N_STATS), NaN rows = campaign produced no attack payout
    valid: np.ndarray      # (N,) bool
    center: np.ndarray
    scale: np.ndarray


_TABLES: dict[tuple, _Table] = {}


def _reference_table(families: tuple[str, ...], n_sims: int, days: int, nhr: int, pool_seed: int) -> _Table:
    key = (families, n_sims, days, nhr, pool_seed)
    if key not in _TABLES:
        rng = np.random.default_rng([pool_seed, days, nhr])
        camps, rows = [], []
        for fam in families:
            for _ in range(n_sims):
                p = sample_params(fam, rng)
                camps.append((fam, p))
                rows.append(simulate_stats(fam, p, np.random.default_rng(rng.integers(0, 2 ** 62)), days, nhr))
        stats = np.array(rows)
        valid = ~np.isnan(stats).any(axis=1)
        v = stats[valid]
        center = np.median(v, axis=0)
        mad = 1.4826 * np.median(np.abs(v - center), axis=0)
        scale = np.maximum(np.maximum(mad, 0.5 * v.std(axis=0)), 0.1)
        _TABLES[key] = _Table(tuple(camps), stats, valid, center, scale)
    return _TABLES[key]


def stat_scale(families: tuple[str, ...] = TRAIN_FAMILIES, days: int = DEFAULT_DAYS, n_sims: int = DEFAULT_N_SIMS,
               pool_seed: int = 20261108, nhr: int = DEFAULT_HONEST_REQUESTERS) -> tuple[np.ndarray, np.ndarray]:
    """(center, scale) used to standardize statistics; fixed by the simulator prior, not by data (shared with the skeptic)."""
    t = _reference_table(families, n_sims, days, nhr, pool_seed)
    return t.center, t.scale


# ---------------------------------------------------------------------------------------------------- the model
def _make_params(family: str, values: Mapping[str, float]):
    kw = {}
    for name, (lo, hi) in BOUNDS[family].items():
        v = float(np.clip(values[name], lo, hi))
        kw[name] = int(round(v)) if name in INT_FIELDS else v
    return PARAMS[family](**kw)


class AbcWorldModel:
    """ABC posterior over campaigns. Satisfies `contract.WorldModel`. Immutable style: `fit` returns a new model."""

    def __init__(self, families: tuple[str, ...] = TRAIN_FAMILIES, n_sims: int = DEFAULT_N_SIMS, accept_frac: float = 0.15,
                 jitter: float = 0.06, pool_seed: int = 20261108, n_honest_requesters: int = DEFAULT_HONEST_REQUESTERS,
                 ridge: float | None = 0.3):
        self.families = tuple(families)
        self.ridge = ridge
        self.n_sims, self.accept_frac, self.jitter = n_sims, accept_frac, jitter
        self.pool_seed, self.n_honest_requesters = pool_seed, n_honest_requesters
        self.days = DEFAULT_DAYS
        self.n_pseudo_campaigns = 0
        self._particles: list[tuple[str, Any, float]] = []   # (family, params, weight), weights sum to 1

    @property
    def fitted(self) -> bool:
        return bool(self._particles)

    def _clone(self) -> "AbcWorldModel":
        m = AbcWorldModel(self.families, self.n_sims, self.accept_frac, self.jitter, self.pool_seed,
                          self.n_honest_requesters, self.ridge)
        m.days = self.days
        return m

    # -- fitting
    def fit(self, observations: Sequence[Observation], rng: np.random.Generator) -> "AbcWorldModel":
        for ob in observations:
            assert_seed_allowed(ob.seed)
        new = self._clone()
        if not observations:
            return new
        new.days = int(round(float(np.median([ob.hours for ob in observations])) / 24.0)) or DEFAULT_DAYS
        obs_stats = observation_stats(observations)
        new.n_pseudo_campaigns = len(obs_stats)
        if not len(obs_stats):
            return new
        table = _reference_table(new.families, new.n_sims, new.days, new.n_honest_requesters, new.pool_seed)
        zt = (table.stats - table.center) / table.scale
        zo = (obs_stats - table.center) / table.scale
        idx_valid = np.flatnonzero(table.valid)
        zv = zt[idx_valid]
        q = max(8, int(np.ceil(new.accept_frac * len(idx_valid))))
        parts: dict[tuple, float] = {}
        objs: dict[tuple, Campaign] = {}
        for z in zo:
            d = np.sqrt(((zv - z) ** 2).sum(axis=1))
            near = np.argpartition(d, q - 1)[:q]
            h = d[near].max() * 1.0001 + 1e-9
            k = 1.0 - (d[near] / h) ** 2
            k = k / k.sum()
            for j, kj, adj in zip(idx_valid[near], k, new._adjust(table, zt, idx_valid[near], k, z)):
                key = (int(j), adj)
                parts[key] = parts.get(key, 0.0) + float(kj) / len(zo)
                objs[key] = table.campaigns[int(j)]
        new._particles = []
        for key, wt in parts.items():
            fam, p = objs[key]
            new._particles.append((fam, _make_params(fam, dict(zip(BOUNDS[fam], key[1]))) if key[1] else p, wt))
        return new

    def _adjust(self, table: _Table, zt: np.ndarray, rows: np.ndarray, k: np.ndarray, z: np.ndarray):
        """Beaumont-style local-linear regression adjustment of the accepted parameters toward the observed
        statistics, per family, with ridge shrinkage (`ridge`). Yields a tuple of adjusted raw values per row, or ()
        for no adjustment. Parameters are scaled to [0, 1] of BOUNDS first."""
        out: list[tuple] = [()] * len(rows)
        if self.ridge is None:
            return out
        fams = np.array([table.campaigns[int(i)][0] for i in rows])
        for fam in set(fams.tolist()):
            sel = np.flatnonzero(fams == fam)
            if len(sel) < 6:
                continue
            names = list(BOUNDS[fam])
            lo = np.array([BOUNDS[fam][n][0] for n in names], float)
            hi = np.array([BOUNDS[fam][n][1] for n in names], float)
            y = np.array([[float(getattr(table.campaigns[int(rows[i])][1], n)) for n in names] for i in sel])
            y = (y - lo) / (hi - lo)
            x = zt[rows[sel]] - z
            wk = k[sel] / k[sel].sum()
            ym, xm = wk @ y, wk @ x
            xc, yc = x - xm, y - ym
            xtw = xc.T * wk
            beta = np.linalg.solve(xtw @ xc + self.ridge * np.eye(x.shape[1]) / len(sel), xtw @ yc)
            adj = np.clip(y - x @ beta, 0.0, 1.0)   # x = s_i - s_obs
            for i, a in zip(sel, adj):
                out[i] = tuple(float(v) for v in lo + a * (hi - lo))
        return out

        return new

    def posterior_particles(self) -> list[tuple[str, Any, float]]:
        """(family, params, weight) with weights summing to 1; empty before any fit."""
        return list(self._particles)

    def family_share(self) -> dict[str, float]:
        out = {f: 0.0 for f in self.families}
        for fam, _, w in self._particles:
            out[fam] += w
        return out

    # -- sampling
    def sample(self, n: int, rng: np.random.Generator, families: tuple[str, ...] = TRAIN_FAMILIES) -> list[Dream]:
        pool = [p for p in self._particles if p[0] in families]
        if not pool or sum(p[2] for p in pool) <= 0:
            return uniform_dreams(n, rng, families)
        w = np.array([p[2] for p in pool])
        w = w / w.sum()
        picks = rng.choice(len(pool), size=n, p=w)
        dreams: list[Dream] = []
        for i in picks:
            fam, p, _ = pool[int(i)]
            noise = rng.standard_normal(len(fields(p)))
            vals = {}
            for f, z in zip(fields(p), noise):
                lo, hi = BOUNDS[fam][f.name]
                v = float(getattr(p, f.name)) + self.jitter * (hi - lo) * float(z)
                vals[f.name] = v
            dreams.append(((fam, _make_params(fam, vals)),))
        return dreams

    # -- predictions (used by the skeptic)
    def predict_campaigns(self, n: int, rng: np.random.Generator) -> tuple[list[Campaign], np.ndarray]:
        """`n` posterior-predictive campaigns and their simulated summary statistics (n, N_STATS)."""
        camps = [d[0] for d in self.sample(n, rng, self.families)]
        stats = np.array([simulate_stats(f, p, np.random.default_rng(rng.integers(0, 2 ** 62)), self.days,
                                         self.n_honest_requesters) for f, p in camps])
        return camps, stats

    def predict_stats(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Posterior-predictive summary statistics, shape (n, N_STATS); NaN row = campaign with no attack payout."""
        return self.predict_campaigns(n, rng)[1]
