"""Parameterized attack generators. Attackers evolve by numeric mutation of these parameters.

  ring         clean PER ACCOUNT by construction: every account looks like an ordinary aged worker
               (normal amounts, rate, age, own device and destination). The only signal is collective:
               a few colluding requesters pay the whole ring (each payout picks one of them), so ring
               workers end up with nearly identical requester sets.
  structuring  splits a large cash-out into payouts just under the platform's review threshold.
  sleeper      (HELD OUT from training) behaves honestly for days, then cashes out well above its own
               baseline but still below the single-payout threshold, spaced out to dodge velocity rules.
               Only the cash-out is an attack (a loss). The trust-building payouts before it are labelled
               honest: in a real bust-out they produce no chargebacks, so no fraud label may reach a
               defender before the cash-out starts (otherwise "remember bad workers" would solve it).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from .types import Truth
from .workers import HONEST, IdAlloc, Raw, draw_amount

REVIEW_THRESHOLD_USD = 500.0  # platform's single-payout review threshold; attackers know it


@dataclass(frozen=True)
class RingParams:
    ring_size: int = 12
    n_requesters: int = 2
    rate_per_day: float = 0.8
    amount_median: float = 60.0
    amount_sigma: float = 0.5
    account_age_days: float = 200.0
    start_day: float = 1.0


@dataclass(frozen=True)
class StructuringParams:
    n_accounts: int = 2
    eps: float = 20.0               # each payout is REVIEW_THRESHOLD_USD - eps
    k: int = 3                      # payouts per burst
    burst_window_hours: float = 6.0
    n_bursts: int = 2
    account_age_days: float = 15.0
    start_day: float = 1.0


@dataclass(frozen=True)
class SleeperParams:
    n_accounts: int = 2
    sleep_days: float = 6.0
    base_median: float = 35.0
    base_rate_per_day: float = 0.5
    ramp: float = 1.05              # slow-roll: per-payout growth during the sleep phase
    cashout_mult: float = 8.0       # cash-out amount relative to the account's own baseline
    cashout_count: int = 3
    cashout_gap_hours: float = 12.0
    account_age_days: float = 45.0
    start_day: float = 0.0


PARAMS = {"ring": RingParams, "structuring": StructuringParams, "sleeper": SleeperParams}

BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "ring": dict(ring_size=(4, 30), n_requesters=(1, 5), rate_per_day=(0.3, 1.5), amount_median=(30, 120),
                 amount_sigma=(0.3, 0.7), account_age_days=(60, 500), start_day=(0, 5)),
    "structuring": dict(n_accounts=(1, 4), eps=(1, 60), k=(2, 6), burst_window_hours=(1, 12), n_bursts=(1, 3),
                        account_age_days=(0, 40), start_day=(0, 6)),
    "sleeper": dict(n_accounts=(1, 4), sleep_days=(2, 9), base_median=(20, 80), base_rate_per_day=(0.3, 1.2),
                    ramp=(1.0, 1.3), cashout_mult=(3, 12), cashout_count=(2, 6), cashout_gap_hours=(2, 36),
                    account_age_days=(20, 120), start_day=(0, 3)),
}
INT_FIELDS = {"ring_size", "n_requesters", "n_accounts", "k", "n_bursts", "cashout_count"}


def sample_params(family: str, rng: np.random.Generator):
    """Uniform draw inside the family's bounds."""
    kw = {}
    for name, (lo, hi) in BOUNDS[family].items():
        kw[name] = int(rng.integers(lo, hi + 1)) if name in INT_FIELDS else float(rng.uniform(lo, hi))
    return PARAMS[family](**kw)


def mutate(params, family: str, rng: np.random.Generator, scale: float = 0.15):
    """Numeric mutation: gaussian jitter of each field, clipped to bounds; integers move by at most 1."""
    kw = {}
    for f in fields(params):
        lo, hi = BOUNDS[family][f.name]
        v = getattr(params, f.name)
        if f.name in INT_FIELDS:
            v = int(np.clip(v + rng.integers(-1, 2), lo, hi))
        else:
            v = float(np.clip(v + rng.normal(0, scale * (hi - lo)), lo, hi))
        kw[f.name] = v
    return replace(params, **kw)


def _truth(family: str, cid: int) -> Truth:
    return Truth(is_attack=True, family=family, campaign_id=cid)


def gen_ring(p: RingParams, rng, ids: IdAlloc, days: int, cid: int, n_honest_requesters: int):
    hours = days * 24.0
    requesters = [ids.requester() for _ in range(p.n_requesters)]
    raws, accounts = [], {}
    for _ in range(p.ring_size):
        w, dev, dst = ids.worker(), ids.device(), ids.destination()
        accounts[w] = -p.account_age_days * 24.0
        n = int(rng.poisson(p.rate_per_day * max(days - p.start_day, 0.0)))
        for t in np.sort(rng.uniform(p.start_day * 24.0, hours, size=n)):
            raws.append(Raw(float(t), w, int(rng.choice(requesters)), dev, dst,
                            draw_amount(rng, p.amount_median, p.amount_sigma), _truth("ring", cid)))
    return raws, accounts


def gen_structuring(p: StructuringParams, rng, ids: IdAlloc, days: int, cid: int, n_honest_requesters: int):
    hours = days * 24.0
    raws, accounts = [], {}
    for _ in range(p.n_accounts):
        w, dev, dst = ids.worker(), ids.device(), ids.destination()
        accounts[w] = -p.account_age_days * 24.0
        for _b in range(p.n_bursts):
            lo = p.start_day * 24.0
            start = float(rng.uniform(lo, max(lo, hours - p.burst_window_hours)))
            for t in np.sort(rng.uniform(start, start + p.burst_window_hours, size=p.k)):
                if t >= hours:
                    continue
                amount = min(REVIEW_THRESHOLD_USD - p.eps + float(rng.normal(0, 1.0)), REVIEW_THRESHOLD_USD - 0.01)
                raws.append(Raw(float(t), w, int(rng.integers(0, n_honest_requesters)), dev, dst,
                                round(amount, 2), _truth("structuring", cid)))
    return raws, accounts


def gen_sleeper(p: SleeperParams, rng, ids: IdAlloc, days: int, cid: int, n_honest_requesters: int):
    hours = days * 24.0
    raws, accounts = [], {}
    for _ in range(p.n_accounts):
        w, dev, dst = ids.worker(), ids.device(), ids.destination()
        accounts[w] = -p.account_age_days * 24.0
        start = p.start_day * 24.0
        n = int(rng.poisson(p.base_rate_per_day * p.sleep_days))
        for i, t in enumerate(np.sort(rng.uniform(start, start + p.sleep_days * 24.0, size=n))):
            if t >= hours:
                continue
            amount = min(draw_amount(rng, p.base_median, 0.3) * p.ramp ** i, REVIEW_THRESHOLD_USD - 50.0)
            raws.append(Raw(float(t), w, int(rng.integers(0, n_honest_requesters)), dev, dst,
                            round(amount, 2), HONEST))  # trust-building phase: not a loss, no fraud label
        cash0 = start + p.sleep_days * 24.0
        for j in range(p.cashout_count):
            t = cash0 + j * p.cashout_gap_hours + float(rng.uniform(0, 2.0))
            if t >= hours:
                continue
            amount = min(p.base_median * p.cashout_mult * float(np.exp(0.2 * rng.standard_normal())),
                         REVIEW_THRESHOLD_USD - 50.0)
            raws.append(Raw(t, w, int(rng.integers(0, n_honest_requesters)), dev, dst,
                            round(amount, 2), _truth("sleeper", cid)))
    return raws, accounts


GENERATORS = {"ring": gen_ring, "structuring": gen_structuring, "sleeper": gen_sleeper}
