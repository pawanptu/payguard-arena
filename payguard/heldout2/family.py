"""Round-2 held-out attack family "h2" (sealed test set TEST_HELDOUT2, seeds 4000-4019).

Full rationale lives in docs/HELDOUT2_FAMILY.md. The generator has the same signature as
`payguard.sim.attacks.gen_sleeper` so `build_episode` can call it through the GENERATORS registry.

Fields listed in `INT_FIELDS_H2` are integers; every other field is a float drawn uniformly inside `BOUNDS_H2`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from payguard.sim.attacks import REVIEW_THRESHOLD_USD
from payguard.sim.types import Truth
from payguard.sim.workers import HONEST, IdAlloc, Raw, draw_amount

NAME = "h2"

# Chance the party behind a STEP_UP challenge passes extra verification (engine.STEPUP_PASS is keyed by
# family). Fixed by reasoning about the actor, NOT tuned against any defender.
STEPUP_PASS_PROB = 0.25


@dataclass(frozen=True)
class H2Params:
    n_victims: int = 6
    n_hubs: int = 1
    n_devices: int = 1
    victim_rate_per_day: float = 0.6
    victim_median: float = 55.0
    account_age_days: float = 250.0
    start_day: float = 4.0
    wave_span_hours: float = 36.0
    drain_count: int = 2
    drain_gap_hours: float = 8.0
    drain_mult: float = 2.0
    cap_margin: float = 80.0   # drain payouts stay at least this far under REVIEW_THRESHOLD_USD


BOUNDS_H2: dict[str, tuple[float, float]] = dict(
    n_victims=(3, 14), n_hubs=(1, 3), n_devices=(1, 3), victim_rate_per_day=(0.3, 1.2),
    victim_median=(30, 120), account_age_days=(60, 600), start_day=(2, 8), wave_span_hours=(6, 72),
    drain_count=(1, 4), drain_gap_hours=(1, 30), drain_mult=(1.0, 4.0), cap_margin=(20, 150),
)
INT_FIELDS_H2 = {"n_victims", "n_hubs", "n_devices", "drain_count"}


def gen_h2(p: H2Params, rng, ids: IdAlloc, days: int, cid: int, n_honest_requesters: int):
    hours = days * 24.0
    truth = Truth(is_attack=True, family=NAME, campaign_id=cid)
    cap = REVIEW_THRESHOLD_USD - p.cap_margin
    hubs = [ids.destination() for _ in range(p.n_hubs)]
    devices = [ids.device() for _ in range(p.n_devices)]
    raws, accounts = [], {}
    for i in range(p.n_victims):
        w, own_dev, own_dst = ids.worker(), ids.device(), ids.destination()
        accounts[w] = -p.account_age_days * 24.0
        homes = rng.choice(n_honest_requesters, size=2, replace=False)

        def requester() -> int:
            return int(homes[0] if rng.random() < 0.7 else homes[1])

        t_take = min(p.start_day * 24.0 + float(rng.uniform(0.0, p.wave_span_hours)), hours - 1.0)
        n = int(rng.poisson(p.victim_rate_per_day * t_take / 24.0))
        for t in np.sort(rng.uniform(0.0, t_take, size=n)):  # the real owner's activity: no loss, no label
            raws.append(Raw(float(t), w, requester(), own_dev, own_dst,
                            draw_amount(rng, p.victim_median, 0.5), HONEST))
        hub, dev = hubs[i % p.n_hubs], devices[int(rng.integers(0, p.n_devices))]
        for j in range(p.drain_count):
            t = t_take + j * p.drain_gap_hours + float(rng.uniform(0.0, 1.0))
            if t >= hours:
                continue
            amount = min(p.victim_median * p.drain_mult * float(np.exp(0.25 * rng.standard_normal())), cap)
            raws.append(Raw(t, w, requester(), dev, hub, round(max(amount, 5.0), 2), truth))
    return raws, accounts
