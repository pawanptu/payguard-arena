"""Round-3 held-out attack family "h3" (sealed test set TEST_HELDOUT3, seeds 5000-5019).

Full rationale lives in docs/HELDOUT3_FAMILY.md. The generator has the same signature as
`payguard.sim.attacks.gen_sleeper` so `build_episode` can call it through the GENERATORS registry.

Fields listed in `INT_FIELDS_H3` are integers; every other field is a float drawn uniformly inside `BOUNDS_H3`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from payguard.sim.attacks import REVIEW_THRESHOLD_USD
from payguard.sim.types import Truth
from payguard.sim.workers import IdAlloc, Raw, draw_amount

NAME = "h3"

# Chance the party behind a STEP_UP challenge passes extra verification (engine.STEPUP_PASS is keyed by
# family). Fixed by reasoning about the actor (synthetic identities, scripted operators), NOT tuned against
# any defender.
STEPUP_PASS_PROB = 0.3


@dataclass(frozen=True)
class H3Params:
    n_accounts: int = 10
    accounts_per_device: int = 3
    accounts_per_dest: int = 4
    spawn_span_days: float = 6.0
    start_day: float = 1.0
    age_at_first_hours: float = 60.0
    payouts_per_account: int = 3
    gap_hours: float = 8.0
    amount_median: float = 120.0
    amount_sigma: float = 0.2
    n_requesters: int = 3


BOUNDS_H3: dict[str, tuple[float, float]] = dict(
    n_accounts=(4, 20), accounts_per_device=(2, 5), accounts_per_dest=(2, 6), spawn_span_days=(2, 9),
    start_day=(0, 4), age_at_first_hours=(26, 200), payouts_per_account=(2, 6), gap_hours=(1, 20),
    amount_median=(40, 200), amount_sigma=(0.1, 0.4), n_requesters=(1, 4),
)
INT_FIELDS_H3 = {"n_accounts", "accounts_per_device", "accounts_per_dest", "payouts_per_account", "n_requesters"}


def gen_h3(p: H3Params, rng, ids: IdAlloc, days: int, cid: int, n_honest_requesters: int):
    hours = days * 24.0
    truth = Truth(is_attack=True, family=NAME, campaign_id=cid)
    cap = REVIEW_THRESHOLD_USD - 50.0
    devices = [ids.device() for _ in range(-(-p.n_accounts // p.accounts_per_device))]
    dests = [ids.destination() for _ in range(-(-p.n_accounts // p.accounts_per_dest))]
    pool = rng.choice(n_honest_requesters, size=min(p.n_requesters, n_honest_requesters), replace=False)
    spawn = np.sort(rng.uniform(p.start_day * 24.0, p.start_day * 24.0 + p.spawn_span_days * 24.0,
                                size=p.n_accounts))
    raws, accounts = [], {}
    for i in range(p.n_accounts):
        w = ids.worker()
        dev, dst = devices[i // p.accounts_per_device], dests[i // p.accounts_per_dest]
        req = int(rng.choice(pool))
        t0 = float(spawn[i]) + p.age_at_first_hours  # the account is created at spawn time
        accounts[w] = float(spawn[i])
        for j in range(p.payouts_per_account):
            t = t0 + j * p.gap_hours + float(rng.uniform(0.0, 1.0))
            if t >= hours:
                continue
            amount = min(draw_amount(rng, p.amount_median, p.amount_sigma), cap)
            raws.append(Raw(t, w, req, dev, dst, round(amount, 2), truth))
    return raws, accounts
