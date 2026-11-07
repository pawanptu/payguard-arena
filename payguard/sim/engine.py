"""Episode construction and payout resolution (what happens after a verdict).

Resolution randomness is keyed by (episode seed, event id), NOT by call order, so a defender that
decides differently never perturbs the random draws seen by later events.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attacks import GENERATORS
from .types import Episode, Event, Truth, Verdict
from .workers import IdAlloc, honest_population

# --- resolution constants (part of the frozen benchmark) ---
STEPUP_DELAY_H = 4.0      # extra verification takes this long
HOLD_DELAY_H = 24.0       # human review turnaround
REVIEW_ACCURACY = 0.9     # a reviewer decides correctly this often
LABEL_DELAY_H = 72.0      # a released payout's chargeback/clean label arrives this much later
STEPUP_PASS = {           # chance the requester passes extra verification, by who they really are
    "honest": 0.95, "ring": 0.5, "structuring": 0.2, "sleeper": 0.35,
}


@dataclass(frozen=True)
class SimConfig:
    days: int = 14
    n_workers: int = 300
    n_requesters: int = 60
    n_small_shops: int = 6

    @property
    def hours(self) -> float:
        return self.days * 24.0


@dataclass(frozen=True)
class Outcome:
    event_id: int
    verdict: Verdict
    paid: bool
    paid_at: float | None  # hours since episode start
    label_at: float        # when the ground-truth label reaches the defender


def build_episode(seed: int, campaigns: list[tuple[str, object]], cfg: SimConfig = SimConfig()) -> Episode:
    """Honest population plus the given attack campaigns, fully determined by `seed`."""
    rng = np.random.default_rng([seed, 0])
    raws, accounts = honest_population(rng, cfg.n_workers, cfg.n_requesters, cfg.n_small_shops, cfg.days)
    ids = IdAlloc(cfg.n_workers, cfg.n_requesters + cfg.n_small_shops, cfg.n_workers, cfg.n_workers)
    for cid, (family, params) in enumerate(campaigns):
        crng = np.random.default_rng([seed, 1, cid])
        r, a = GENERATORS[family](params, crng, ids, cfg.days, cid, cfg.n_requesters)
        raws += r
        accounts.update(a)
    raws.sort(key=lambda r: (r.t, r.worker))
    events: list[Event] = []
    truth: dict[int, Truth] = {}
    for i, r in enumerate(raws):
        events.append(Event(i, r.t, r.worker, r.requester, r.device, r.destination, r.amount))
        truth[i] = r.truth
    return Episode(seed, cfg.hours, events, truth, accounts)


def resolve(seed: int, event: Event, truth: Truth, verdict: Verdict) -> Outcome:
    rng = np.random.default_rng([seed, event.id, 7])
    if verdict is Verdict.RELEASE:
        return Outcome(event.id, verdict, True, event.t, event.t + LABEL_DELAY_H)
    if verdict is Verdict.STEP_UP:
        ok = bool(rng.random() < STEPUP_PASS[truth.family])
        return Outcome(event.id, verdict, ok, event.t + STEPUP_DELAY_H if ok else None,
                       event.t + (LABEL_DELAY_H if ok else STEPUP_DELAY_H))
    correct = bool(rng.random() < REVIEW_ACCURACY)
    paid = (not truth.is_attack) if correct else truth.is_attack
    return Outcome(event.id, verdict, paid, event.t + HOLD_DELAY_H if paid else None, event.t + HOLD_DELAY_H)
