"""Shared types. `Event` is everything a defender may see; ground truth is kept apart in `Truth`."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    RELEASE = "RELEASE"  # pay now
    HOLD = "HOLD"        # human review queue, delayed
    STEP_UP = "STEP_UP"  # ask the worker for extra verification, short delay


FAMILIES = ("ring", "structuring", "sleeper")


@dataclass(frozen=True)
class Event:
    """One payout request, as visible to a defender in production."""

    id: int
    t: float             # hours since episode start
    worker_id: int
    requester_id: int    # who approved the work that triggered this payout
    device_id: int
    destination_id: int  # payout destination (payee email / account fingerprint)
    amount: float        # USD


@dataclass(frozen=True)
class Truth:
    """Ground truth, never shown to defenders (only delivered later as a delayed label)."""

    is_attack: bool
    family: str          # "honest" or one of FAMILIES
    campaign_id: int     # -1 for honest


@dataclass
class Episode:
    seed: int
    hours: float
    events: list[Event]          # sorted by time, ids ascending
    truth: dict[int, Truth]
    accounts: dict[int, float]   # worker_id -> account creation time in hours (negative = before start)
