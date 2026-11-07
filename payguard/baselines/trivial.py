"""Bounds for sanity checks: any real defender must sit between these two extremes."""
from __future__ import annotations

from ..sim.features import PlatformLog
from ..sim.types import Event, Verdict


class AlwaysRelease:
    """Pay everything instantly: zero friction, maximum leakage."""

    name = "always_release"

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        return Verdict.RELEASE

    def learn(self, event: Event, is_attack: bool) -> None:
        pass


class AlwaysHold:
    """Send everything to human review: the review step alone does the work, at maximum friction."""

    name = "always_hold"

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        return Verdict.HOLD

    def learn(self, event: Event, is_attack: bool) -> None:
        pass
