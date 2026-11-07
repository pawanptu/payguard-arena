"""Baseline (b): the best single specialist on its own, with the same verdict thresholds as the swarm."""
from __future__ import annotations

from ..sim.features import PlatformLog
from ..sim.types import Event, Verdict
from .coordinator import verdict_for
from .defenders import SPECIALISTS, make_specialist
from .params import SwarmParams, default_params

SPECIALIST_NAMES = tuple(SPECIALISTS)


class SingleDefender:
    """One specialist, no coordination, no learning: its own risk score goes straight through the thresholds."""

    def __init__(self, name: str, params: SwarmParams | None = None):
        if name not in SPECIALISTS:
            raise ValueError(f"unknown specialist {name!r}; choose from {SPECIALIST_NAMES}")
        self.params = params if params is not None else default_params()
        self.specialist = make_specialist(name, self.params)
        self.name = f"single_{name}"
        self.last_signals: dict = {}

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        risk = self.specialist.score(event, log)
        verdict = verdict_for(risk, self.params)
        self.last_signals = {"event_id": event.id, "scores": {self.specialist.name: risk},
                             "risk": risk, "verdict": verdict.value}
        return verdict

    def learn(self, event: Event, is_attack: bool) -> None:
        pass  # a single specialist never adapts


def make_single(name: str, params: SwarmParams | None = None) -> SingleDefender:
    return SingleDefender(name, params)
