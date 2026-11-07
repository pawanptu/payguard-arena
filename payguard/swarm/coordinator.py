"""Coordinator: reputation-weighted vote of the specialists, with L1 online learning.

Decision. Each specialist returns a risk in [0, 1]. The aggregated risk is a reputation- and
confidence-weighted average of the specialists:

    risk = sum_i w_i * s_i^2 / sum_i w_i * s_i

A specialist's say in the vote is its reputation w_i times its own score s_i, so a specialist that sees
nothing (s_i = 0) abstains instead of voting "benign". This matters because the specialists are different
lenses: the graph specialist is silent on a lone structuring account, the policy specialist is silent on a
ring. A plain weighted mean would dilute every lone alarm by the three silent voices; here a lone alarm
passes through at its full strength (risk = s_i), several alarms are blended in proportion to reputation x
confidence, and no veto rule is needed. The risk never exceeds the highest individual score. It maps to a
verdict with two thresholds: >= hold_threshold -> HOLD, >= step_up_threshold -> STEP_UP, else RELEASE.

L1 learning (multiplicative weights from DELAYED labels). The scores of every decided event are remembered
until its ground-truth label arrives (the runner delivers it >= 4 h to 72 h later, never earlier), then
    w_i <- w_i * exp(-eta * |score_i - is_attack|),  renormalised to sum 1, every w_i >= min_weight.
A specialist that raised the alarm on an attack, or stayed quiet on an honest payout, keeps its weight; a
specialist that missed an attack or cried wolf loses some. `learn_weights=False` gives the frozen swarm
(baseline d): same specialists, same initial weights, never updated.
"""
from __future__ import annotations

import math

from ..sim.features import PlatformLog
from ..sim.types import Event, Verdict
from .defenders import SPECIALISTS, make_specialist
from .params import SwarmParams, default_params

NAMES = tuple(SPECIALISTS)
ACTIVE_SCORE = 0.05     # a specialist at or above this score counts as "speaking up" in last_signals
MAX_PENDING = 100_000   # safety bound on remembered, not yet labelled events (in practice ~72 h of traffic)


def verdict_for(risk: float, params: SwarmParams) -> Verdict:
    """The shared mapping from an aggregated risk to a verdict (also used by SingleDefender)."""
    if risk >= params.hold_threshold:
        return Verdict.HOLD
    if risk >= params.step_up_threshold:
        return Verdict.STEP_UP
    return Verdict.RELEASE


def normalize_with_floor(weights: dict[str, float], floor: float) -> dict[str, float]:
    """Scale to sum 1 while keeping every entry >= floor (entries below it are pinned, the rest rescaled)."""
    floor = min(floor, 1.0 / len(weights))
    pinned: set[str] = set()
    while True:
        free = [n for n in weights if n not in pinned]
        room = 1.0 - floor * len(pinned)
        total = sum(weights[n] for n in free)
        out = {n: floor for n in pinned}
        out.update({n: room * weights[n] / total for n in free})
        low = [n for n in free if out[n] < floor]
        if not low:
            return {n: out[n] for n in weights}
        pinned.update(low)


class Swarm:
    """Implements `payguard.sim.runner.Defender`."""

    def __init__(self, params: SwarmParams | None = None, *, learn_weights: bool = True):
        self.params = params if params is not None else default_params()
        self.learn_weights = learn_weights
        self.name = "swarm_l1" if learn_weights else "swarm_frozen"
        self.defenders = [make_specialist(n, self.params) for n in NAMES]
        self.weights: dict[str, float] = normalize_with_floor(
            {n: getattr(self.params, f"w0_{n}") for n in NAMES}, self.params.min_weight)
        self.last_signals: dict = {}
        self._pending: dict[int, dict[str, float]] = {}  # event id -> specialist scores, until the label arrives

    # --- decision ---
    def aggregate(self, scores: dict[str, float]) -> float:
        """Reputation- and confidence-weighted average of the specialists that speak up (see module doc)."""
        num = den = 0.0
        for n in NAMES:
            s = scores[n]
            num += self.weights[n] * s * s
            den += self.weights[n] * s
        return min(num / den, 1.0) if den > 1e-12 else 0.0

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        features = log.features(event)  # computed once, shared by the specialists that need it
        scores = {d.name: d.score(event, log, features) for d in self.defenders}
        risk = self.aggregate(scores)
        verdict = verdict_for(risk, self.params)
        self.last_signals = {"event_id": event.id, "scores": scores, "weights": dict(self.weights),
                             "risk": risk, "top": max(NAMES, key=scores.__getitem__),
                             "active": [n for n in NAMES if scores[n] >= ACTIVE_SCORE],
                             "verdict": verdict.value}
        if self.learn_weights:
            if len(self._pending) >= MAX_PENDING:
                self._pending.pop(next(iter(self._pending)))  # drop the oldest unlabelled event
            self._pending[event.id] = scores
        return verdict

    # --- L1 ---
    def learn(self, event: Event, is_attack: bool) -> None:
        scores = self._pending.pop(event.id, None)
        if scores is None or not self.learn_weights:
            return
        y = 1.0 if is_attack else 0.0
        eta = self.params.eta
        updated = {n: self.weights[n] * math.exp(-eta * abs(scores[n] - y)) for n in NAMES}
        self.weights = normalize_with_floor(updated, self.params.min_weight)


def make_swarm(params: SwarmParams | None = None, learn_weights: bool = True) -> Swarm:
    return Swarm(params if params is not None else default_params(), learn_weights=learn_weights)
