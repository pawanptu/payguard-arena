"""Baseline (a): a hand-written static rule engine, the way most platforms start.

Per-account features and simple counters only (no graph reasoning), fixed thresholds, no learning.
It must be a FAIR baseline: strong enough that beating it means something, but it cannot see
cross-account structure, which is exactly the gap the swarm is meant to close.

Rules (each per-account; the most severe verdict among the rules that fire wins, so order is irrelevant):
  big       one payout >= big_usd
  window    total requested in the last 24 h >= window_usd
  device    device shared by >= device_workers workers
  young     account younger than young_days asking for >= young_usd
  velocity  >= velocity_n payouts in the last 24 h
  drift     payout >= drift_ratio x the worker's own median (needs >= 3 prior payouts)
A threshold of INF disables a rule. The defaults below were tuned ONCE, on TRAIN seeds only, with the
pre-registered objective (scripts/tune_static.py, decision D-020), before any swarm result was seen.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..sim.features import PlatformLog
from ..sim.types import Event, Verdict

INF = float("inf")
SEVERITY = {Verdict.RELEASE: 0, Verdict.STEP_UP: 1, Verdict.HOLD: 2}


@dataclass(frozen=True)
class StaticRuleConfig:
    """Defaults = the tuned baseline (D-020). The original hand-picked policy is UNTUNED below."""
    big_usd: float = 300.0
    big_action: Verdict = Verdict.HOLD
    window_usd: float = INF
    window_action: Verdict = Verdict.STEP_UP
    device_workers: float = INF
    device_action: Verdict = Verdict.STEP_UP
    young_days: float = 1.0
    young_usd: float = 250.0
    young_action: Verdict = Verdict.STEP_UP
    velocity_n: float = INF
    velocity_action: Verdict = Verdict.STEP_UP
    drift_ratio: float = 2.0
    drift_action: Verdict = Verdict.HOLD


# The hand-picked starting point before tuning, kept for the record and for the tuning report.
UNTUNED = StaticRuleConfig(big_usd=500.0, window_usd=800.0, window_action=Verdict.HOLD, device_workers=3,
                           device_action=Verdict.HOLD, young_days=7.0, young_usd=150.0, velocity_n=6,
                           drift_ratio=INF, drift_action=Verdict.STEP_UP)


def fired_actions(f: dict[str, float], c: StaticRuleConfig) -> list[Verdict]:
    """Verdicts of every rule that fires on features `f` (per-account features only)."""
    out = []
    if f["amount"] >= c.big_usd:
        out.append(c.big_action)
    if f["window_usd"] >= c.window_usd:
        out.append(c.window_action)
    if f["device_workers"] >= c.device_workers:
        out.append(c.device_action)
    if f["age_days"] < c.young_days and f["amount"] >= c.young_usd:
        out.append(c.young_action)
    if f["window_n"] >= c.velocity_n:
        out.append(c.velocity_action)
    if f["amount_ratio"] >= c.drift_ratio:
        out.append(c.drift_action)
    return out


class StaticRules:
    name = "static_rules"

    def __init__(self, cfg: StaticRuleConfig = StaticRuleConfig()):
        self.cfg = cfg

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        return max(fired_actions(log.features(event), self.cfg), key=SEVERITY.__getitem__,
                   default=Verdict.RELEASE)

    def learn(self, event: Event, is_attack: bool) -> None:
        pass  # static: never adapts
