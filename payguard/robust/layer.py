"""swarm_r2: the L1 swarm plus an honest-calibrated anomaly union layer.

Principle (generic, not tied to any attack family). The honest population is the one thing that is the same in
every episode; attacks are not. An attack that the swarm has never seen still has to deviate from honest behaviour
on SOME lens to be worth anything to the attacker. So instead of tuning thresholds to the attacks seen in training
(which is how a lens that is silent on the training attacks gets loosened to save honest friction), every lens gets a
fixed honest false-positive budget:

  * the layer watches nine history-only statistics (the raw signals behind the four specialists and the static
    rules: 24 h USD and count, single amount, amount against the worker's own median, amount and 24 h USD against
    the account-age cap, the 24 h rate z-score against the worker's own rate, the graph cluster score, device /
    destination sharing);
  * for each statistic the cut-off is an upper quantile of that statistic over HONEST fit events only (no attack
    event is used), with one common tail level per verdict, found by bisection so that the union of all
    statistics flags `beta_h` (HOLD) / `beta_s` (STEP_UP or worse) of the honest fit events;
  * the layer's verdict is HOLD if any statistic is above its HOLD cut-off, else STEP_UP if above its STEP_UP
    cut-off, else RELEASE; the defender returns the more severe of the swarm's verdict and the layer's.

Attack data only chooses the two budgets (eval/r2/SELECTION_RULE.md). Nothing here reads ground truth at decision
time: the statistics come from the Event and the PlatformLog, labels reach the swarm through `learn` as before.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from ..sim.features import PlatformLog
from ..sim.types import Episode, Event, Verdict
from ..swarm.coordinator import Swarm
from ..swarm.defenders import Graph, Policy
from ..swarm.params import SwarmParams, default_params

STAT_NAMES = ("window_usd", "window_n", "amount", "amount_ratio", "amount_over_cap", "window_over_cap",
              "rate_z", "graph_cluster", "sharing")
_SEVERITY = {Verdict.RELEASE: 0, Verdict.STEP_UP: 1, Verdict.HOLD: 2}

# The Graph lens with open ramps (continuous in the block density) and its device/destination ramp switched off
# (share_min at its bound), so that "graph_cluster" is the cluster signal alone; sharing is its own statistic.
_OPEN_GRAPH = SwarmParams(graph_dens_lo=0.2, graph_dens_width=0.6, graph_share_min=6)


class StatExtractor:
    def __init__(self) -> None:
        self.graph = Graph(_OPEN_GRAPH)
        self.policy = Policy(default_params())

    def stats(self, event: Event, log: PlatformLog, f: dict[str, float] | None = None) -> list[float]:
        f = f if f is not None else log.features(event)
        cap = self.policy.cap(f["age_days"])
        if f["n_prior"] >= 4:
            observed_days = max(min(event.t / 24.0, f["age_days"]), 1.0)
            rate = f["n_prior"] / observed_days
            z = (f["window_n"] - rate) / math.sqrt(rate + 1.0)
        else:
            z = 0.0
        return [f["window_usd"], f["window_n"], f["amount"], f["amount_ratio"], f["amount"] / cap,
                f["window_usd"] / (3.0 * cap), z, self.graph.score(event, log, f),
                max(f["device_workers"], f["destination_workers"])]


def collect_stats(ep: Episode) -> tuple[np.ndarray, np.ndarray]:
    """(n_events x 9 statistics, is_attack flags) of one episode. Independent of any verdict."""
    ex, log = StatExtractor(), PlatformLog(ep.accounts)
    rows, flags = [], []
    for e in ep.events:
        rows.append(ex.stats(e, log))
        flags.append(ep.truth[e.id].is_attack)
        log.add(e)
    return np.asarray(rows, dtype=float), np.asarray(flags, dtype=bool)


# ---------------------------------------------------------------- calibration (honest events only) ------
def _cutoffs(sorted_cols: list[np.ndarray], q: float) -> np.ndarray:
    n = len(sorted_cols[0])
    k = int(q * n)
    return np.array([c[n - 1 - k] for c in sorted_cols])  # flag iff stat > cutoff: at most k honest events do


def _union_rate(x: np.ndarray, cut: np.ndarray) -> float:
    return float((x > cut).any(axis=1).mean())


def calibrate_level(x_honest: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    """Common tail level q (largest on a bisection grid) whose per-statistic upper-q cut-offs flag at most a share
    `beta` of the honest events in the UNION of the statistics. Returns (cut-offs, achieved union rate)."""
    cols = [np.sort(x_honest[:, j]) for j in range(x_honest.shape[1])]
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        if _union_rate(x_honest, _cutoffs(cols, mid)) <= beta:
            lo = mid
        else:
            hi = mid
    cut = _cutoffs(cols, lo)
    return cut, _union_rate(x_honest, cut)


def calibrate(x_honest: np.ndarray, beta_h: float, beta_s: float) -> dict:
    """Layer spec (plain JSON types) from honest fit statistics; betas are fractions (0.01 = 1 %)."""
    if beta_s < beta_h:
        raise ValueError("beta_s must be >= beta_h")
    hold, rate_h = calibrate_level(x_honest, beta_h)
    step, rate_s = calibrate_level(x_honest, beta_s)
    step = np.minimum(step, hold)  # STEP_UP cut-offs never above the HOLD ones
    return {"stats": list(STAT_NAMES), "hold": [float(v) for v in hold], "step": [float(v) for v in step],
            "beta_h": beta_h, "beta_s": beta_s, "honest_union_rate_hold": rate_h, "honest_union_rate_step": rate_s,
            "n_honest_events": int(len(x_honest))}


class AnomalyLayer:
    def __init__(self, spec: dict):
        if list(spec["stats"]) != list(STAT_NAMES):
            raise ValueError("layer spec was calibrated for different statistics")
        self.spec = spec
        self.hold = list(spec["hold"])
        self.step = list(spec["step"])
        self.extractor = StatExtractor()

    def verdict(self, event: Event, log: PlatformLog, f: dict[str, float] | None = None) -> tuple[Verdict, list[str]]:
        s = self.extractor.stats(event, log, f)
        if any(v > t for v, t in zip(s, self.hold)):
            return Verdict.HOLD, [n for n, v, t in zip(STAT_NAMES, s, self.hold) if v > t]
        if any(v > t for v, t in zip(s, self.step)):
            return Verdict.STEP_UP, [n for n, v, t in zip(STAT_NAMES, s, self.step) if v > t]
        return Verdict.RELEASE, []


class R2Defender:
    """Implements `payguard.sim.runner.Defender`: the L1 swarm (`params`) and, if given, the anomaly layer."""

    def __init__(self, params: SwarmParams | None = None, layer: dict | None = None, name: str = "swarm_r2"):
        self.params = params if params is not None else default_params()
        self.swarm = Swarm(self.params, learn_weights=True)
        self.layer = AnomalyLayer(layer) if layer else None
        self.name = name
        self.last_signals: dict = {}

    @property
    def weights(self) -> dict[str, float]:
        return self.swarm.weights

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        v = self.swarm.decide(event, log)
        sig = dict(self.swarm.last_signals)
        if self.layer is not None:
            lv, fired = self.layer.verdict(event, log)
            if _SEVERITY[lv] > _SEVERITY[v]:
                v = lv
            sig["layer"] = {"verdict": lv.value, "fired": fired}
            sig["verdict"] = v.value
        self.last_signals = sig
        return v

    def learn(self, event: Event, is_attack: bool) -> None:
        self.swarm.learn(event, is_attack)


def make_r2(params: SwarmParams | None = None, layer: dict | None = None, name: str = "swarm_r2") -> R2Defender:
    return R2Defender(params, layer, name)


def params_dict(p: SwarmParams) -> dict:
    return asdict(p)
