"""CALIBRATOR: how much the swarm trusts its own imagination (P3).

After a generation that promoted a challenger, the dreams predicted a gain and the real gate measured one. The gap
|predicted - realized| (objective points) feeds an exponential moving average; the weight of dreamed fitness is

    lam = lambda0 / (1 + gap_ema / sigma)        in (0, lambda0]

so big prediction errors shrink the dream weight toward 0, perfect predictions keep it at lambda0.

START VALUE. `gap_ema` starts at 0, i.e. the first generation trusts dreams at the full lambda0. Reason: the
selection rule fixes lambda0 as the maximum weight the experimenter is willing to grant, the dream arm must
actually be exercised in generation 0 (an a-priori distrust would make early generations a silent no-op), and the
damage is bounded: lambda0 <= 0.6 in the grid, and the real gate still decides promotions. The EMA is therefore
seeded by the first observed gap with `beta`, not initialised to it (gap_ema_1 = beta * gap_1); a conservative
start would need an extra pre-registered constant.

`enabled=False` (arm dreamer_nocalib): lam stays lambda0, but `update` still records history so the gap is logged.
`update(None, None)` (nobody promoted) changes nothing and records nothing.
"""
from __future__ import annotations


class EmaCalibrator:
    def __init__(self, lambda0: float, sigma: float, ema_beta: float, enabled: bool = True):
        if not (0 < lambda0 <= 1):
            raise ValueError("lambda0 must be in (0, 1]")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if not (0 < ema_beta <= 1):
            raise ValueError("ema_beta must be in (0, 1]")
        self.lambda0, self.sigma, self.ema_beta, self.enabled = float(lambda0), float(sigma), float(ema_beta), bool(enabled)
        self.gap_ema: float = 0.0
        self.history: list[dict] = []

    def update(self, predicted_gain: float | None, realized_gain: float | None) -> None:
        if predicted_gain is None or realized_gain is None:
            return
        gap = abs(float(predicted_gain) - float(realized_gain))
        self.gap_ema = self.ema_beta * gap + (1.0 - self.ema_beta) * self.gap_ema
        self.history.append({"predicted": float(predicted_gain), "realized": float(realized_gain), "gap": gap,
                             "gap_ema": self.gap_ema, "lam": self.lam()})

    def lam(self) -> float:
        if not self.enabled:
            return self.lambda0
        return self.lambda0 / (1.0 + self.gap_ema / self.sigma)
