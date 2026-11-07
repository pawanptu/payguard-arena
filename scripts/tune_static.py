"""Tune the static-rules baseline (a) ONCE, on TRAIN seeds only, with the pre-registered objective.

Fair-baseline procedure (decision D-020): the baseline gets a real tuning budget on the same data and the
same objective as the swarm, so beating it means something. TEST_SEEN / TEST_HELDOUT are never touched.

Trick: features never depend on earlier verdicts, and resolution randomness is keyed by (seed, event id),
so each event's outcome under all three verdicts is precomputed once; a config is then scored with numpy.
The winner is re-scored with the real runner + frozen metrics to prove the shortcut is exact.

Run from the repo root:  python -m scripts.tune_static
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from payguard.baselines.static_rules import INF, UNTUNED, StaticRuleConfig, StaticRules
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episodes
from payguard.sim.engine import resolve
from payguard.sim.features import PlatformLog
from payguard.sim.metrics import evaluate
from payguard.sim.runner import run_episode
from payguard.sim.types import Verdict

OUT = Path(__file__).resolve().parent.parent / "eval" / "static_rules_tuning.json"
N_RANDOM, SEED = 20_000, 0
GRID = {
    "big_usd": [300, 400, 500, 600, 800, 1000, INF],
    "window_usd": [600, 800, 1000, 1200, 1500, 2000, INF],
    "device_workers": [2, 3, 4, INF],
    "young_days": [1, 3, 7, 14],
    "young_usd": [50, 100, 150, 250, 400, INF],
    "velocity_n": [3, 4, 5, 6, 8, 10, INF],
    "drift_ratio": [2, 3, 4, 6, 8, INF],
}
ACTION_FIELDS = [f.name for f in fields(StaticRuleConfig) if f.name.endswith("_action")]
CODE = {Verdict.RELEASE: 0, Verdict.STEP_UP: 1, Verdict.HOLD: 2}


def objective(leak: float, block: float, unpaid: float) -> float:
    return leak + 0.5 * block + 1.0 * unpaid  # pre-registered in D-019


class Table:
    """Per-event features and the outcome of every verdict, for all TRAIN episodes."""

    def __init__(self, eps):
        cols = {k: [] for k in ("amount", "window_usd", "device_workers", "age_days", "window_n", "amount_ratio")}
        attack, paid = [], {v: [] for v in Verdict}
        for ep in eps:
            log = PlatformLog(ep.accounts)
            for e in ep.events:
                f = log.features(e)
                log.add(e)
                for k in cols:
                    cols[k].append(f[k])
                tr = ep.truth[e.id]
                attack.append(tr.is_attack)
                for v in Verdict:
                    paid[v].append(resolve(ep.seed, e, tr, v).paid)
        self.f = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
        self.attack = np.asarray(attack)
        self.paid = [np.asarray(paid[v]) for v in (Verdict.RELEASE, Verdict.STEP_UP, Verdict.HOLD)]
        self._cache: dict = {}

    def _cond(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def conditions(self, c: StaticRuleConfig):
        f = self.f
        return [
            (self._cond(("big", c.big_usd), lambda: f["amount"] >= c.big_usd), c.big_action),
            (self._cond(("win", c.window_usd), lambda: f["window_usd"] >= c.window_usd), c.window_action),
            (self._cond(("dev", c.device_workers), lambda: f["device_workers"] >= c.device_workers), c.device_action),
            (self._cond(("young", c.young_days, c.young_usd),
                        lambda: (f["age_days"] < c.young_days) & (f["amount"] >= c.young_usd)), c.young_action),
            (self._cond(("vel", c.velocity_n), lambda: f["window_n"] >= c.velocity_n), c.velocity_action),
            (self._cond(("drift", c.drift_ratio), lambda: f["amount_ratio"] >= c.drift_ratio), c.drift_action),
        ]

    def score(self, c: StaticRuleConfig) -> dict[str, float]:
        sev = np.zeros(len(self.attack), dtype=np.int8)
        for cond, action in self.conditions(c):
            np.maximum(sev, cond * np.int8(CODE[action]), out=sev)
        paid = np.choose(sev, self.paid)
        amount, att, hon = self.f["amount"], self.attack, ~self.attack
        leak = 100.0 * float((amount * paid * att).sum() / (amount * att).sum())
        block = 100.0 * float((sev[hon] > 0).mean())
        unpaid = 100.0 * float((~paid[hon]).mean())
        return {"leakage_pct": leak, "honest_block_rate": block, "honest_unpaid_rate": unpaid,
                "objective": objective(leak, block, unpaid)}


def random_config(rng) -> StaticRuleConfig:
    kw = {k: float(vals[rng.integers(len(vals))]) for k, vals in GRID.items()}
    kw.update({a: (Verdict.STEP_UP, Verdict.HOLD)[rng.integers(2)] for a in ACTION_FIELDS})
    return StaticRuleConfig(**kw)


def coordinate_descent(table: Table, best: StaticRuleConfig) -> StaticRuleConfig:
    best_obj = table.score(best)["objective"]
    improved = True
    while improved:
        improved = False
        options = [(k, float(v)) for k, vals in GRID.items() for v in vals]
        options += [(a, v) for a in ACTION_FIELDS for v in (Verdict.STEP_UP, Verdict.HOLD)]
        for name, value in options:
            cand = replace(best, **{name: value})
            obj = table.score(cand)["objective"]
            if obj < best_obj - 1e-9:
                best, best_obj, improved = cand, obj, True
    return best


def jsonable(c: StaticRuleConfig) -> dict:
    return {k: (v.name if isinstance(v, Verdict) else (None if v == INF else v)) for k, v in asdict(c).items()}


def main() -> int:
    eps = episodes(TRAIN_SEEDS, TRAIN_FAMILIES)
    table = Table(eps)
    untuned = UNTUNED
    rng = np.random.default_rng(SEED)
    best = min((random_config(rng) for _ in range(N_RANDOM)), key=lambda c: table.score(c)["objective"])
    best = coordinate_descent(table, best)
    fast = table.score(best)

    real = evaluate([run_episode(ep, StaticRules(best)) for ep in eps])  # exactness check with frozen code
    real_obj = objective(real.leakage_pct, real.honest_block_rate, real.honest_unpaid_rate)
    assert abs(real_obj - fast["objective"]) < 1e-6, (real_obj, fast["objective"])

    report = {"split": "TRAIN only", "seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]], "random_configs": N_RANDOM,
              "search_seed": SEED, "objective": "leakage_pct + 0.5*honest_block_rate + 1.0*honest_unpaid_rate",
              "untuned": {"config": jsonable(untuned), "train": table.score(untuned)},
              "tuned": {"config": jsonable(best), "train": fast,
                        "ring_leakage_pct": real.leakage_pct_by_family.get("ring"),
                        "structuring_leakage_pct": real.leakage_pct_by_family.get("structuring"),
                        "honest_median_ttp_h": real.honest_median_ttp_h}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
