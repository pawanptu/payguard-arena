"""DREAM ENGINE: the object the evolution loop talks to (implements `DreamHooks`) (P3).

Per generation the loop calls, in this order:

    engine.begin_generation(g, observations)      fit world model, run skeptic, draw dreams
    fits = engine.dream_fitness(candidates, ev)   predicted objective of each candidate (one batch on `ev`)
    fit  = engine.blend(real, fits)               (1 - lam) * real + lam * dream
    ... real gate decides promotion ...
    engine.end_generation(pg, rg)                 pg, rg = `predicted_gain(...)` / realized gain; None, None if no promotion
    log  = engine.record()                        JSON-serializable lineage entry

Arm switches (all in `DreamConfig`): use_wm False -> `UniformWorldModel` (no fit on observations, and the imaginer
draws `uniform_dreams`); use_skeptic False -> `FlatReliability` and every dream weighs 1; use_calibration False ->
`lam()` stays lambda0.

Randomness: every stage draws from `np.random.default_rng([base_seed, g, stage])` (stages: fit 1, assess 2, dream 3),
so a generation's dreams are a pure function of (base_seed, g, observations) and do not depend on call history.

GAIN CONVENTION (used by `predicted_gain` and by the loop for the realized side): gain = champion objective minus
challenger objective, both lower-is-better, so a POSITIVE gain means the challenger is better. Predicted side: the
dream objectives from `dream_fitness` (same dreams for both). Realized side: the real gate-set objectives.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from ..evolve.splits import TRAIN_A
from ..swarm.params import SwarmParams
from .calibrate import EmaCalibrator
from .contract import (Dream, DreamConfig, DreamScore, FlatReliability, Observation, ReliabilityMap, Skeptic,
                       UniformWorldModel, WorldModel)
from .evaluate import predict_many
from .imaginer import Imaginer

STAGE_FIT, STAGE_ASSESS, STAGE_DREAM = 1, 2, 3


def predicted_gain(champion_fit: float, challenger_fit: float) -> float:
    """champion objective minus challenger objective (positive = challenger better). Works for dream and real fitness."""
    return float(champion_fit) - float(challenger_fit)


class DreamEngine:
    def __init__(self, cfg: DreamConfig, make_model: Callable[[], WorldModel], skeptic: Skeptic,
                 seeds: Sequence[int] = TRAIN_A, base_seed: int = 0):
        self.cfg = cfg
        self.make_model: Callable[[], WorldModel] = make_model if cfg.use_wm else UniformWorldModel
        self.skeptic = skeptic
        self.seeds = tuple(seeds)
        self.base_seed = int(base_seed)
        self.imaginer = Imaginer(cfg)
        self.calibrator = EmaCalibrator(cfg.lambda0, cfg.sigma, cfg.ema_beta, enabled=cfg.use_calibration)
        self.g: int | None = None
        self.model: WorldModel | None = None
        self.reliability: ReliabilityMap | None = None
        self.dreams: list[Dream] = []
        self.weights: tuple[float, ...] = ()
        self.lam_used: float = cfg.lambda0
        self.n_obs = 0
        self.fit_seconds = 0.0
        self.last_scores: list[DreamScore] = []
        self.last_gap: dict[str, float] | None = None

    def _rng(self, stage: int) -> np.random.Generator:
        assert self.g is not None
        return np.random.default_rng([self.base_seed, self.g, stage])

    # ------------------------------------------------------------------------------------------ DreamHooks
    def begin_generation(self, g: int, observations: Sequence[Observation]) -> None:
        self.g = int(g)
        self.n_obs = len(observations)
        self.last_scores, self.last_gap = [], None
        t0 = time.perf_counter()
        if self.cfg.use_wm:
            self.model = self.make_model().fit(observations, self._rng(STAGE_FIT))
        else:
            self.model = UniformWorldModel()
        if self.cfg.use_skeptic:
            self.reliability = self.skeptic.assess(observations, self.make_model, self._rng(STAGE_ASSESS))
        else:
            self.reliability = FlatReliability()
        self.fit_seconds = time.perf_counter() - t0
        self.dreams = self.imaginer.dream(self.model, self.reliability, self.cfg.n_dreams, self._rng(STAGE_DREAM),
                                          self.cfg.families)
        self.weights = tuple(1.0 if not self.cfg.use_skeptic else float(self.reliability.weight(d, self.cfg.tau))
                             for d in self.dreams)
        self.lam_used = self.calibrator.lam()

    def dream_fitness(self, candidates: Sequence[SwarmParams], evaluator: Any) -> list[float]:
        if self.reliability is None:
            raise RuntimeError("begin_generation must be called before dream_fitness")
        self.last_scores = predict_many(candidates, self.dreams, self.reliability, seeds=self.seeds, evaluator=evaluator,
                                        tau=self.cfg.tau, use_skeptic=self.cfg.use_skeptic)
        return [s.objective for s in self.last_scores]

    def blend(self, real_fitness: Sequence[float], dream_fitness: Sequence[float]) -> list[float]:
        if len(real_fitness) != len(dream_fitness):
            raise ValueError("real and dream fitness must have the same length")
        lam = self.calibrator.lam()
        return [(1.0 - lam) * float(r) + lam * float(d) for r, d in zip(real_fitness, dream_fitness)]

    def end_generation(self, predicted_gain: float | None, realized_gain: float | None) -> None:
        self.calibrator.update(predicted_gain, realized_gain)
        if predicted_gain is not None and realized_gain is not None:
            self.last_gap = {"predicted_gain": float(predicted_gain), "realized_gain": float(realized_gain),
                             "gap": abs(float(predicted_gain) - float(realized_gain))}

    def record(self) -> dict:
        w = self.weights
        counts = dict(self.imaginer.last_counts)
        out = {"g": self.g, "lam": float(self.lam_used), "lam_next": float(self.calibrator.lam()),
               "gap_ema": float(self.calibrator.gap_ema), "n_dreams": len(self.dreams),
               "mean_weight": float(np.mean(w)) if w else None, "min_weight": float(min(w)) if w else None,
               "share_counts": {k: int(v) for k, v in counts.items()}, "n_observations": int(self.n_obs),
               "fit_seconds": float(self.fit_seconds), "arm": {"use_wm": self.cfg.use_wm,
                                                               "use_skeptic": self.cfg.use_skeptic,
                                                               "use_calibration": self.cfg.use_calibration}}
        if self.last_gap is not None:
            out.update(self.last_gap)
        return out

    # ------------------------------------------------------------------------------------------ convenience
    @staticmethod
    def predicted_gain(champion_fit: float, challenger_fit: float) -> float:
        return predicted_gain(champion_fit, challenger_fit)
