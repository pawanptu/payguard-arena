"""Round-3 'dream' package: the shared contract (P0). Every other module in `payguard.dream` codes against THIS file.

Idea (docs/PREREG_R3.md, in the project docs folder): the defender swarm builds a WORLD MODEL of the attackers from
what it has actually observed, DREAMS attackers from it (plus uncertain regions and recombinations), evaluates
challenger defenders against the dreams, and CALIBRATES how much it trusts its dreams by comparing predicted with
realized gains. A SKEPTIC marks the regions where the world model is unreliable. The promotion gate stays real.

Rules every module must follow (the benchmark's honesty depends on them):
  * Observations carry NO ground-truth family and NO campaign id: only `Truth.is_attack` per event, as a delayed
    label would deliver it (`observe`). A world model that peeks at `Truth.family` voids the run.
  * Seeds: only TRAIN seeds (1000-1019) and dev seeds (9000-9999) may be built. `assert_seed_allowed` raises on
    anything else; only the one-shot final script passes `allow_test=True`.
  * Everything random takes an explicit `np.random.Generator`; same inputs, same outputs, any worker count.
  * No file under payguard/sim/, payguard/protocol.py, payguard/baselines/ is edited (frozen).
  * Dreams are lists of campaigns that existing generators can build: `(family, params dataclass)` with
    `family in payguard.sim.attacks.GENERATORS`; nothing new is registered in the simulator.

Arms (PREREG_R3 section 3) are switches of `DreamConfig`: dreamer_full, dreamer_nowm (uniform dreams, same
parameter ranges), dreamer_noskeptic (reliability weight fixed at 1), dreamer_nocalib (lambda fixed at lambda0).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..protocol import TRAIN_FAMILIES
from ..sim.attacks import BOUNDS, INT_FIELDS, PARAMS, sample_params
from ..sim.types import Episode, Event
from ..swarm.params import SwarmParams

# ---------------------------------------------------------------------------------------------------- seeds
ALLOWED_SEED_RANGES: tuple[tuple[int, int], ...] = ((1000, 1020), (9000, 10000))  # [lo, hi): TRAIN, dev


def assert_seed_allowed(seed: int, *, allow_test: bool = False) -> None:
    """Raise unless `seed` is a TRAIN or dev seed. TEST seeds (2000-5019) need `allow_test=True` (final script only)."""
    if allow_test:
        return
    if not any(lo <= seed < hi for lo, hi in ALLOWED_SEED_RANGES):
        raise AssertionError(f"seed {seed} is not a TRAIN (1000-1019) or dev (9000-9999) seed")


# ---------------------------------------------------------------------------------------------------- data types
Campaign = tuple[str, Any]          # (family, params dataclass instance of payguard.sim.attacks.PARAMS[family])
Dream = tuple[Campaign, ...]        # the campaigns of ONE imagined episode (>= 1); built with evolve.evaluation.arena_spec
Region = tuple[str, tuple[int, ...]]  # (family, bin index per parameter field, in dataclass field order)
N_BINS = 4


@dataclass(frozen=True)
class Observation:
    """What the swarm has seen of ONE episode: events, the labels that have arrived, account creation times."""

    seed: int
    hours: float
    events: tuple[Event, ...]
    confirmed_attack_ids: frozenset[int]     # events whose delayed label says "attack" (Truth.is_attack only)
    accounts: Mapping[int, float]            # worker_id -> creation time in hours (negative = before start)


def observe(ep: Episode) -> Observation:
    """Strip an episode down to what a production swarm could know once labels arrive. Uses `is_attack` ONLY."""
    assert_seed_allowed(ep.seed)
    return Observation(ep.seed, ep.hours, tuple(ep.events),
                       frozenset(i for i, t in ep.truth.items() if t.is_attack), dict(ep.accounts))


def region_of(campaign: Campaign, n_bins: int = N_BINS) -> Region:
    """Discretize a campaign's parameters into a grid cell: `n_bins` equal-width bins per field inside BOUNDS."""
    family, p = campaign
    idx = []
    for f in fields(p):
        lo, hi = BOUNDS[family][f.name]
        v = float(getattr(p, f.name))
        k = int((v - lo) / (hi - lo) * n_bins) if hi > lo else 0
        idx.append(min(max(k, 0), n_bins - 1))
    return family, tuple(idx)


# ---------------------------------------------------------------------------------------------------- components
@runtime_checkable
class WorldModel(Protocol):
    """Learns a distribution over attacker campaigns from observations. Immutable style: `fit` returns a model."""

    def fit(self, observations: Sequence[Observation], rng: np.random.Generator) -> "WorldModel": ...

    def sample(self, n: int, rng: np.random.Generator, families: tuple[str, ...] = TRAIN_FAMILIES) -> list[Dream]:
        """`n` posterior dreams (each a Dream). Must work (e.g. fall back to uniform) before any fit."""
        ...


@runtime_checkable
class ReliabilityMap(Protocol):
    """Where the world model can be trusted. Error 0 = reliable; a region never checked gets `max_error`."""

    max_error: float

    def error(self, region: Region) -> float: ...

    def weight(self, dream: Dream, tau: float) -> float:
        """In (0, 1]: mean over the dream's campaigns of 1 / (1 + error(region) / tau)."""
        ...

    def uncertain_regions(self, k: int) -> list[Region]:
        """The `k` highest-error regions, most uncertain first (deterministic order)."""
        ...


@runtime_checkable
class Skeptic(Protocol):
    def assess(self, observations: Sequence[Observation], make_model: Callable[[], WorldModel],
               rng: np.random.Generator) -> ReliabilityMap:
        """Fit `make_model()` on part of the observations, test its predictions on the held-back part."""
        ...


@dataclass(frozen=True)
class DreamScore:
    objective: float                  # reliability-weighted mean objective of the defender over the dreams (lower = better)
    per_dream: tuple[float, ...]
    weights: tuple[float, ...]
    leakage_pct: float
    honest_block_rate: float


@runtime_checkable
class Calibrator(Protocol):
    """Trust in the swarm's own imagination. lam() = lambda0 / (1 + gap_ema / sigma), in (0, lambda0]."""

    gap_ema: float
    history: list[dict]

    def update(self, predicted_gain: float, realized_gain: float) -> None:
        """gap = |predicted - realized| (objective points); EMA with `ema_beta`; append a history record."""
        ...

    def lam(self) -> float: ...


@runtime_checkable
class DreamHooks(Protocol):
    """What the evolution loop (P5) calls. Implemented by `payguard.dream.engine.DreamEngine` (P3)."""

    def begin_generation(self, g: int, observations: Sequence[Observation]) -> None:
        """Fit the world model, run the skeptic, draw this generation's dreams. Called once per generation."""
        ...

    def dream_fitness(self, candidates: Sequence[SwarmParams], evaluator: Any) -> list[float]:
        """Predicted objective of each candidate on this generation's dreams (same order). `evaluator` is
        `payguard.evolve.evaluation.Evaluator`; dream episodes use TRAIN_A seeds as honest traffic."""
        ...

    def blend(self, real_fitness: Sequence[float], dream_fitness: Sequence[float]) -> list[float]:
        """(1 - lam) * real + lam * dream, per candidate, with lam from the calibrator."""
        ...

    def end_generation(self, predicted_gain: float | None, realized_gain: float | None) -> None:
        """Gains of the promoted challenger over the champion, predicted (dreams) vs realized (real gate set).
        None, None when nobody was promoted: the calibrator then keeps its state."""
        ...

    def record(self) -> dict:
        """JSON-serializable log of this generation (lam, gap_ema, n dreams, mean weight, ...) for the lineage."""
        ...


# ---------------------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DreamConfig:
    n_dreams: int = 32
    share_posterior: float = 0.5        # shares of the dreams: posterior / uncertain regions / recombinations (sum to 1)
    share_uncertain: float = 0.25
    share_recombined: float = 0.25
    tau: float = 1.0                    # reliability weight 1 / (1 + err / tau); see eval/r3/SELECTION_RULE.md
    sigma: float = 1.0                  # calibration scale in objective points
    lambda0: float = 0.3                # maximum weight of dreamed fitness, in (0, 1]
    ema_beta: float = 0.5               # gap_ema = beta * gap + (1 - beta) * gap_ema
    n_bins: int = N_BINS
    use_wm: bool = True                 # False -> dreams drawn uniformly (dreamer_nowm)
    use_skeptic: bool = True            # False -> every dream weighs 1 (dreamer_noskeptic)
    use_calibration: bool = True        # False -> lam fixed at lambda0 (dreamer_nocalib)
    families: tuple[str, ...] = TRAIN_FAMILIES

    def __post_init__(self) -> None:
        s = self.share_posterior + self.share_uncertain + self.share_recombined
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"dream shares must sum to 1, got {s}")
        if not (0 < self.lambda0 <= 1):
            raise ValueError("lambda0 must be in (0, 1]")

    @classmethod
    def for_arm(cls, arm: str, **kw: Any) -> "DreamConfig":
        base = cls(**kw)
        flags = {"dreamer_full": {}, "dreamer_nowm": {"use_wm": False}, "dreamer_noskeptic": {"use_skeptic": False},
                 "dreamer_nocalib": {"use_calibration": False}}
        if arm not in flags:
            raise KeyError(arm)
        return replace(base, **flags[arm])


# ---------------------------------------------------------------------------------------------------- stand-ins
def uniform_dreams(n: int, rng: np.random.Generator, families: tuple[str, ...] = TRAIN_FAMILIES) -> list[Dream]:
    """Reference sampler for the no-world-model arm: single-campaign dreams, round-robin over `families`,
    parameters uniform inside BOUNDS (same ranges the world model works in)."""
    return [((fam, sample_params(fam, rng)),) for fam in (families[i % len(families)] for i in range(n))]


class UniformWorldModel:
    """World model that knows nothing: `fit` is a no-op, `sample` is `uniform_dreams`. Stand-in for tests and the nowm arm."""

    def fit(self, observations: Sequence[Observation], rng: np.random.Generator) -> "UniformWorldModel":
        return self

    def sample(self, n: int, rng: np.random.Generator, families: tuple[str, ...] = TRAIN_FAMILIES) -> list[Dream]:
        return uniform_dreams(n, rng, families)


class FlatReliability:
    """Everything is reliable (error 0): weight 1. Stand-in for tests and the no-skeptic arm."""

    max_error = 0.0

    def error(self, region: Region) -> float:
        return 0.0

    def weight(self, dream: Dream, tau: float) -> float:
        return 1.0

    def uncertain_regions(self, k: int) -> list[Region]:
        return []


class FlatSkeptic:
    def assess(self, observations: Sequence[Observation], make_model: Callable[[], WorldModel],
               rng: np.random.Generator) -> FlatReliability:
        return FlatReliability()
