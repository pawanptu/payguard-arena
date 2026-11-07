"""Baseline (f): the swarm plus SHARED MEMORY, aimed at faster time-to-immunity.

Why. A campaign's first payouts look normal to every specialist (a ring is clean per account; a structuring
burst is only visible from its 2nd/3rd payout). Once ONE payout of the campaign is confirmed as fraud, its
label (`learn(event, is_attack=True)`, 4-72 h after the request) says which entities the campaign uses. The
siblings of that payout (other ring workers paid by the same requesters, the same account's next burst) can
then be stopped sooner than the specialists could see them.

What is stored. For every entity the confirmed-fraud payout touched (worker, device, destination, requester;
kept apart by kind, because ids of different kinds may collide numerically):

    A  attack mass   += strength[kind] for every attack label        (decays with half-life `half_life_h`)
    H  honest mass   += 1 for every honest label                      (same decay)
    taint(kind, id)   = min(A, 1) * A / (A + honest_relief * H)

so honest labels can only weaken a taint (a requester with many confirmed-honest payouts is a popular
platform requester, not a ring's puppeteer). Decay runs on simulated time: a label that arrives late is aged
from the REQUEST time `event.t` of the payout it describes, so slow evidence counts for less. Memory sees
labels ONLY through `learn()` and never anything else; it is per-episode (a new instance per run).

Score. At decide time, memory_score = clip(max over the event's entities of taint * weight[kind], 0, 1);
0 when nothing the event touches is tainted (memory abstains). It enters the coordinator's
`sum(w*s^2)/sum(w*s)` aggregation as a fifth voice with the fixed weight `w_memory`. The four specialists
and their L1 learning are exactly the parent's (their scores are stored and updated by `Swarm` unchanged).
Memory has no reputation of its own to learn: its strengths and weights are fixed parameters.

Kinds. Rings share REQUESTERS (every account has its own device and destination), so requester memory is the
channel that reaches un-attacked ring siblings. Worker/device/destination memory only helps against the same
account coming back (structuring bursts, a ring worker's next payout). Structuring may use ordinary
platform requesters, so requester taint is dangerous for honest workers: the honest relief and the per-kind
weight are what keep it safe, and each kind's contribution is measured separately (scripts/run_memory.py).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from ..sim.features import PlatformLog
from ..sim.types import Event, Verdict
from .coordinator import NAMES, Swarm
from .params import SwarmParams, default_params

KINDS = ("worker", "device", "destination", "requester")
TAINT_FLOOR = 0.01  # a decayed taint below this counts as "nothing tainted"


@dataclass(frozen=True)
class MemoryParams:
    # strength: taint added to an entity by one confirmed-fraud payout (taint saturates at 1)
    strength_worker: float = 0.6
    strength_device: float = 0.6
    strength_destination: float = 0.6
    strength_requester: float = 1.0
    # weight: multiplier of a kind's taint in the memory score (its ceiling); 0 switches the kind off
    weight_worker: float = 1.0
    weight_device: float = 1.0
    weight_destination: float = 1.0
    weight_requester: float = 1.0
    half_life_h: float = 96.0      # hours (simulated) for a taint or an honest count to halve
    honest_relief: float = 5.0     # how much one (decayed) confirmed-honest payout of an entity counts against its fraud mass
    w_memory: float = 0.5          # the memory voice's weight in the coordinator aggregation


BOUNDS: dict[str, tuple[float, float]] = dict(
    strength_worker=(0.05, 1.0), strength_device=(0.05, 1.0),
    strength_destination=(0.05, 1.0), strength_requester=(0.05, 1.0),
    weight_worker=(0.0, 1.0), weight_device=(0.0, 1.0),
    weight_destination=(0.0, 1.0), weight_requester=(0.0, 1.0),
    half_life_h=(12.0, 336.0), honest_relief=(0.0, 20.0), w_memory=(0.05, 3.0),
)
INT_FIELDS: set[str] = set()


def default_memory_params() -> MemoryParams:
    return MemoryParams()


def sample_memory_params(rng: np.random.Generator) -> MemoryParams:
    return MemoryParams(**{n: float(rng.uniform(lo, hi)) for n, (lo, hi) in BOUNDS.items()})


def mutate_memory_params(params: MemoryParams, rng: np.random.Generator, scale: float = 0.2) -> MemoryParams:
    kw = {}
    for f in fields(params):
        lo, hi = BOUNDS[f.name]
        kw[f.name] = float(np.clip(getattr(params, f.name) + rng.normal(0, scale * (hi - lo)), lo, hi))
    return replace(params, **kw)


class _Mass:
    """A decaying sum stored as (value at `t_ref`, t_ref); lazily aged to any later time."""

    __slots__ = ("v", "t")

    def __init__(self) -> None:
        self.v = 0.0
        self.t = 0.0

    def at(self, now: float, half_life: float) -> float:
        dt = now - self.t
        return self.v if dt <= 0.0 else self.v * 0.5 ** (dt / half_life)

    def add(self, amount: float, t: float, half_life: float) -> None:
        ref = max(self.t, t) if self.v > 0.0 else t
        self.v = self.at(ref, half_life) + amount * 0.5 ** (max(ref - t, 0.0) / half_life)
        self.t = ref


class EntityMemory:
    """Decaying attack / honest mass per (kind, id). Pure bookkeeping; knows nothing about verdicts."""

    def __init__(self, params: MemoryParams):
        self.p = params
        self.attack: dict[tuple[str, int], _Mass] = {}
        self.honest: dict[tuple[str, int], _Mass] = {}

    @staticmethod
    def keys(event: Event) -> tuple[tuple[str, int], ...]:
        return (("worker", event.worker_id), ("device", event.device_id),
                ("destination", event.destination_id), ("requester", event.requester_id))

    def observe(self, event: Event, is_attack: bool) -> None:
        hl = self.p.half_life_h
        if is_attack:
            for kind, ident in self.keys(event):
                self.attack.setdefault((kind, ident), _Mass()).add(getattr(self.p, f"strength_{kind}"), event.t, hl)
        else:
            for key in self.keys(event):  # tracked for every entity: honest history may predate the first fraud label
                self.honest.setdefault(key, _Mass()).add(1.0, event.t, hl)

    def taint(self, key: tuple[str, int], now: float) -> float:
        a = self.attack.get(key)
        if a is None:
            return 0.0
        hl = self.p.half_life_h
        h = self.honest.get(key)
        mass = a.at(now, hl)
        if h is None or mass <= 0.0:
            return min(mass, 1.0)
        # precision of the entity: its fraud mass against honest_relief x its honest mass (both decayed)
        return min(mass, 1.0) * mass / (mass + self.p.honest_relief * h.at(now, hl))

    def score(self, event: Event) -> tuple[float, str | None]:
        """(memory score in [0, 1], the kind that produced it or None)."""
        best, best_kind = 0.0, None
        for key in self.keys(event):
            t = self.taint(key, event.t)
            if t < TAINT_FLOOR:
                continue
            s = min(t * getattr(self.p, f"weight_{key[0]}"), 1.0)
            if s > best:
                best, best_kind = s, key[0]
        return best, best_kind


class MemorySwarm(Swarm):
    """The L1 swarm with a fifth voice, `memory`. Implements `payguard.sim.runner.Defender`."""

    def __init__(self, params: SwarmParams | None = None, memory_params: MemoryParams | None = None,
                 *, learn_weights: bool = True, name: str = "swarm_memory"):
        super().__init__(params if params is not None else default_params(), learn_weights=learn_weights)
        self.memory_params = memory_params if memory_params is not None else default_memory_params()
        self.memory = EntityMemory(self.memory_params)
        self.name = name
        self._memory_score = 0.0

    def aggregate(self, scores: dict[str, float]) -> float:
        """The parent's weighted average with the memory voice added (memory score set by `decide`)."""
        s = self._memory_score
        num = den = 0.0
        for n in NAMES:
            num += self.weights[n] * scores[n] * scores[n]
            den += self.weights[n] * scores[n]
        num += self.memory_params.w_memory * s * s
        den += self.memory_params.w_memory * s
        return min(num / den, 1.0) if den > 1e-12 else 0.0

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        self._memory_score, matched = self.memory.score(event)
        # Parent: computes the four specialist scores, calls our aggregate, stores the four scores for L1.
        verdict = super().decide(event, log)
        sig = self.last_signals
        sig["memory"] = self._memory_score
        sig["memory_match"] = matched
        if self._memory_score >= sig["scores"][sig["top"]] and self._memory_score > 0.0:
            sig["top"] = "memory"
        if self._memory_score >= 0.05:
            sig["active"] = [*sig["active"], "memory"]
        return verdict

    def learn(self, event: Event, is_attack: bool) -> None:
        super().learn(event, is_attack)        # the four specialists' L1 update, exactly as the parent
        self.memory.observe(event, is_attack)  # then remember what this label says about the entities


def make_memory_swarm(params: SwarmParams | None = None, memory_params: MemoryParams | None = None,
                      name: str = "swarm_memory") -> MemorySwarm:
    return MemorySwarm(params, memory_params, name=name)


__all__ = ["KINDS", "BOUNDS", "INT_FIELDS", "MemoryParams", "EntityMemory", "MemorySwarm", "make_memory_swarm",
           "default_memory_params", "sample_memory_params", "mutate_memory_params"]
