"""Run one defender over one episode, delivering ground-truth labels only after their delay."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Protocol

from .engine import Outcome, resolve
from .features import PlatformLog
from .types import Episode, Event, Verdict


class Defender(Protocol):
    name: str

    def decide(self, event: Event, log: PlatformLog) -> Verdict: ...

    def learn(self, event: Event, is_attack: bool) -> None:
        """Delayed ground-truth label for an earlier event (chargeback or review result)."""


@dataclass
class RunResult:
    episode: Episode
    verdicts: dict[int, Verdict]
    outcomes: dict[int, Outcome]
    llm_calls: int = 0


def run_episode(episode: Episode, defender: Defender) -> RunResult:
    log = PlatformLog(episode.accounts)
    by_id = {e.id: e for e in episode.events}
    pending: list[tuple[float, int]] = []
    verdicts: dict[int, Verdict] = {}
    outcomes: dict[int, Outcome] = {}
    for e in episode.events:
        while pending and pending[0][0] <= e.t:
            _, eid = heapq.heappop(pending)
            defender.learn(by_id[eid], episode.truth[eid].is_attack)
        v = defender.decide(e, log)
        log.add(e)
        oc = resolve(episode.seed, e, episode.truth[e.id], v)
        verdicts[e.id], outcomes[e.id] = v, oc
        heapq.heappush(pending, (oc.label_at, e.id))
    return RunResult(episode, verdicts, outcomes, int(getattr(defender, "llm_calls", 0)))
