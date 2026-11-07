"""Frozen metric definitions. All values come from SIMULATED replays.

  leakage_pct             share of attack USD that was actually paid out
  honest_block_rate       share of honest payouts NOT released instantly (HOLD or STEP_UP): friction cost
  honest_unpaid_rate      share of honest payouts never paid (failed step-up or wrongly cancelled review)
  honest_median_ttp_h     median hours from request to payment for honest payouts that were paid
  immunity_attempts       mean number of attack payouts that SUCCEED in a campaign before it is contained,
                          where "contained" = the first point after which 3 consecutive payouts of that
                          campaign are all blocked (HOLD/STEP_UP); a campaign never contained leaks them all
  never_contained_rate    share of campaigns that are never contained
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .runner import RunResult
from .types import FAMILIES, Verdict

CONTAINMENT_RUN = 3


@dataclass
class Metrics:
    n_episodes: int
    attack_usd: float
    leaked_usd: float
    leakage_pct: float
    leakage_pct_by_family: dict[str, float]
    honest_n: int
    honest_block_rate: float
    honest_unpaid_rate: float
    honest_median_ttp_h: float
    honest_p95_ttp_h: float
    immunity_attempts: float
    never_contained_rate: float
    llm_calls: int
    extra: dict = field(default_factory=dict)


def _campaign_leak_before_containment(blocked: list[bool], paid: list[bool]) -> tuple[int, bool]:
    n = len(blocked)
    need = min(CONTAINMENT_RUN, n)
    for i in range(n - need + 1):
        if all(blocked[i:i + need]):
            return sum(paid[:i]), True
    return sum(paid), False


def evaluate(results: list[RunResult]) -> Metrics:
    attack_usd = leaked_usd = 0.0
    fam_attack: dict[str, float] = defaultdict(float)
    fam_leak: dict[str, float] = defaultdict(float)
    honest_n = honest_block = honest_unpaid = 0
    ttp: list[float] = []
    campaigns: dict[tuple[int, int], list[tuple[bool, bool]]] = defaultdict(list)
    for idx, r in enumerate(results):
        for e in r.episode.events:
            tr, oc, v = r.episode.truth[e.id], r.outcomes[e.id], r.verdicts[e.id]
            if tr.is_attack:
                attack_usd += e.amount
                fam_attack[tr.family] += e.amount
                if oc.paid:
                    leaked_usd += e.amount
                    fam_leak[tr.family] += e.amount
                campaigns[(idx, tr.campaign_id)].append((v is not Verdict.RELEASE, oc.paid))
            else:
                honest_n += 1
                honest_block += v is not Verdict.RELEASE
                if oc.paid:
                    ttp.append(oc.paid_at - e.t)
                else:
                    honest_unpaid += 1
    leaks, contained = [], 0
    for rows in campaigns.values():
        n_leak, ok = _campaign_leak_before_containment([b for b, _ in rows], [p for _, p in rows])
        leaks.append(n_leak)
        contained += ok
    pct = lambda a, b: 100.0 * a / b if b else 0.0  # noqa: E731
    return Metrics(
        n_episodes=len(results),
        attack_usd=attack_usd,
        leaked_usd=leaked_usd,
        leakage_pct=pct(leaked_usd, attack_usd),
        leakage_pct_by_family={f: pct(fam_leak[f], fam_attack[f]) for f in FAMILIES if fam_attack[f] > 0},
        honest_n=honest_n,
        honest_block_rate=pct(honest_block, honest_n),
        honest_unpaid_rate=pct(honest_unpaid, honest_n),
        honest_median_ttp_h=float(np.median(ttp)) if ttp else float("nan"),
        honest_p95_ttp_h=float(np.percentile(ttp, 95)) if ttp else float("nan"),
        immunity_attempts=float(np.mean(leaks)) if leaks else 0.0,
        never_contained_rate=pct(len(leaks) - contained, len(leaks)),
        llm_calls=sum(r.llm_calls for r in results),
    )
