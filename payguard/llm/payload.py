"""Discretized, truth-free view of one payout for the LLM.

Bucketing is the cost lever: payouts that fall in the same buckets produce the identical payload, hence
the same cache key, hence one API call in total. The payload carries NO ids, NO timestamps and nothing
from `Truth`; every value is a coarse range label computed from history strictly before the event
(call it before `log.add(event)`, as with `PlatformLog.features`).
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Callable

from ..sim.features import PlatformLog
from ..sim.types import Event

# Few fields and few buckets on purpose: on the 20 TRAIN episodes (~84k payouts) this yields ~400-500
# distinct payloads, i.e. a few hundred paid calls on a cold cache. Finer buckets (10 fields) gave 12,000+.
# field -> (value computed from the feature dict, ascending bucket edges, labels = len(edges)+1)
_SPECS: dict[str, tuple[Callable[[dict], float], tuple[float, ...], tuple[str, ...]]] = {
    "account_age": (lambda f: f["age_days"], (7, 60), ("<7d", "7-60d", ">60d")),
    "amount_usd": (lambda f: f["amount"], (75, 250, 400, 500), ("<75", "75-250", "250-400", "400-500", "500+")),
    "payouts_24h": (lambda f: f["window_n"], (3, 5), ("1-2", "3-4", "5+")),
    "amount_vs_own_median": (lambda f: f["amount_ratio"], (2,), ("<2x", "2x+")),
    "workers_sharing_device_or_destination": (
        lambda f: max(f["device_workers"], f["destination_workers"]), (2,), ("1", "2+")),
    "workers_per_requester": (lambda f: f["requester_workers"], (10,), ("<10", "10+")),
}
COHORT_FIELD = "workers_with_similar_requesters"
_COHORT_SPEC = ((4,), ("1-3", "4+"))
NO_HISTORY = "n/a"  # amount_vs_own_median before the worker has 3 prior payouts

LABELS: dict[str, tuple[str, ...]] = {k: v[2] for k, v in _SPECS.items()} | {COHORT_FIELD: _COHORT_SPEC[1]}
FIELDS = tuple(LABELS)
GRAPH_FIELDS = ("workers_sharing_device_or_destination", "workers_per_requester", COHORT_FIELD)
PER_ACCOUNT_FIELDS = tuple(f for f in FIELDS if f not in GRAPH_FIELDS)


def level(payload: dict, field: str) -> int:
    """Ordinal position of the payload's bucket for `field` (0 = lowest); -1 for NO_HISTORY."""
    labels = LABELS[field]
    return labels.index(payload[field]) if payload[field] in labels else -1


def similar_requester_workers(event: Event, log: PlatformLog) -> int:
    """Workers (this one included) whose requester set overlaps this worker's set by Jaccard >= 0.5.

    "This worker's set" includes the requester of the current event. Candidates are the workers of the
    rarest requester in that set, which keeps the cost per payout small.
    """
    mine = log.requesters_by_worker.get(event.worker_id, set()) | {event.requester_id}
    rarest = min(mine, key=lambda r: len(log.workers_by_requester.get(r, ())))
    n = 1
    for w in log.workers_by_requester.get(rarest, ()):
        if w == event.worker_id:
            continue
        theirs = log.requesters_by_worker[w]
        if len(theirs & mine) / len(theirs | mine) >= 0.5:
            n += 1
    return n


def _bucket(x: float, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    return labels[bisect_right(edges, x)]


def judge_payload(event: Event, log: PlatformLog) -> dict[str, str]:
    """Bucketed features of `event` against the history in `log` (call before `log.add(event)`)."""
    f = log.features(event)
    out = {name: _bucket(fn(f), edges, labels) for name, (fn, edges, labels) in _SPECS.items()}
    if f["n_prior"] < 3:  # features() reports a neutral 1.0 here; say "no history" instead
        out["amount_vs_own_median"] = NO_HISTORY
    out[COHORT_FIELD] = _bucket(similar_requester_workers(event, log), *_COHORT_SPEC)
    return out
