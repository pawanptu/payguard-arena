"""Four deterministic specialist defenders.

Each one maps a payout request to a risk score in [0, 1] (0 = benign) using ONLY the `Event` and the
`PlatformLog` history. Scores are computed from the log BEFORE the event is added (the runner adds it after
`decide`). No specialist ever reads ground truth: labels reach the swarm only through `Swarm.learn`.

Every specialist is a different lens, and each is blind to something on purpose:

  Velocity       sees how fast and how much one account is paid in 24 h. Blind to who pays it.
  Graph          sees who is paid by whom. Blind to amounts, rates and a worker's own past.
  BaselineDrift  sees a worker against ITSELF. Blind to anything with no history (new accounts) and to
                 everyone else.
  Policy         sees limits that scale with account age. Blind to graph structure and to patterns that
                 stay under the limits.

Cost per event is small and bounded: the trailing window and the worker's own history for Velocity, Drift
and Policy (via `PlatformLog.features`), and at most `MAX_NEIGHBORS` neighbour set comparisons for Graph.
`score(event, log)` is the whole interface; the optional `features` argument only lets the coordinator pass
`log.features(event)` once instead of three times (Graph ignores it).
"""
from __future__ import annotations

import math
from itertools import islice

from ..sim.features import PlatformLog
from ..sim.types import Event
from .params import SwarmParams

MAX_NEIGHBORS = 60          # Graph compares a worker with at most this many co-workers of the same requester
_EMPTY: frozenset[int] = frozenset()


def ramp(x: float, lo: float, width: float) -> float:
    """0 at or below `lo`, 1 at or above `lo + width`, linear in between."""
    return min(max((x - lo) / max(width, 1e-9), 0.0), 1.0)


class Velocity:
    """Payout rate and USD volume in the trailing 24 h window of one account.

    Structuring (many payouts kept just under a review threshold) shows up as a window TOTAL far above what
    any single payout would suggest, so the USD ramp is on the window sum, not on the single amount. The
    first payout of a burst looks like a normal one: Velocity cannot see it, only the ones that follow.
    """

    name = "velocity"

    def __init__(self, params: SwarmParams):
        self.p = params

    def score(self, event: Event, log: PlatformLog, features: dict[str, float] | None = None) -> float:
        f = features if features is not None else log.features(event)
        p = self.p
        return max(ramp(f["window_usd"], p.vel_usd_lo, p.vel_usd_width),
                   ramp(f["window_n"], p.vel_n_lo, p.vel_n_width))


class Graph:
    """Collusion structure around the requester that approved this payout.

    A ring is a closed knot: a set of workers who are all paid by (nearly) the same few requesters. For
    the current requester we take the workers it has paid so far, plus this worker, and the union Q of all
    requesters those workers have used. The block density is the average share of Q that a worker in the
    cluster uses: sum(|requesters of w|) / (workers * |Q|). It is a requester-set overlap measure like the
    Jaccard similarity of tests/test_sim.py, but computed over the whole cluster, so it stays high while
    members have revealed only part of the ring's requester pool (pairwise Jaccard collapses there).

      ring (closed, few requesters)          density near 1 and growing with every payout
      popular requester (hundreds of workers, each also paid by an unrelated requester)   density near 0
      small shop (about 3 workers)           density 1, but the size ramp keeps it at 0
      mid-sized honest requester             density 0.2-0.4

    Counting workers per requester is NOT enough: a popular requester pays hundreds of unrelated workers.
    The density is multiplied by a size ramp starting at `graph_min_workers`, because a cluster of 3 is a
    shop. A device or payout destination shared by `graph_share_min` or more workers is a second,
    independent kind of evidence (pairs sharing one are normal).

    What it can and cannot see (important early in a campaign):
      * It needs history. With fewer than `graph_min_workers` distinct workers seen on the requester
        there is nothing collective to see, so the first payouts of a ring always pass this specialist.
      * Workers reveal their requester set one payout at a time. A ring that pays out rarely, or uses many
        requesters, stays ambiguous for days (density is a lower bound until sets are revealed).
      * When the platform log itself is young (first day), every busy honest requester looks like a closed
        cluster too, because each worker has used only one requester so far: early scores are noisy.
      * It looks at no amounts and no rates. It cannot flag a lone account, and a ring that spreads its
        members over many unrelated requesters has no density to find.
    """

    name = "graph"

    def __init__(self, params: SwarmParams):
        self.p = params

    def score(self, event: Event, log: PlatformLog, features: dict[str, float] | None = None) -> float:
        p = self.p
        w = event.worker_id
        community = log.workers_by_requester.get(event.requester_id, _EMPTY)
        size = len(community) + (0 if w in community else 1)

        cluster = 0.0
        if size >= p.graph_min_workers:
            requesters = set(log.requesters_by_worker.get(w, _EMPTY))
            requesters.add(event.requester_id)
            members = len(requesters)           # this worker's own requesters (including the current one)
            seen = 1
            for other in islice((o for o in community if o != w), MAX_NEIGHBORS):
                theirs = log.requesters_by_worker.get(other, _EMPTY)
                requesters |= theirs
                members += len(theirs)
                seen += 1
            density = members / (seen * len(requesters))
            cluster = (ramp(density, p.graph_dens_lo, p.graph_dens_width)
                       * ramp(size, p.graph_min_workers - 1, p.graph_size_width))

        sharing = max(len(log.workers_by_device.get(event.device_id, _EMPTY) | {w}),
                      len(log.workers_by_destination.get(event.destination_id, _EMPTY) | {w}))
        return max(cluster, ramp(sharing, p.graph_share_min - 1, 2.0))


class BaselineDrift:
    """This worker's amount and rate versus its OWN history (a generic per-worker anomaly detector).

    Amount: payout divided by the median of the worker's prior payouts. Rate: z-score of the 24 h count
    against the worker's long-run daily rate (Poisson-style). Both need `drift_min_history` prior payouts;
    with less, the worker has no baseline and the score is 0 (a brand-new account cannot drift from nothing).
    It knows nothing about attack families: any worker whose behaviour jumps away from its own past scores
    high, whether that is a compromised account or a legitimate one-off.
    """

    name = "drift"

    def __init__(self, params: SwarmParams):
        self.p = params

    def score(self, event: Event, log: PlatformLog, features: dict[str, float] | None = None) -> float:
        p = self.p
        f = features if features is not None else log.features(event)
        if f["n_prior"] < max(p.drift_min_history, 3):  # features() needs 3 priors for a median
            return 0.0
        amount = ramp(f["amount_ratio"], p.drift_ratio_lo, p.drift_ratio_width)
        observed_days = max(min(event.t / 24.0, f["age_days"]), 1.0)
        rate = f["n_prior"] / observed_days
        z = (f["window_n"] - rate) / math.sqrt(rate + 1.0)
        return max(amount, ramp(z, p.drift_rate_lo, p.drift_rate_width))


class Policy:
    """Business-policy exposure limits that grow with account age.

    A single payout is limited to `base + per_day * age` (up to the mature-account maximum) and the 24 h
    total to `policy_window_mult` times that. Risk starts at 75% of a limit and is full at 125%. A young
    account asking for a lot is exactly what such a policy exists to stop. A per-requester exposure limit
    is NOT implemented: the platform log keeps no per-requester USD, and defenders hold no state of their own.
    """

    name = "policy"

    def __init__(self, params: SwarmParams):
        self.p = params

    def cap(self, age_days: float) -> float:
        p = self.p
        return min(p.policy_cap_max_usd, p.policy_cap_base_usd + p.policy_cap_per_day_usd * max(age_days, 0.0))

    def score(self, event: Event, log: PlatformLog, features: dict[str, float] | None = None) -> float:
        f = features if features is not None else log.features(event)
        cap = self.cap(f["age_days"])
        return max(ramp(f["amount"] / cap, 0.75, 0.5),
                   ramp(f["window_usd"] / (self.p.policy_window_mult * cap), 0.75, 0.5))


SPECIALISTS = {c.name: c for c in (Velocity, Graph, BaselineDrift, Policy)}


def make_specialist(name: str, params: SwarmParams):
    return SPECIALISTS[name](params)
