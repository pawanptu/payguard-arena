"""Platform log and the features a defender can compute from it (history only, never ground truth)."""
from __future__ import annotations

from collections import defaultdict
from statistics import median

from .types import Event


class PlatformLog:
    """Append-only history of payout requests seen so far, with the indexes defenders need.

    `features(event)` must be called BEFORE `add(event)`: it describes the event against prior history.
    """

    def __init__(self, accounts: dict[int, float]):
        self.accounts = accounts
        self._hist: dict[int, list[tuple[float, float]]] = defaultdict(list)  # worker -> [(t, amount)]
        self.workers_by_device: dict[int, set[int]] = defaultdict(set)
        self.workers_by_destination: dict[int, set[int]] = defaultdict(set)
        self.workers_by_requester: dict[int, set[int]] = defaultdict(set)
        self.requesters_by_worker: dict[int, set[int]] = defaultdict(set)

    def add(self, e: Event) -> None:
        self._hist[e.worker_id].append((e.t, e.amount))
        self.workers_by_device[e.device_id].add(e.worker_id)
        self.workers_by_destination[e.destination_id].add(e.worker_id)
        self.workers_by_requester[e.requester_id].add(e.worker_id)
        self.requesters_by_worker[e.worker_id].add(e.requester_id)

    def features(self, e: Event) -> dict[str, float]:
        prior = self._hist.get(e.worker_id, [])
        n_win, usd_win = 1, e.amount  # the current request counts toward its own 24h window
        for t, amount in reversed(prior):
            if t <= e.t - 24.0:
                break
            n_win += 1
            usd_win += amount
        n_prior = len(prior)
        ratio = e.amount / max(median(a for _, a in prior), 1.0) if n_prior >= 3 else 1.0
        return {
            "age_days": (e.t - self.accounts[e.worker_id]) / 24.0,
            "amount": e.amount,
            "window_n": float(n_win),
            "window_usd": usd_win,
            "n_prior": float(n_prior),
            "amount_ratio": ratio,
            "device_workers": float(len(self.workers_by_device[e.device_id] | {e.worker_id})),
            "destination_workers": float(len(self.workers_by_destination[e.destination_id] | {e.worker_id})),
            "requester_workers": float(len(self.workers_by_requester[e.requester_id] | {e.worker_id})),
            "worker_requesters": float(len(self.requesters_by_worker[e.worker_id] | {e.requester_id})),
        }
