"""Honest platform population: four personas plus hard-negative structures.

Hard negatives (so a defender cannot win by flagging anything unusual):
  * `new` workers are young accounts with real payouts,
  * `power` workers sometimes receive large legitimate amounts,
  * `bursty` workers are quiet for days, then get several payouts at once,
  * small shops are tiny closed worker/requester communities (look like a ring to a naive graph rule),
  * a few workers legitimately share a device or a payout destination (family members).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .types import Truth


class Raw(NamedTuple):
    t: float
    worker: int
    requester: int
    device: int
    destination: int
    amount: float
    truth: Truth


class IdAlloc:
    """Hands out fresh ids for attacker-controlled entities so they never collide with honest ones."""

    def __init__(self, workers: int, requesters: int, devices: int, destinations: int):
        self._n = {"w": workers, "r": requesters, "d": devices, "p": destinations}

    def _next(self, k: str) -> int:
        v = self._n[k]
        self._n[k] += 1
        return v

    def worker(self) -> int:
        return self._next("w")

    def requester(self) -> int:
        return self._next("r")

    def device(self) -> int:
        return self._next("d")

    def destination(self) -> int:
        return self._next("p")


PERSONAS = {
    # share of workers, payouts/day (None = bursty), median USD, lognormal sigma, account age range (days)
    "regular": dict(share=0.60, rate=0.6, median=55.0, sigma=0.5, age=(60, 700)),
    "new": dict(share=0.15, rate=1.0, median=40.0, sigma=0.6, age=(0, 10)),
    "power": dict(share=0.08, rate=3.0, median=140.0, sigma=0.6, age=(100, 900)),
    "bursty": dict(share=0.17, rate=None, median=70.0, sigma=0.5, age=(30, 500)),
}
HONEST = Truth(is_attack=False, family="honest", campaign_id=-1)


def draw_amount(rng: np.random.Generator, median: float, sigma: float) -> float:
    return round(float(np.clip(median * np.exp(sigma * rng.standard_normal()), 5.0, 1500.0)), 2)


def honest_population(rng: np.random.Generator, n_workers: int, n_requesters: int, n_shops: int,
                      days: int) -> tuple[list[Raw], dict[int, float]]:
    """Return honest payout requests and account creation times. Requesters 0..n_requesters-1 are
    popular; the next `n_shops` ids are small shops serving 3 fixed workers each."""
    hours = days * 24.0
    names = list(PERSONAS)
    persona = rng.choice(names, size=n_workers, p=[PERSONAS[k]["share"] for k in names])

    weights = 1.0 / np.arange(1, n_requesters + 1) ** 1.1
    weights /= weights.sum()
    homes = [list(rng.choice(n_requesters, size=2, replace=False, p=weights)) for _ in range(n_workers)]

    shop_workers = rng.choice(n_workers, size=min(n_workers, 3 * n_shops), replace=False)
    for j in range(n_shops):
        for w in shop_workers[3 * j: 3 * j + 3]:
            homes[int(w)] = [n_requesters + j, n_requesters + j]

    device = list(range(n_workers))
    destination = list(range(n_workers))
    pair_dev = rng.choice(n_workers, size=2 * int(0.025 * n_workers), replace=False)
    for a, b in zip(pair_dev[0::2], pair_dev[1::2]):
        device[int(b)] = device[int(a)]
    pair_dst = rng.choice(n_workers, size=2 * int(0.015 * n_workers), replace=False)
    for a, b in zip(pair_dst[0::2], pair_dst[1::2]):
        destination[int(b)] = destination[int(a)]

    raws: list[Raw] = []
    accounts: dict[int, float] = {}
    for w in range(n_workers):
        p = PERSONAS[str(persona[w])]
        accounts[w] = -float(rng.uniform(*p["age"])) * 24.0
        if p["rate"] is None:  # bursty
            times: list[float] = []
            for d in range(days):
                if rng.random() < 0.25:
                    k = int(rng.integers(3, 7))
                    times += list(d * 24.0 + rng.uniform(0, 24.0, size=k))
        else:
            n = int(rng.poisson(p["rate"] * days))
            times = list(rng.uniform(0, hours, size=n))
        for t in sorted(times):
            req = homes[w][0] if rng.random() < 0.7 else homes[w][1]
            raws.append(Raw(float(t), w, int(req), device[w], destination[w],
                            draw_amount(rng, p["median"], p["sigma"]), HONEST))
    return raws, accounts
