"""Numeric parameters of the swarm: the genome that generation-level evolution (L2) mutates.

Everything a specialist or the coordinator treats as a threshold or a scale lives here, as plain numbers
(floats and ints only), so that `sample_params` / `mutate` can search the space and a candidate can be
stored as a flat JSON object. Conventions mirror payguard/sim/attacks.py (`BOUNDS`, `INT_FIELDS`,
`sample_params`, `mutate`).

Soft thresholds come as a pair (`*_lo`, `*_width`): a signal contributes 0 at or below `lo`, 1 at or above
`lo + width`, and rises linearly in between. Bounds keep every width positive, so the pair stays valid under
any mutation. The two verdict thresholds have disjoint bounds so that step_up < hold always.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np


@dataclass(frozen=True)
class SwarmParams:
    # --- Velocity: payout rate and USD volume in the trailing 24 h window (per account only) ---
    vel_usd_lo: float = 1000.0          # window USD at which risk starts (about the top 4% of honest 24 h volumes)
    vel_usd_width: float = 1000.0
    vel_n_lo: float = 7.0               # payouts in the window at which risk starts
    vel_n_width: float = 4.0

    # --- Graph: collusion structure over the worker / requester / device / destination graph ---
    graph_dens_lo: float = 0.4          # block density (share of the cluster's requesters each worker uses) at which risk starts
    graph_dens_width: float = 0.2
    graph_min_workers: int = 4          # cluster size at which risk starts (small shops have 3 workers)
    graph_size_width: float = 2.0
    graph_share_min: int = 3            # workers on one device or destination at which risk starts (pairs are normal)

    # --- BaselineDrift: this worker's amount and rate versus its own history ---
    drift_min_history: int = 4          # prior payouts needed before the worker has a baseline at all
    drift_ratio_lo: float = 3.0         # amount / own median at which risk starts
    drift_ratio_width: float = 3.0
    drift_rate_lo: float = 3.5          # z-score of the 24 h count versus own long-run rate at which risk starts
    drift_rate_width: float = 2.0

    # --- Policy: exposure caps (account-age-scaled) ---
    policy_cap_base_usd: float = 150.0      # single-payout limit of a brand-new account
    policy_cap_per_day_usd: float = 8.0     # limit growth per day of account age
    policy_cap_max_usd: float = 500.0       # limit of a mature account (the platform review threshold)
    policy_window_mult: float = 3.0         # 24 h limit as a multiple of the single-payout limit

    # --- Coordinator ---
    w0_velocity: float = 1.0            # initial reputation weights (normalised; the frozen swarm keeps them)
    w0_graph: float = 1.0
    w0_drift: float = 1.0
    w0_policy: float = 1.0
    step_up_threshold: float = 0.3      # aggregated risk at which the payout needs extra verification
    hold_threshold: float = 0.6         # aggregated risk at which the payout goes to human review
    eta: float = 0.02                   # L1 learning rate of the multiplicative-weights update
    min_weight: float = 0.05            # no specialist's weight ever falls below this
    l1_abstain: int = 0                 # 1 = abstention-aware L1 loss: a silent specialist is not blamed for a missed attack


BOUNDS: dict[str, tuple[float, float]] = dict(
    vel_usd_lo=(200.0, 1500.0), vel_usd_width=(200.0, 2000.0),
    vel_n_lo=(2.0, 10.0), vel_n_width=(1.0, 8.0),
    graph_dens_lo=(0.2, 0.9), graph_dens_width=(0.05, 0.6),
    graph_min_workers=(3, 8), graph_size_width=(1.0, 10.0), graph_share_min=(3, 6),
    drift_min_history=(3, 10), drift_ratio_lo=(1.5, 6.0), drift_ratio_width=(0.5, 6.0),
    drift_rate_lo=(1.5, 6.0), drift_rate_width=(0.5, 4.0),
    policy_cap_base_usd=(50.0, 400.0), policy_cap_per_day_usd=(1.0, 20.0),
    policy_cap_max_usd=(250.0, 1000.0), policy_window_mult=(1.5, 6.0),
    w0_velocity=(0.1, 1.0), w0_graph=(0.1, 1.0), w0_drift=(0.1, 1.0), w0_policy=(0.1, 1.0),
    step_up_threshold=(0.1, 0.5), hold_threshold=(0.55, 0.95),
    eta=(0.005, 0.5), min_weight=(0.01, 0.2), l1_abstain=(0, 1),
)
INT_FIELDS = {"graph_min_workers", "graph_share_min", "drift_min_history", "l1_abstain"}


def default_params() -> SwarmParams:
    return SwarmParams()


def sample_params(rng: np.random.Generator) -> SwarmParams:
    """Uniform draw inside the bounds."""
    kw = {}
    for name, (lo, hi) in BOUNDS.items():
        kw[name] = int(rng.integers(lo, hi + 1)) if name in INT_FIELDS else float(rng.uniform(lo, hi))
    return SwarmParams(**kw)


def mutate(params: SwarmParams, rng: np.random.Generator, scale: float = 0.2) -> SwarmParams:
    """Numeric mutation: gaussian jitter of each float (sigma = scale * range), clipped to bounds;
    integers move by at most 1. Deterministic given the Generator state."""
    kw = {}
    for f in fields(params):
        lo, hi = BOUNDS[f.name]
        v = getattr(params, f.name)
        if f.name in INT_FIELDS:
            v = int(np.clip(v + rng.integers(-1, 2), lo, hi))
        else:
            v = float(np.clip(v + rng.normal(0, scale * (hi - lo)), lo, hi))
        kw[f.name] = v
    return replace(params, **kw)
