"""Pre-committed evaluation protocol. Seeds and splits are fixed BEFORE any swarm code exists.

  TRAIN         seeds 1000-1019, families ring + structuring   (evolution may tune on these)
  TEST_SEEN     seeds 2000-2019, families ring + structuring   (same families, unseen seeds)
  TEST_HELDOUT  seeds 3000-3019, family sleeper ONLY           (never used for tuning anything)
"""
from __future__ import annotations

import numpy as np

from .sim.attacks import sample_params
from .sim.engine import SimConfig, build_episode
from .sim.types import Episode

TRAIN_SEEDS = tuple(range(1000, 1020))
TEST_SEEN_SEEDS = tuple(range(2000, 2020))
TEST_HELDOUT_SEEDS = tuple(range(3000, 3020))
TRAIN_FAMILIES = ("ring", "structuring")
HELDOUT_FAMILIES = ("sleeper",)
CAMPAIGNS_PER_EPISODE = (2, 4)  # inclusive range


def episode_for(seed: int, families: tuple[str, ...], cfg: SimConfig = SimConfig()) -> Episode:
    """Episode with 2-4 campaigns, round-robin over `families` from a random start, params uniform in bounds."""
    rng = np.random.default_rng([seed, 99])
    n = int(rng.integers(CAMPAIGNS_PER_EPISODE[0], CAMPAIGNS_PER_EPISODE[1] + 1))
    offset = int(rng.integers(0, len(families)))
    campaigns = []
    for i in range(n):
        fam = families[(offset + i) % len(families)]
        campaigns.append((fam, sample_params(fam, rng)))
    return build_episode(seed, campaigns, cfg)


def episodes(seeds: tuple[int, ...], families: tuple[str, ...]) -> list[Episode]:
    return [episode_for(s, families) for s in seeds]
