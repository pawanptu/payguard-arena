"""Round-3 evaluation protocol: sealed held-out set. Never use TEST_HELDOUT3 seeds for tuning anything.

  TEST_HELDOUT3   seeds 5000-5019, family "h3" ONLY   (never built before the defenders are frozen)

Episodes are built exactly like round 1 (`payguard.protocol.episode_for`): 2-4 campaigns, params uniform in bounds.
"""
from __future__ import annotations

from payguard import protocol as _round1
from payguard.sim.engine import SimConfig
from payguard.sim.types import Episode

from . import family as _family  # noqa: F401  (importing the package registers the family; see __init__)

TEST_HELDOUT3_SEEDS = tuple(range(5000, 5020))
HELDOUT3_FAMILIES = (_family.NAME,)


def episode_for(seed: int, families: tuple[str, ...] = HELDOUT3_FAMILIES, cfg: SimConfig = SimConfig()) -> Episode:
    return _round1.episode_for(seed, families, cfg)


def episodes(seeds: tuple[int, ...], families: tuple[str, ...] = HELDOUT3_FAMILIES) -> list[Episode]:
    return [episode_for(s, families) for s in seeds]
