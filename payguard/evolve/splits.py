"""Evolution splits, both sliced from the protocol's TRAIN seeds (no new seeds are invented here).

  TRAIN_A  first 10 TRAIN seeds   fitness: the genetic algorithm selects defenders and attackers on these
  TRAIN_B  last 10 TRAIN seeds    gate: only the champion/challenger decision and the final checks see these
"""
from __future__ import annotations

from ..protocol import TRAIN_FAMILIES, TRAIN_SEEDS

TRAIN_A: tuple[int, ...] = tuple(TRAIN_SEEDS[:10])
TRAIN_B: tuple[int, ...] = tuple(TRAIN_SEEDS[10:])
FAMILIES: tuple[str, ...] = tuple(TRAIN_FAMILIES)
