"""Champion/challenger gate and rollback rule, as pure functions of numbers (unit-tested with synthetic values).

Promotion needs BOTH: a real gain on the objective (at least MIN_GAIN lower than the champion's) and no
meaningful extra friction on honest workers (block rate at most MAX_BLOCK_INCREASE percentage points higher).
Rollback: a champion promoted in generation g is re-measured in generation g+1 against the NEW attacker
generation next to the champion it replaced; if it is worse by more than ROLLBACK_MARGIN, the old one returns.
"""
from __future__ import annotations

MIN_GAIN = 0.5
MAX_BLOCK_INCREASE = 1.0   # percentage points of honest_block_rate
ROLLBACK_MARGIN = 0.5


def gate_decision(champion_obj: float, challenger_obj: float, champion_block: float, challenger_block: float,
                  *, min_gain: float = MIN_GAIN, max_block_increase: float = MAX_BLOCK_INCREASE) -> bool:
    """True = promote the challenger. All inputs are GATE-set values (lower objective is better)."""
    return challenger_obj <= champion_obj - min_gain and challenger_block <= champion_block + max_block_increase


def rollback_decision(prev_obj: float, champion_obj: float, *, margin: float = ROLLBACK_MARGIN) -> bool:
    """True = roll back to the previous champion. Both objectives are measured on the same new gate set."""
    return champion_obj > prev_obj + margin
