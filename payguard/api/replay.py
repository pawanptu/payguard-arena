"""The replay contract (schema_version 1) as Pydantic models, plus a loader with field-naming errors.

One JSON file holds everything the dashboard shows. All numbers are a SIMULATED replay on synthetic
data. The evolution engine will emit this same file later; the UI never hard-codes results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ValidationError

SCHEMA_VERSION = 1
OBJECTIVE_FORMULA = "leakage_pct + 0.5*honest_block_rate + 1.0*honest_unpaid_rate"


def _finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("must be a finite number (no NaN/Infinity)")
    return v


Num = Annotated[float, AfterValidator(_finite)]
Split = Literal["TRAIN", "TEST_SEEN", "TEST_HELDOUT"]
Verdict = Literal["RELEASE", "STEP_UP", "HOLD"]


class ResultRow(BaseModel):
    defender: str
    split: Split
    leakage_pct: Num
    honest_block_rate: Num
    honest_unpaid_rate: Num
    honest_median_ttp_h: Num
    immunity_attempts: Num
    never_contained_rate: Num
    objective: Num


class EvolutionRow(BaseModel):
    generation: int
    champion_objective: Num
    challenger_objective: Num
    promoted: bool
    attacker_family: str | None = None
    attacker_leakage_pct: Num | None = None
    weights: dict[str, Num]


class EpisodeEvent(BaseModel):
    id: int
    t: Num
    worker_id: int
    requester_id: int
    amount: Num
    verdict: Verdict
    paid: bool
    paid_at: Num | None = None
    is_attack: bool
    family: str
    signals: dict[str, Num] | None = None
    weights: dict[str, Num] | None = None


class Episode(BaseModel):
    seed: int
    split: str
    defender: str
    events: list[EpisodeEvent]


class Payout(BaseModel):
    event_id: int
    mode: Literal["sim", "live"]
    sender_batch_id: str
    payout_batch_id: str
    batch_status: str
    item_status: str
    amount_usd: Num
    created_at: str


class Replay(BaseModel):
    schema_version: Literal[1]
    simulated: Literal[True]
    synthetic_fixture: bool
    generated_at: str
    objective: str
    results: list[ResultRow]
    evolution: list[EvolutionRow]
    episode: Episode
    payouts: list[Payout]


class ReplayError(ValueError):
    """The replay file is missing, not JSON, or breaks the contract. The message names the bad field(s)."""


def _loc(loc: tuple) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "<root>"


def parse_replay(data: object, source: str = "replay") -> Replay:
    try:
        return Replay.model_validate(data)
    except ValidationError as exc:
        shown = [f"{_loc(e['loc'])}: {e['msg']}" for e in exc.errors()[:10]]
        more = exc.error_count() - len(shown)
        if more > 0:
            shown.append(f"... and {more} more")
        raise ReplayError(f"{source} breaks the replay contract (schema_version {SCHEMA_VERSION}): "
                          + "; ".join(shown)) from None


def load_replay(path: str | Path) -> Replay:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReplayError(f"cannot read replay file {p}: {exc}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReplayError(f"{p} is not valid JSON: {exc}") from None
    return parse_replay(data, str(p))
