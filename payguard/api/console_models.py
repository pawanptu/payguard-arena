"""Pydantic models of the ops console files (docs/CONSOLE_CONTRACT.md, sections A and B).

They validate what the API serves: eval/console/queue.json and eval/console/explanations.json. Required fields and
types are checked; unknown extra fields are kept (the contract may grow), so the UI never loses data. The replay
contract in replay.py is separate and unchanged.
"""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("must be a finite number (no NaN/Infinity)")
    return v


Num = Annotated[float, AfterValidator(_finite)]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Layer(_Open):
    verdict: Literal["RELEASE", "STEP_UP", "HOLD"]
    fired: list[str]


class Outcome(_Open):
    paid: bool
    paid_at: Num | None = None
    delay_h: Num | None = None


class Truth(_Open):
    is_attack: bool
    family: str


class Case(_Open):
    event_id: int
    t: Num
    worker_id: int
    requester_id: int
    amount: Num
    verdict: Literal["STEP_UP", "HOLD"]
    risk: Num
    decided_by: Literal["swarm", "layer", "both"]
    scores: dict[str, Num]
    weights: dict[str, Num]
    vote_share: dict[str, Num]
    layer: Layer
    features: dict[str, Num]
    outcome: Outcome
    truth: Truth
    rank: int = Field(ge=1)


class Specialist(_Open):
    key: str
    label: str
    what: str


class Queue(_Open):
    schema_version: Literal[1]
    simulated: Literal[True]
    generated_at: str
    source: dict
    thresholds: dict[str, Num]
    counts: dict[str, Num]
    specialists: list[Specialist]
    cases: list[Case]


class Explanation(_Open):
    source: Literal["llm", "template"]
    summary: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1, max_length=3)
    check_next: str = Field(min_length=1)
    suggested_action: Literal["HOLD", "STEP_UP", "RELEASE"]


class Explanations(_Open):
    schema_version: Literal[1]
    provider: str
    model: str
    prompt_version: str
    generated_at: str
    grounding: str
    counts: dict[str, int]
    explanations: dict[str, Explanation]
