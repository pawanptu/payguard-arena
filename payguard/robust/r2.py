"""The frozen swarm_r2 defender: `make_r2_defender()` builds it from `eval/r2/champion.json`.

The champion file is written once by `python -m scripts.r2_select` (selection rule: eval/r2/SELECTION_RULE.md) and
then frozen: it holds the swarm parameters and, if the selected mechanism has one, the calibrated layer cut-offs.
Nothing is read from episodes or from the network at construction time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..swarm.params import SwarmParams
from .layer import R2Defender, make_r2

CHAMPION_PATH = Path(__file__).resolve().parents[2] / "eval" / "r2" / "champion.json"


def load_champion(path: Path = CHAMPION_PATH) -> tuple[SwarmParams, dict | None, dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return SwarmParams(**doc["params"]), doc.get("layer"), doc


def champion_sha256(path: Path = CHAMPION_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_r2_defender(path: Path = CHAMPION_PATH) -> R2Defender:
    """A fresh swarm_r2 (own L1 weights and own layer state) for ONE episode."""
    params, layer, _ = load_champion(path)
    return make_r2(params, layer, name="swarm_r2")
