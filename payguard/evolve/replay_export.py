"""Replay export (contract: schema_version 1, shared with the dashboard). TRAIN data only, all SIMULATED.

Top-level keys, exactly: schema_version, simulated, synthetic_fixture, generated_at, objective, results,
evolution, episode, payouts.

  results[]    one row per defender on the TRAIN split (split "TRAIN"): always_release, always_hold, static_rules,
               the best single specialist (chosen on TRAIN), swarm_frozen, swarm_l1 (default params), and the
               evolved champion as "swarm_evolved". The director adds TEST rows later.
  evolution[]  one entry per generation; objectives are GATE-set objectives; attacker_* is the strongest attacker
               (most USD leaked) against the champion of that generation; weights = the champion's L1 weights at
               the end of its gate run.
  episode      ONE TRAIN episode (TRAIN_SEEDS[0]) played by the champion, event by event with the coordinator's
               signals and weights at decision time.
  payouts      empty: the payout rail fills it at runtime.
Floats are rounded to 3 decimals. If the file would exceed MAX_BYTES, every attack event is kept and only a
deterministic 1-in-N sample of honest events (N reported in episode["honest_sample_every"], present only then).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from ..sim.metrics import Metrics
from ..sim.runner import run_episode
from ..swarm.coordinator import make_swarm
from ..swarm.objective import objective
from ..swarm.params import SwarmParams, default_params
from ..swarm.single import SPECIALIST_NAMES
from .evaluation import Evaluator, proto_spec, swarm_job

SCHEMA_VERSION = 1
MAX_BYTES = 5 * 1024 * 1024
TOP_LEVEL_KEYS = ("schema_version", "simulated", "synthetic_fixture", "generated_at", "objective", "results",
                  "evolution", "episode", "payouts")
OBJECTIVE_TEXT = "leakage_pct + 0.5*honest_block_rate + 1.0*honest_unpaid_rate (lower is better)"
RESULT_KEYS = ("defender", "split", "leakage_pct", "honest_block_rate", "honest_unpaid_rate", "honest_median_ttp_h",
               "immunity_attempts", "never_contained_rate", "objective")
EVOLUTION_KEYS = ("generation", "champion_objective", "challenger_objective", "promoted", "attacker_family",
                  "attacker_leakage_pct", "weights")
EVENT_KEYS = ("id", "t", "worker_id", "requester_id", "amount", "verdict", "paid", "paid_at", "is_attack", "family",
              "signals", "weights")


def clean(x):
    """Round floats to 3 decimals; NaN/inf become None (JSON has no NaN)."""
    if isinstance(x, float):
        return round(x, 3) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def result_row(defender: str, m: Metrics, split: str = "TRAIN") -> dict:
    return {"defender": defender, "split": split, "leakage_pct": m.leakage_pct,
            "honest_block_rate": m.honest_block_rate, "honest_unpaid_rate": m.honest_unpaid_rate,
            "honest_median_ttp_h": m.honest_median_ttp_h, "immunity_attempts": m.immunity_attempts,
            "never_contained_rate": m.never_contained_rate, "objective": objective(m)}


def defender_jobs(champion: SwarmParams, specs: list[tuple]) -> list[tuple[str, tuple]]:
    d = default_params()
    return [("always_release", ("always_release", tuple(specs))), ("always_hold", ("always_hold", tuple(specs))),
            ("static_rules", ("static", tuple(specs))),
            *[(f"single_{n}", ("single", n, asdict(d), tuple(specs))) for n in SPECIALIST_NAMES],
            ("swarm_frozen", swarm_job(d, specs, False)), ("swarm_l1", swarm_job(d, specs, True)),
            ("swarm_evolved", swarm_job(champion, specs, True))]


def train_metrics(ev: Evaluator, champion: SwarmParams, seeds: tuple[int, ...]) -> dict[str, Metrics]:
    """Every defender (all four singles included) on the protocol TRAIN episodes of `seeds`."""
    jobs = defender_jobs(champion, [proto_spec(s) for s in seeds])
    res = ev.run([j for _, j in jobs])
    return {name: m for (name, _), (m, _w) in zip(jobs, res)}


def best_single(metrics: dict[str, Metrics]) -> str:
    return min((n for n in metrics if n.startswith("single_")), key=lambda n: (objective(metrics[n]), n))


def results_rows(metrics: dict[str, Metrics]) -> list[dict]:
    bs = best_single(metrics)
    names = ["always_release", "always_hold", "static_rules", bs, "swarm_frozen", "swarm_l1", "swarm_evolved"]
    return [result_row(n, metrics[n]) for n in names]


def evolution_rows(lineage: list[dict]) -> list[dict]:
    out = []
    for e in lineage:
        fam = e["strongest_attacker_family"]
        out.append({"generation": e["generation"], "champion_objective": e["champion_gate"]["objective"],
                    "challenger_objective": e["challenger"]["objective_b"] if e["challenger"] else None,
                    "promoted": bool(e["promoted"]), "attacker_family": fam,
                    "attacker_leakage_pct": e["strongest_attacker"][fam]["leakage_pct"],
                    "weights": dict(e["champion_weights"])})
    return out


class _Recorder:
    """Defender wrapper that stores the swarm's signals and weights at decision time."""

    def __init__(self, swarm):
        self.swarm, self.name, self.signals = swarm, swarm.name, {}

    def decide(self, event, log):
        v = self.swarm.decide(event, log)
        s = self.swarm.last_signals
        self.signals[event.id] = ({"scores": dict(s["scores"]), "risk": s["risk"], "top": s["top"],
                                   "active": list(s["active"])}, dict(s["weights"]))
        return v

    def learn(self, event, is_attack):
        self.swarm.learn(event, is_attack)


def episode_events(params: SwarmParams, seed: int) -> list[dict]:
    ep = episode_for(seed, TRAIN_FAMILIES)
    rec = _Recorder(make_swarm(params, True))
    res = run_episode(ep, rec)
    events = []
    for e in ep.events:
        oc, tr = res.outcomes[e.id], ep.truth[e.id]
        sig, w = rec.signals[e.id]
        events.append({"id": e.id, "t": e.t, "worker_id": e.worker_id, "requester_id": e.requester_id,
                       "amount": e.amount, "verdict": res.verdicts[e.id].value, "paid": bool(oc.paid),
                       "paid_at": oc.paid_at, "is_attack": bool(tr.is_attack), "family": tr.family,
                       "signals": sig, "weights": w})
    return events


def build_replay(rows: list[dict], lineage: list[dict], champion: SwarmParams, *, seed: int | None = None,
                 max_bytes: int = MAX_BYTES, generated_at: str | None = None) -> dict:
    seed = TRAIN_SEEDS[0] if seed is None else seed
    events = episode_events(champion, seed)
    doc = {"schema_version": SCHEMA_VERSION, "simulated": True, "synthetic_fixture": False,
           "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "objective": OBJECTIVE_TEXT, "results": rows, "evolution": evolution_rows(lineage),
           "episode": {"seed": seed, "split": "TRAIN", "defender": "swarm_evolved", "events": events},
           "payouts": []}
    doc = clean(doc)
    size = len(json.dumps(doc, separators=(",", ":")))
    if size > max_bytes:
        evs = doc["episode"]["events"]
        n = 2
        while True:
            honest_i = 0
            kept = []
            for e in evs:
                if e["is_attack"]:
                    kept.append(e)
                else:
                    if honest_i % n == 0:
                        kept.append(e)
                    honest_i += 1
            doc["episode"]["events"], doc["episode"]["honest_sample_every"] = kept, n  # only added when sampling
            if len(json.dumps(doc, separators=(",", ":"))) <= max_bytes or n > len(evs):
                break
            n += 1
    return doc


def write_replay(path: Path, doc: dict) -> int:
    """Write compactly; returns the file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def validate_replay(doc: dict) -> None:
    """Raises AssertionError when the document breaks the schema_version 1 contract."""
    assert set(doc) == set(TOP_LEVEL_KEYS), set(doc) ^ set(TOP_LEVEL_KEYS)
    assert doc["schema_version"] == 1 and doc["simulated"] is True and doc["synthetic_fixture"] is False
    assert isinstance(doc["generated_at"], str) and isinstance(doc["objective"], str)
    assert doc["payouts"] == []
    for row in doc["results"]:
        assert set(row) == set(RESULT_KEYS) and row["split"] == "TRAIN", row
    for row in doc["evolution"]:
        assert set(row) == set(EVOLUTION_KEYS), row
        assert isinstance(row["promoted"], bool) and isinstance(row["weights"], dict)
    ep = doc["episode"]
    assert ep["split"] == "TRAIN" and isinstance(ep["seed"], int) and ep["defender"] == "swarm_evolved"
    assert ep["events"], "no events"
    for e in ep["events"]:
        assert set(e) == set(EVENT_KEYS), e
        assert isinstance(e["id"], int) and isinstance(e["paid"], bool) and isinstance(e["is_attack"], bool)
        assert e["verdict"] in {"RELEASE", "HOLD", "STEP_UP"}
        assert set(e["signals"]) == {"scores", "risk", "top", "active"} and isinstance(e["weights"], dict)
