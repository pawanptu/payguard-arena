"""Review queue of the ops console (docs/CONSOLE_CONTRACT.md, section A).

`record_episode` plays ONE episode with a defender wrapped in `CaseRecorder`, which notes at decision time what a
reviewer may see: the swarm scores and weights, the aggregated risk, the anomaly layer's verdict and the statistics it
fired on, and the platform features of the event (`PlatformLog.features` is read-only). `assemble_queue` checks that
every verdict equals the one in the published replay (`ParityError` otherwise: nothing is written) and turns the
HOLD / STEP_UP events into the queue, sorted with `payguard.console.priority.priority_key`.

Nothing here reads ground truth to decide anything. `truth` and `outcome` are attached to each case only for the UI's
"reveal" step, after the sort key has been computed from decision-time fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from ..sim.features import PlatformLog
from ..sim.runner import RunResult, run_episode
from ..sim.types import Episode, Event, Verdict
from ..swarm.coordinator import NAMES, verdict_for
from .priority import priority_key

SCHEMA_VERSION = 1
SEVERITY = {"RELEASE": 0, "STEP_UP": 1, "HOLD": 2}
SPECIALISTS = [
    {"key": "velocity", "label": "Velocity", "what": "Payout count and USD volume in the account's last 24 h"},
    {"key": "graph", "label": "Graph",
     "what": "Collusion structure: accounts tied to the same requester, device or destination"},
    {"key": "drift", "label": "Drift", "what": "Payout rate against the account's own normal rate"},
    {"key": "policy", "label": "Policy", "what": "Amount against the limit for the account's age"},
    {"key": "layer", "label": "Anomaly layer", "what": "Nine statistics with cut-offs set on honest traffic only"},
]
FEATURE_KEYS = ("age_days", "n_prior", "window_n", "window_usd", "amount_ratio", "device_workers",
                "destination_workers", "requester_workers", "worker_requesters")


class ParityError(RuntimeError):
    """The re-played episode disagrees with the published replay; the queue must not be written."""


def r3(x: float) -> float:
    return round(float(x), 3)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------- pure helpers -----------------------------
def vote_share(scores: dict[str, float], weights: dict[str, float]) -> dict[str, float]:
    """Each specialist's share of the vote: w_i * s_i / sum_j w_j * s_j (all zeros when nobody speaks up)."""
    total = sum(weights[n] * scores[n] for n in scores)
    if total <= 1e-12:
        return {n: 0.0 for n in scores}
    return {n: r3(weights[n] * scores[n] / total) for n in scores}


def decided_by(swarm_verdict: str, layer_verdict: str, final_verdict: str) -> str:
    """Which part set the final severity: 'swarm', 'layer' (it raised it) or 'both' (both reach it)."""
    sw, ly, fin = SEVERITY[swarm_verdict], SEVERITY[layer_verdict], SEVERITY[final_verdict]
    if ly > sw:
        return "layer"
    if sw == fin and ly == fin:
        return "both"
    return "swarm"


# ---------------------------------------------------------------- recording ------------------------------
class CaseRecorder:
    """Wraps a swarm_r2-like defender (with `last_signals`, `params`) and records the decision-time evidence."""

    def __init__(self, defender: Any):
        self.defender, self.name = defender, defender.name
        self.at: dict[int, dict] = {}

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        v = self.defender.decide(event, log)
        sig = self.defender.last_signals
        layer = sig.get("layer") or {"verdict": "RELEASE", "fired": []}
        risk = float(sig["risk"])
        self.at[event.id] = {
            "scores": dict(sig["scores"]), "weights": dict(sig["weights"]), "risk": risk,
            "swarm_verdict": verdict_for(risk, self.defender.params).value,
            "layer": {"verdict": layer["verdict"], "fired": list(layer["fired"])},
            "features": log.features(event),  # read-only; called before the runner adds the event
        }
        return v

    def learn(self, event: Event, is_attack: bool) -> None:
        self.defender.learn(event, is_attack)


def record_episode(episode: Episode, defender: Any) -> tuple[RunResult, dict[int, dict]]:
    rec = CaseRecorder(defender)
    return run_episode(episode, rec), rec.at


# ---------------------------------------------------------------- parity ---------------------------------
def check_parity(verdicts: dict[int, Verdict], replay_events: list[dict]) -> int:
    """Number of events checked. Raises ParityError naming the first differences."""
    theirs = {int(e["id"]): e["verdict"] for e in replay_events}
    mine = {i: v.value for i, v in verdicts.items()}
    problems: list[str] = []
    if set(theirs) != set(mine):
        only_a, only_b = sorted(set(mine) - set(theirs)), sorted(set(theirs) - set(mine))
        problems.append(f"event ids differ: {len(only_a)} only re-played (e.g. {only_a[:3]}), "
                        f"{len(only_b)} only in the replay (e.g. {only_b[:3]})")
    bad = sorted(i for i in set(theirs) & set(mine) if theirs[i] != mine[i])
    if bad:
        shown = ", ".join(f"#{i}: re-played {mine[i]} vs replay {theirs[i]}" for i in bad[:5])
        problems.append(f"{len(bad)} verdicts differ ({shown})")
    if problems:
        raise ParityError("parity with the published replay failed, nothing written: " + "; ".join(problems))
    return len(mine)


# ---------------------------------------------------------------- assembly -------------------------------
def build_case(event: Event, rec: dict, verdict: str, truth: Any, outcome: Any) -> dict:
    scores = {n: r3(rec["scores"][n]) for n in NAMES}
    weights = {n: r3(rec["weights"][n]) for n in NAMES}
    paid = bool(outcome.paid)
    return {
        "event_id": event.id, "t": r3(event.t), "worker_id": event.worker_id, "requester_id": event.requester_id,
        "amount": round(event.amount, 2), "verdict": verdict, "risk": r3(rec["risk"]),
        "decided_by": decided_by(rec["swarm_verdict"], rec["layer"]["verdict"], verdict),
        "scores": scores, "weights": weights,
        # shares come from the unrounded values so that they are not distorted by the rounding above
        "vote_share": vote_share(rec["scores"], rec["weights"]),
        "layer": {"verdict": rec["layer"]["verdict"], "fired": list(rec["layer"]["fired"])},
        "features": {k: r3(rec["features"][k]) if k in ("age_days", "window_usd", "amount_ratio")
                     else int(rec["features"][k]) for k in FEATURE_KEYS},
        "outcome": {"paid": paid, "paid_at": r3(outcome.paid_at) if paid and outcome.paid_at is not None else None,
                    "delay_h": r3(outcome.paid_at - event.t) if paid and outcome.paid_at is not None else None},
        "truth": {"is_attack": bool(truth.is_attack), "family": truth.family},
    }


def assemble_queue(episode: Episode, result: RunResult, at: dict[int, dict], replay: dict, *,
                   thresholds: dict[str, float], source: dict,
                   generated_at: str | None = None,
                   sort_key: Callable[[dict], tuple] = priority_key) -> dict:
    """The queue document. Verifies parity with `replay` first (raises ParityError); writes nothing itself."""
    checked = check_parity(result.verdicts, replay["episode"]["events"])
    counts = {"events": len(episode.events), "RELEASE": 0, "STEP_UP": 0, "HOLD": 0}
    attack_total = attack_paid = 0.0
    cases = []
    for e in episode.events:
        v = result.verdicts[e.id].value
        counts[v] += 1
        tr, oc = episode.truth[e.id], result.outcomes[e.id]
        if tr.is_attack:
            attack_total += e.amount
            if oc.paid:
                attack_paid += e.amount
        if v != "RELEASE":
            cases.append(build_case(e, at[e.id], v, tr, oc))
    cases.sort(key=sort_key)
    for i, c in enumerate(cases, 1):
        c["rank"] = i
    n_attack = sum(1 for c in cases if c["truth"]["is_attack"])
    counts.update({"cases": len(cases), "cases_attack": n_attack, "cases_honest": len(cases) - n_attack,
                   "attack_usd_total": round(attack_total, 2),
                   "attack_usd_stopped": round(attack_total - attack_paid, 2),
                   "attack_usd_leaked": round(attack_paid, 2)})
    return {
        "schema_version": SCHEMA_VERSION, "simulated": True, "generated_at": generated_at or now_iso(),
        "source": {**source, "parity_checked_events": checked},
        "thresholds": {k: r3(v) for k, v in thresholds.items()},
        "counts": counts, "specialists": SPECIALISTS, "cases": cases,
    }
