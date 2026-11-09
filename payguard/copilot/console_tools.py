"""Local console data as copilot tools (source "payguard-console").

Reads `eval/console/queue.json` (contract A) and `eval/console/explanations.json` (contract B) and the real
sandbox payouts listed in `eval/replay_live.json`. The copilot never sees ground truth: `truth` and `outcome` are
stripped from every case, and no truth-derived count is returned.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .tooling import SOURCE_CONSOLE, Tool, ToolError

ROOT = Path(__file__).resolve().parents[2]
NOT_BUILT = "console data not built"
HIDDEN_CASE_KEYS = ("truth", "outcome")
# counts{} keys that are not derived from per-case ground truth (cases_attack / cases_honest are).
SAFE_COUNT_KEYS = ("events", "RELEASE", "STEP_UP", "HOLD", "cases")
# The contract's whole-episode totals (counts.attack_usd_* cover the whole episode, not only the cases).
EPISODE_TOTAL_KEYS = ("attack_usd_total", "attack_usd_stopped", "attack_usd_leaked")
MAX_LIST = 50


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        raise ToolError(NOT_BUILT) from None
    except (ValueError, OSError):
        raise ToolError(f"{path.name} is unreadable") from None


def _shift(stamp: str, days: int) -> str:
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return stamp
    return (dt + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strip(case: dict) -> dict:
    return {k: v for k, v in case.items() if k not in HIDDEN_CASE_KEYS}


def _int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


class ConsoleTools:
    def __init__(self, queue_path: Path | None = None, explanations_path: Path | None = None,
                 replay_live_path: Path | None = None):
        self.queue_path = Path(queue_path) if queue_path else ROOT / "eval" / "console" / "queue.json"
        self.explanations_path = (Path(explanations_path) if explanations_path
                                  else ROOT / "eval" / "console" / "explanations.json")
        self.replay_live_path = Path(replay_live_path) if replay_live_path else ROOT / "eval" / "replay_live.json"

    # -- implementations
    def list_hold_cases(self, args: dict) -> dict:
        queue = _load(self.queue_path)
        cases = queue.get("cases", [])
        verdict = args.get("verdict")
        if verdict not in (None, ""):
            v = str(verdict).upper()
            if v not in ("HOLD", "STEP_UP"):
                raise ToolError("verdict must be HOLD or STEP_UP")
            cases = [c for c in cases if c.get("verdict") == v]
        limit = _int(args.get("limit"), 10, 1, MAX_LIST)
        rows = []
        for rank, c in enumerate(cases[:limit], start=1):
            share = c.get("vote_share") or {}
            top = max(share, key=lambda k: share[k]) if share and max(share.values()) > 0 else None
            rows.append({"rank": rank, "event_id": c.get("event_id"), "verdict": c.get("verdict"),
                         "risk": c.get("risk"), "amount": c.get("amount"), "decided_by": c.get("decided_by"),
                         "top_specialist": top})
        return {"total_matching": len(cases), "returned": len(rows), "cases": rows}

    def get_hold_case(self, args: dict) -> dict:
        try:
            event_id = int(args.get("event_id"))
        except (TypeError, ValueError):
            raise ToolError("event_id must be an integer") from None
        queue = _load(self.queue_path)
        case = next((c for c in queue.get("cases", []) if c.get("event_id") == event_id), None)
        if case is None:
            raise ToolError(f"no hold or step-up case with event_id {event_id} (released payouts are not in the queue)")
        out = _strip(case)
        try:
            expl = _load(self.explanations_path).get("explanations", {}).get(str(event_id))
        except ToolError:
            expl = None
        if expl:
            out["explanation"] = expl
        out["thresholds"] = queue.get("thresholds")
        return out

    def queue_stats(self, args: dict) -> dict:
        queue = _load(self.queue_path)
        counts = queue.get("counts", {})
        return {
            "simulated": True,
            "source": queue.get("source"),
            "thresholds": queue.get("thresholds"),
            "counts": {k: counts[k] for k in SAFE_COUNT_KEYS if k in counts},
            "episode_totals_usd": {k: counts[k] for k in EPISODE_TOTAL_KEYS if k in counts},
            "specialists": [s.get("key") for s in queue.get("specialists", [])],
        }

    def list_live_payouts(self, args: dict) -> dict:
        replay = _load(self.replay_live_path)
        rows = [{"event_id": p.get("event_id"), "payout_batch_id": p.get("payout_batch_id"),
                 "batch_status": p.get("batch_status"), "amount": p.get("amount_usd"),
                 "currency": "USD", "created_at": p.get("created_at")}
                for p in replay.get("payouts", [])]
        stamps = sorted(r["created_at"] for r in rows if isinstance(r.get("created_at"), str))
        out: dict[str, Any] = {"note": "Real PayPal sandbox payouts sent earlier by this project.", "payouts": rows}
        if stamps:
            # Handy for reconciliation: pass these to list_transactions as start_date / end_date.
            out["suggested_transaction_window"] = {"start_date": _shift(stamps[0], -1), "end_date": _shift(stamps[-1], 1)}
        return out

    # -- tool objects
    def tools(self) -> list[Tool]:
        none = {"type": "object", "properties": {}}
        return [
            Tool("list_hold_cases", SOURCE_CONSOLE,
                 "Local console data. Held or stepped-up payouts of the replay episode in review-queue rank order "
                 "(compact rows: event_id, verdict, risk, amount, decided_by, top_specialist).",
                 {"type": "object", "properties": {
                     "verdict": {"type": "string", "enum": ["HOLD", "STEP_UP"], "description": "optional filter"},
                     "limit": {"type": "integer", "description": "rows to return, default 10, max 50"}}},
                 self.list_hold_cases),
            Tool("get_hold_case", SOURCE_CONSOLE,
                 "Local console data. One case with its decision-time evidence: per-specialist scores, weights, "
                 "vote shares, anomaly-layer result, features, and the AI explanation when built. Never ground truth.",
                 {"type": "object", "properties": {"event_id": {"type": "integer"}}, "required": ["event_id"]},
                 self.get_hold_case),
            Tool("queue_stats", SOURCE_CONSOLE,
                 "Local console data. Verdict counts of the replay episode and the HOLD / STEP_UP thresholds.",
                 none, self.queue_stats),
            Tool("list_live_payouts", SOURCE_CONSOLE,
                 "Local console data. The real PayPal sandbox payouts this project sent earlier: event_id, "
                 "payout_batch_id, batch_status, amount, created_at, plus a suggested date window for list_transactions. Use payout_batch_id with get_payout_batch.",
                 none, self.list_live_payouts),
        ]
