"""Record a copilot run as a contract-C session file: `eval/copilot/sessions/<id>.json`."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .tooling import SOURCE_CONSOLE, SOURCE_REST, SOURCE_TOOLKIT

ROOT = Path(__file__).resolve().parents[2]
SESSIONS_DIR = ROOT / "eval" / "copilot" / "sessions"
SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
REDACTION_NOTE = ("Emails, names, addresses, phone numbers and payer ids are removed; "
                  "ids, statuses, amounts and dates are kept.")
SOURCES = {SOURCE_TOOLKIT, SOURCE_REST, SOURCE_CONSOLE}
PREVIEW_CHARS = 4 * 1024  # characters of a capped result kept as a preview
MAX_DATA_CHARS = 20000  # a recorded tool result larger than this is replaced by a truncated preview


class RecordError(ValueError):
    pass


def validate_slug(session_id: str) -> str:
    if not isinstance(session_id, str) or not SLUG.match(session_id):
        raise RecordError("session id must be a lowercase slug: letters, digits and hyphens (1-64 chars)")
    return session_id


def _cap(turn: dict) -> dict:
    if turn.get("role") == "tool_result" and "data" in turn:
        text = json.dumps(turn["data"], ensure_ascii=False, default=str)
        if len(text) > MAX_DATA_CHARS:
            turn = {**turn, "data": {"truncated": True, "original_chars": len(text), "preview": text[:PREVIEW_CHARS]}}
    return turn


def build_session(
    *,
    session_id: str,
    title: str,
    order: int,
    turns: list[dict],
    llm_provider: str,
    llm_model: str,
    toolkit_version: str,
    enabled_tools: list[str],
    recorded_at: datetime | None = None,
) -> dict:
    validate_slug(session_id)
    when = (recorded_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    session = {
        "schema_version": 1,
        "id": session_id,
        "order": int(order),
        "title": title.strip(),
        "recorded_at": when.isoformat(timespec="seconds"),
        "environment": "paypal-sandbox",
        "llm": {"provider": llm_provider, "model": llm_model},
        "toolkit": {"package": "paypal-agent-toolkit", "version": toolkit_version, "read_only": True,
                    "enabled_tools": list(enabled_tools)},
        "redaction": REDACTION_NOTE,
        "turns": [_cap(t) for t in turns],
    }
    validate_session(session)
    return session


def validate_session(session: dict) -> None:
    """Check the shape contract C promises to the site. Raises RecordError."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise RecordError(msg)

    need(session.get("schema_version") == 1, "schema_version must be 1")
    validate_slug(session.get("id"))
    need(isinstance(session.get("order"), int), "order must be an integer")
    need(bool(str(session.get("title", "")).strip()), "title is required")
    need(session.get("environment") == "paypal-sandbox", "environment must be paypal-sandbox")
    need(isinstance(session.get("llm"), dict) and {"provider", "model"} <= set(session["llm"]), "llm needs provider and model")
    tk = session.get("toolkit")
    need(isinstance(tk, dict) and tk.get("read_only") is True and isinstance(tk.get("enabled_tools"), list),
         "toolkit block malformed")
    turns = session.get("turns")
    need(isinstance(turns, list) and turns and turns[0].get("role") == "user", "turns must start with a user turn")
    need(turns[-1].get("role") == "assistant", "turns must end with an assistant turn")
    open_call: dict | None = None
    for t in turns:
        role = t.get("role")
        if role in ("user", "assistant"):
            need(isinstance(t.get("text"), str) and t["text"].strip() != "", f"{role} turn needs text")
        elif role == "tool_call":
            need(open_call is None, "tool_call without a result")
            need(t.get("source") in SOURCES, "tool_call source invalid")
            need(isinstance(t.get("args"), dict) and isinstance(t.get("step"), int), "tool_call needs step and args")
            open_call = t
        elif role == "tool_result":
            need(open_call is not None and t.get("step") == open_call["step"] and t.get("tool") == open_call["tool"],
                 "tool_result does not match its tool_call")
            need(isinstance(t.get("ok"), bool) and isinstance(t.get("ms"), int), "tool_result needs ok and ms")
            need(("data" in t) if t["ok"] else isinstance(t.get("error"), str), "tool_result needs data or error")
            open_call = None
        else:
            raise RecordError(f"unknown turn role {role!r}")
    need(open_call is None, "tool_call without a result")


def write_session(session: dict, *, sessions_dir: Path | None = None, force: bool = False) -> Path:
    """Write the session atomically; refuse to overwrite an existing id unless `force`."""
    validate_session(session)
    directory = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
    path = directory / f"{session['id']}.json"
    if path.exists() and not force:
        raise RecordError(f"{path.name} already exists (use --force to overwrite)")
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
