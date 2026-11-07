"""FastAPI app: a read-only replay API plus the prebuilt React app (web/dist) when it exists.

Run:  uvicorn payguard.api.app:app          (replay: env PAYGUARD_REPLAY_PATH, else eval/replay_live.json, else eval/replay.json, else the sample fixture)
No secrets are needed; Replay mode works on any machine.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..rail import webhook
from ..rail.base import PayoutReceipt, parse_iso
from ..rail.events import WebhookEventLog
from ..rail.store import ReceiptStore
from .replay import Replay, load_replay

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY = Path(__file__).resolve().parent / "fixtures" / "sample_replay.json"
DEFAULT_WEB_DIST = REPO_ROOT / "web" / "dist"
MAX_EVENTS_PER_PAGE = 1000
MAX_RUNTIME_RECEIPTS = 50
DEFAULT_WEBHOOKS_LIMIT = 50
MAX_WEBHOOKS_LIMIT = 200
# Without an explicit path or PAYGUARD_REPLAY_PATH the most complete committed artefact wins: the recorded
# live run, then the evolution replay, then the synthetic sample fixture (a checkout without eval outputs).
REPLAY_CANDIDATES = (REPO_ROOT / "eval" / "replay_live.json", REPO_ROOT / "eval" / "replay.json", DEFAULT_REPLAY)


def default_replay_path() -> Path:
    env = os.environ.get("PAYGUARD_REPLAY_PATH", "").strip()
    if env:
        return Path(env)
    return next((p for p in REPLAY_CANDIDATES if p.is_file()), DEFAULT_REPLAY)


def create_app(replay_path: str | Path | None = None, *, web_dist: str | Path | None = None,
               receipt_store: ReceiptStore | None = None) -> FastAPI:
    """Build the app. Raises ReplayError at startup if the replay file is missing or breaks the contract.

    `web_dist` overrides where the built React app is looked for (default: <repo>/web/dist).
    `receipt_store` is shared with the payout rails and the webhook (default: a fresh in-memory store).
    """
    path = Path(replay_path) if replay_path else default_replay_path()
    replay: Replay = load_replay(path)
    dist = Path(web_dist) if web_dist is not None else DEFAULT_WEB_DIST

    app = FastAPI(title="PayGuard Arena API", version="0.1.0")
    app.state.replay = replay
    app.state.replay_path = str(path)

    # --- Payout receipts: one ReceiptStore shared by the rails and the verified PayPal webhook. -----
    # The store is seeded with the replay's recorded LIVE payouts, so a webhook for a batch that a local
    # run sent (and recorded) finds its receipt even after this process restarted. Receipts written at
    # runtime (SimRail / LiveRail, then webhook status updates) come first, most recently updated first;
    # the replay file's remaining payouts follow. Rows carry their own mode (sim|live).
    store = receipt_store if receipt_store is not None else ReceiptStore()
    event_log = WebhookEventLog()
    app.state.receipt_store = store
    app.state.event_log = event_log
    seeded: dict[str, PayoutReceipt] = {}
    for p in replay.payouts:
        if p.mode == "live" and store.get(p.event_id) is None:  # never clobber a receipt already held
            rec = store.upsert(PayoutReceipt(
                event_id=p.event_id, mode=p.mode, sender_batch_id=p.sender_batch_id,
                payout_batch_id=p.payout_batch_id, batch_status=p.batch_status, item_status=p.item_status,
                amount_usd=p.amount_usd, created_at=p.created_at, updated_at=p.created_at))
            seeded[rec.sender_batch_id] = rec
    app.include_router(webhook.router)
    app.dependency_overrides[webhook.get_receipt_store] = lambda: store
    app.dependency_overrides[webhook.get_event_log] = lambda: event_log

    def payouts_provider() -> list[dict]:
        rows = [asdict(r) for r in store.recent(MAX_RUNTIME_RECEIPTS)]
        shown = {r["sender_batch_id"] for r in rows}  # one row per sender_batch_id: the store's wins
        return rows + [p.model_dump() for p in replay.payouts if p.sender_batch_id not in shown]

    def has_runtime_rows() -> bool:
        """True when the store holds anything beyond the untouched rows seeded from the replay."""
        return any(seeded.get(r.sender_batch_id) != r for r in store.recent(len(store)))

    app.state.payouts_provider = payouts_provider
    # -----------------------------------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "mode": "replay", "simulated": True,
                "synthetic_fixture": replay.synthetic_fixture, "schema_version": replay.schema_version}

    @app.get("/api/replay/summary")
    def summary() -> dict:
        data = replay.model_dump(exclude={"episode"})
        ep = replay.episode
        data["episode"] = {"seed": ep.seed, "split": ep.split, "defender": ep.defender, "n_events": len(ep.events)}
        return data

    @app.get("/api/replay/events")
    def events(offset: int = Query(0, ge=0), limit: int = Query(200, ge=1)) -> dict:
        """Paged episode events. `limit` is capped at 1000 (larger values are clamped, not rejected)."""
        limit = min(limit, MAX_EVENTS_PER_PAGE)
        evs = replay.episode.events
        return {"total": len(evs), "offset": offset, "limit": limit,
                "events": [e.model_dump() for e in evs[offset:offset + limit]]}

    @app.get("/api/payouts")
    def payouts() -> dict:
        provider = app.state.payouts_provider
        return {"source": "runtime+replay" if has_runtime_rows() else "replay", "payouts": provider()}

    @app.get("/api/webhooks/recent")
    def webhooks_recent(
        limit: int = DEFAULT_WEBHOOKS_LIMIT,
        webhook_id: str | None = Depends(webhook.get_webhook_id),
    ) -> dict:
        """Verified PayPal webhook events, newest first: this process's log merged with the replay's
        recorded ones (runtime wins on the same event_id). `limit` is clamped to 1..200."""
        limit = max(1, min(limit, MAX_WEBHOOKS_LIMIT))
        merged: dict[str, dict] = {}
        for e in event_log.recent(MAX_WEBHOOKS_LIMIT):
            merged[e["event_id"]] = {**e, "source": "runtime"}
        for w in replay.verified_webhooks:
            merged.setdefault(w.event_id, {**w.model_dump(), "outcome": None, "source": "replay"})
        ordered = sorted(merged.values(), key=lambda e: _when(e["received_at"]), reverse=True)
        return {"configured": bool(webhook_id), "events": ordered[:limit]}

    @app.get("/api/{rest:path}", include_in_schema=False)
    def api_not_found(rest: str):
        raise HTTPException(status_code=404, detail=f"no such endpoint: /api/{rest}")

    index = dist / "index.html"
    if index.is_file():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")
        root = dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            # Real files in dist (favicon etc.) are served as is; every other path gets the SPA shell.
            candidate = (root / path).resolve()
            if path and candidate.is_file() and root in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)

    return app


def _when(iso: str) -> datetime:
    """Sort key for received_at; an unparsable value sorts oldest."""
    try:
        dt = parse_iso(iso)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


app = create_app()
