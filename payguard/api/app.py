"""FastAPI app: a read-only replay API plus the prebuilt React app (web/dist) when it exists.

Run:  uvicorn payguard.api.app:app          (replay path: env PAYGUARD_REPLAY_PATH, else the sample fixture)
No secrets are needed; Replay mode works on any machine.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..rail import webhook
from ..rail.store import ReceiptStore
from .replay import Replay, load_replay

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY = Path(__file__).resolve().parent / "fixtures" / "sample_replay.json"
DEFAULT_WEB_DIST = REPO_ROOT / "web" / "dist"
MAX_EVENTS_PER_PAGE = 1000
MAX_RUNTIME_RECEIPTS = 50


def create_app(replay_path: str | Path | None = None, *, web_dist: str | Path | None = None,
               receipt_store: ReceiptStore | None = None) -> FastAPI:
    """Build the app. Raises ReplayError at startup if the replay file is missing or breaks the contract.

    `web_dist` overrides where the built React app is looked for (default: <repo>/web/dist).
    `receipt_store` is shared with the payout rails and the webhook (default: a fresh in-memory store).
    """
    path = Path(replay_path or os.environ.get("PAYGUARD_REPLAY_PATH") or DEFAULT_REPLAY)
    replay: Replay = load_replay(path)
    dist = Path(web_dist) if web_dist is not None else DEFAULT_WEB_DIST

    app = FastAPI(title="PayGuard Arena API", version="0.1.0")
    app.state.replay = replay
    app.state.replay_path = str(path)

    # --- Payout receipts: one ReceiptStore shared by the rails and the verified PayPal webhook. -----
    # Receipts written at runtime (SimRail / LiveRail, then webhook status updates) come first,
    # newest first; the replay file's recorded payouts follow. Rows carry their own mode (sim|live).
    store = receipt_store if receipt_store is not None else ReceiptStore()
    app.state.receipt_store = store
    app.include_router(webhook.router)
    app.dependency_overrides[webhook.get_receipt_store] = lambda: store

    def payouts_provider() -> list[dict]:
        return [asdict(r) for r in store.recent(MAX_RUNTIME_RECEIPTS)] + [p.model_dump() for p in replay.payouts]

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
        return {"source": "runtime+replay" if len(store) else "replay", "payouts": provider()}

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


app = create_app()
