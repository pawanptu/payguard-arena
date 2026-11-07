"""FastAPI app: a read-only replay API plus the prebuilt React app (web/dist) when it exists.

Run:  uvicorn payguard.api.app:app          (replay path: env PAYGUARD_REPLAY_PATH, else the sample fixture)
No secrets are needed; Replay mode works on any machine.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .replay import Replay, load_replay

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY = Path(__file__).resolve().parent / "fixtures" / "sample_replay.json"
DEFAULT_WEB_DIST = REPO_ROOT / "web" / "dist"
MAX_EVENTS_PER_PAGE = 1000


def create_app(replay_path: str | Path | None = None, *, web_dist: str | Path | None = None) -> FastAPI:
    """Build the app. Raises ReplayError at startup if the replay file is missing or breaks the contract.

    `web_dist` overrides where the built React app is looked for (default: <repo>/web/dist).
    """
    path = Path(replay_path or os.environ.get("PAYGUARD_REPLAY_PATH") or DEFAULT_REPLAY)
    replay: Replay = load_replay(path)
    dist = Path(web_dist) if web_dist is not None else DEFAULT_WEB_DIST

    app = FastAPI(title="PayGuard Arena API", version="0.1.0")
    app.state.replay = replay
    app.state.replay_path = str(path)

    # --- SEAM (director): live PayPal receipts. ---------------------------------------------------
    # Replace this callable with one backed by the ReceiptStore from the rail track, e.g.
    #   app.state.payouts_provider = lambda: [r.to_payout_dict() for r in store.list()]
    # It must return dicts shaped like replay.Payout (event_id, mode "sim"|"live", sender_batch_id,
    # payout_batch_id, batch_status, item_status, amount_usd, created_at). Until then: replay payouts.
    payouts_provider: Callable[[], list[dict]] = lambda: [p.model_dump() for p in replay.payouts]  # noqa: E731
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
        return {"source": "replay", "payouts": provider()}

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
