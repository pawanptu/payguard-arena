"""The ops console endpoints of payguard/api/app.py. Files live in tmp_path; the real app is untouched."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from payguard.api import console_models as models
from payguard.api.app import DEFAULT_REPLAY, create_app
from payguard.console import rounds as R

ROOT = Path(__file__).resolve().parents[1]


def make_case(i, rank):
    return {
        "event_id": i, "t": 1.5, "worker_id": 3, "requester_id": 4, "amount": 50.0,
        "verdict": "HOLD", "risk": 0.7, "decided_by": "both",
        "scores": {"velocity": 0.0, "graph": 0.7, "drift": 0.0, "policy": 0.0},
        "weights": {"velocity": 0.25, "graph": 0.25, "drift": 0.25, "policy": 0.25},
        "vote_share": {"velocity": 0.0, "graph": 1.0, "drift": 0.0, "policy": 0.0},
        "layer": {"verdict": "HOLD", "fired": ["graph_cluster"]},
        "features": {"age_days": 3.2, "n_prior": 4, "window_n": 2, "window_usd": 140.1, "amount_ratio": 1.4,
                     "device_workers": 1, "destination_workers": 1, "requester_workers": 6, "worker_requesters": 1},
        "outcome": {"paid": True, "paid_at": 25.5, "delay_h": 24.0},
        "truth": {"is_attack": False, "family": "honest"}, "rank": rank,
    }


QUEUE = {"schema_version": 1, "simulated": True, "generated_at": "2026-01-01T00:00:00+00:00",
         "source": {"split": "S", "seed": 1, "defender": "swarm_r2", "replay": "x", "parity_checked_events": 3},
         "thresholds": {"step_up": 0.3, "hold": 0.6},
         "counts": {"events": 3, "RELEASE": 1, "STEP_UP": 0, "HOLD": 2, "cases": 2},
         "specialists": [{"key": "graph", "label": "Graph", "what": "x"}],
         "cases": [make_case(24, 1), make_case(25, 2)]}
EXPL = {"schema_version": 1, "provider": "gemini", "model": "m", "prompt_version": "explain-v1",
        "generated_at": "2026-01-01T00:00:00+00:00", "grounding": "g", "counts": {"llm": 1, "template": 1},
        "explanations": {"24": {"source": "llm", "summary": "s", "evidence": ["e"], "check_next": "c",
                                "suggested_action": "HOLD"},
                         "25": {"source": "template", "summary": "s", "evidence": ["e", "f"], "check_next": "c",
                                "suggested_action": "STEP_UP"}}}


def put(dirpath: Path, name: str, doc) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return p


@pytest.fixture
def dirs(tmp_path):
    return {"console_dir": tmp_path / "console", "copilot_dir": tmp_path / "sessions", "results_root": ROOT}


def client_for(dirs) -> TestClient:
    return TestClient(create_app(DEFAULT_REPLAY, web_dist=dirs["console_dir"].parent / "no-dist", **dirs))


# ---------------------------------------------------------------- queue / explanations --------------------
def test_queue_is_served_and_validated(dirs):
    put(dirs["console_dir"], "queue.json", QUEUE)
    r = client_for(dirs).get("/api/console/queue")
    assert r.status_code == 200
    body = r.json()
    assert [c["event_id"] for c in body["cases"]] == [24, 25] and body["thresholds"]["hold"] == 0.6
    assert body["cases"][0]["truth"]["family"] == "honest" and body["cases"][0]["rank"] == 1


def test_queue_keeps_unknown_extra_fields(dirs):
    doc = json.loads(json.dumps(QUEUE))
    doc["cases"][0]["note"] = "kept"
    put(dirs["console_dir"], "queue.json", doc)
    assert client_for(dirs).get("/api/console/queue").json()["cases"][0]["note"] == "kept"


def test_explanations_are_served(dirs):
    put(dirs["console_dir"], "explanations.json", EXPL)
    r = client_for(dirs).get("/api/console/explanations")
    assert r.status_code == 200 and r.json()["explanations"]["25"]["source"] == "template"


@pytest.mark.parametrize("url", ["/api/console/queue", "/api/console/explanations"])
def test_missing_console_files_are_404_not_built(dirs, url):
    r = client_for(dirs).get(url)
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_built" and "detail" in body


@pytest.mark.parametrize("name,url", [("queue.json", "/api/console/queue"),
                                      ("explanations.json", "/api/console/explanations")])
@pytest.mark.parametrize("content", ["not json at all", "[]", '{"schema_version": 1}'])
def test_invalid_console_files_are_500_invalid_not_a_crash(dirs, name, url, content):
    put(dirs["console_dir"], name, content)
    r = client_for(dirs).get(url)
    assert r.status_code == 500 and r.json()["error"] == "invalid"


def test_queue_with_a_contract_violation_is_rejected(dirs):
    bad = json.loads(json.dumps(QUEUE))
    bad["cases"][0]["verdict"] = "RELEASE"  # the queue holds HOLD and STEP_UP cases only
    put(dirs["console_dir"], "queue.json", bad)
    assert client_for(dirs).get("/api/console/queue").status_code == 500
    bad2 = json.loads(json.dumps(EXPL))
    bad2["explanations"]["24"]["evidence"] = []
    put(dirs["console_dir"], "explanations.json", bad2)
    assert client_for(dirs).get("/api/console/explanations").status_code == 500


def test_a_file_built_after_startup_is_picked_up_and_changes_are_seen(dirs):
    c = client_for(dirs)
    assert c.get("/api/console/queue").status_code == 404
    put(dirs["console_dir"], "queue.json", QUEUE)
    assert len(c.get("/api/console/queue").json()["cases"]) == 2
    smaller = {**QUEUE, "cases": QUEUE["cases"][:1]}
    put(dirs["console_dir"], "queue.json", smaller)
    assert len(c.get("/api/console/queue").json()["cases"]) == 1


def test_models_accept_the_committed_files_if_present():
    for name, model in (("queue.json", models.Queue), ("explanations.json", models.Explanations)):
        p = ROOT / "eval" / "console" / name
        if p.is_file():
            model.model_validate_json(p.read_bytes())


# ---------------------------------------------------------------- copilot sessions ------------------------
def session(id_, order=None, **kw):
    d = {"schema_version": 1, "id": id_, "title": f"t-{id_}", "turns": []}
    if order is not None:
        d["order"] = order
    return {**d, **kw}


def test_sessions_empty_when_directory_is_missing_or_empty(dirs):
    assert client_for(dirs).get("/api/copilot/sessions").json() == {"sessions": []}
    dirs["copilot_dir"].mkdir()
    assert client_for(dirs).get("/api/copilot/sessions").json() == {"sessions": []}


def test_sessions_sorted_by_order_then_id(dirs):
    d = dirs["copilot_dir"]
    put(d, "a.json", session("zeta", 2))
    put(d, "b.json", session("alpha", 2))
    put(d, "c.json", session("first", 1))
    put(d, "d.json", session("noorder"))  # no order: after the ordered ones
    r = client_for(dirs).get("/api/copilot/sessions")
    assert [s["id"] for s in r.json()["sessions"]] == ["first", "alpha", "zeta", "noorder"]


def test_sessions_skip_and_log_bad_files(dirs, caplog):
    d = dirs["copilot_dir"]
    put(d, "good.json", session("good", 1))
    put(d, "broken.json", "{not json")
    put(d, "list.json", "[1, 2]")
    put(d, "v2.json", session("v2", 2, schema_version=2))
    put(d, "nov.json", {"id": "nov"})
    put(d, "notes.txt", "ignored, not .json")
    with caplog.at_level(logging.WARNING, logger="payguard.api"):
        r = client_for(dirs).get("/api/copilot/sessions")
    assert [s["id"] for s in r.json()["sessions"]] == ["good"]
    logged = " ".join(rec.getMessage() for rec in caplog.records)
    for name in ("broken.json", "list.json", "v2.json", "nov.json"):
        assert name in logged


def test_sessions_are_returned_whole(dirs):
    s = session("x", 1, turns=[{"role": "user", "text": "hi"}], toolkit={"read_only": True})
    put(dirs["copilot_dir"], "x.json", s)
    assert client_for(dirs).get("/api/copilot/sessions").json()["sessions"] == [s]


# ---------------------------------------------------------------- rounds ----------------------------------
def test_rounds_endpoint_matches_rounds_doc(dirs):
    r = client_for(dirs).get("/api/results/rounds")
    assert r.status_code == 200
    assert r.json() == json.loads(json.dumps(R.rounds_doc(ROOT)))


def test_rounds_is_cached_after_the_first_call(tmp_path):
    import shutil
    for rel in (R.R1_PATH, R.R2_PATH, R.R2_GEMINI_PATH, R.R3_PATH):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    c = client_for({"console_dir": tmp_path / "c", "copilot_dir": tmp_path / "s", "results_root": tmp_path})
    first = c.get("/api/results/rounds").json()
    (tmp_path / R.R1_PATH).unlink()  # gone from disk: the memoised answer is served
    assert c.get("/api/results/rounds").json() == first


def test_rounds_missing_files_are_404_not_built(tmp_path):
    c = client_for({"console_dir": tmp_path / "c", "copilot_dir": tmp_path / "s", "results_root": tmp_path})
    r = c.get("/api/results/rounds")
    assert r.status_code == 404 and r.json()["error"] == "not_built"


# ---------------------------------------------------------------- nothing else changed --------------------
def test_unknown_api_paths_still_404_json_after_the_new_routes(dirs):
    c = client_for(dirs)
    for path in ("/api/nope", "/api/console", "/api/console/nope", "/api/copilot", "/api/results", "/api/results/x"):
        r = c.get(path)
        assert r.status_code == 404 and "detail" in r.json() and "no such endpoint" in r.json()["detail"], path


def test_catch_all_is_still_registered_after_the_new_routes(dirs):
    app = create_app(DEFAULT_REPLAY, web_dist=dirs["console_dir"].parent / "no-dist", **dirs)
    paths = [getattr(r, "path", "") for r in app.routes]
    catch = paths.index("/api/{rest:path}")
    for p in ("/api/console/queue", "/api/console/explanations", "/api/copilot/sessions", "/api/results/rounds"):
        assert paths.index(p) < catch


def test_existing_endpoints_still_work(dirs):
    c = client_for(dirs)
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.get("/api/replay/summary").status_code == 200
    assert c.get("/api/replay/events?limit=3").json()["limit"] == 3
    assert c.get("/api/payouts").status_code == 200
    assert c.get("/api/webhooks/recent").status_code == 200


def test_default_app_reads_the_repo_files():
    """No overrides: the default directories are the repo's eval/console, eval/copilot/sessions and the repo root."""
    from payguard.api import app as app_module

    assert app_module.DEFAULT_CONSOLE_DIR == ROOT / "eval" / "console"
    assert app_module.DEFAULT_COPILOT_DIR == ROOT / "eval" / "copilot" / "sessions"
    c = TestClient(create_app(DEFAULT_REPLAY, web_dist=ROOT / "nowhere"))
    assert c.get("/api/results/rounds").status_code == 200
    assert c.get("/api/copilot/sessions").status_code == 200
    q = ROOT / "eval" / "console" / "queue.json"
    assert c.get("/api/console/queue").status_code == (200 if q.is_file() else 404)
