"""Dashboard API: replay contract validation and every endpoint (no network, no secrets)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from payguard.api.app import DEFAULT_REPLAY, create_app
from payguard.api.replay import ReplayError, load_replay, parse_replay


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(DEFAULT_REPLAY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(DEFAULT_REPLAY, web_dist=Path("does-not-exist")))


def write(tmp_path: Path, data: object, name: str = "r.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---- contract -------------------------------------------------------------------------------------

def test_fixture_loads_and_is_honest_about_what_it_is(raw):
    r = load_replay(DEFAULT_REPLAY)
    assert r.schema_version == 1 and r.simulated is True
    assert r.synthetic_fixture is True
    assert {x.split for x in r.results} == {"TRAIN"}  # TEST_SEEN / TEST_HELDOUT never run for the fixture
    assert {x.defender for x in r.results} == {"always_release", "always_hold", "static_rules"}
    assert r.episode.split == "TRAIN" and r.episode.seed == 1000
    assert all(e.signals is None and e.weights is None for e in r.episode.events)
    assert any(e.is_attack for e in r.episode.events)
    assert all(p.sender_batch_id.startswith("PLACEHOLDER") for p in r.payouts)
    for row in r.results:
        assert row.objective == pytest.approx(row.leakage_pct + 0.5 * row.honest_block_rate + row.honest_unpaid_rate)


@pytest.mark.parametrize("mutate, field", [
    (lambda d: d.pop("simulated"), "simulated"),
    (lambda d: d.update(simulated=False), "simulated"),
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d.pop("synthetic_fixture"), "synthetic_fixture"),
    (lambda d: d["results"][1].update(split="VALIDATION"), "results[1].split"),
    (lambda d: d["results"][0].pop("leakage_pct"), "results[0].leakage_pct"),
    (lambda d: d["results"][2].update(objective="n/a"), "results[2].objective"),
    (lambda d: d["evolution"][0].update(promoted="maybe"), "evolution[0].promoted"),
    (lambda d: d["episode"]["events"][3].update(verdict="BLOCK"), "episode.events[3].verdict"),
    (lambda d: d["episode"].pop("seed"), "episode.seed"),
    (lambda d: d["payouts"][0].update(mode="prod"), "payouts[0].mode"),
])
def test_broken_replay_names_the_bad_field(raw, tmp_path, mutate, field):
    bad = copy.deepcopy(raw)
    mutate(bad)
    with pytest.raises(ReplayError) as exc:
        load_replay(write(tmp_path, bad))
    assert field in str(exc.value)


def test_non_finite_number_is_rejected_with_its_field(raw, tmp_path):
    bad = copy.deepcopy(raw)
    bad["results"][0]["honest_median_ttp_h"] = float("nan")
    p = tmp_path / "nan.json"
    p.write_text(json.dumps(bad), encoding="utf-8")  # json.dumps writes NaN, json.loads reads it back
    with pytest.raises(ReplayError, match=r"results\[0\]\.honest_median_ttp_h"):
        load_replay(p)


def test_unreadable_and_non_json_files(tmp_path):
    with pytest.raises(ReplayError, match="cannot read"):
        load_replay(tmp_path / "missing.json")
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReplayError, match="not valid JSON"):
        load_replay(p)
    with pytest.raises(ReplayError, match="schema_version"):
        parse_replay({})


def test_create_app_fails_loudly_on_a_broken_replay(raw, tmp_path):
    bad = copy.deepcopy(raw)
    bad["results"][0]["split"] = "NOPE"
    with pytest.raises(ReplayError, match=r"results\[0\]\.split"):
        create_app(write(tmp_path, bad))


# ---- endpoints ------------------------------------------------------------------------------------

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["simulated"] is True and body["synthetic_fixture"] is True


def test_summary_has_everything_but_the_events(client, raw):
    r = client.get("/api/replay/summary")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"schema_version", "simulated", "synthetic_fixture", "generated_at", "objective",
                         "results", "evolution", "episode", "payouts"}
    assert "events" not in body["episode"]
    assert body["episode"] == {"seed": 1000, "split": "TRAIN", "defender": "static_rules",
                               "n_events": len(raw["episode"]["events"])}
    assert len(body["results"]) == len(raw["results"]) and len(body["evolution"]) == len(raw["evolution"])


def test_events_paging(client, raw):
    total = len(raw["episode"]["events"])
    first = client.get("/api/replay/events", params={"offset": 0, "limit": 5}).json()
    assert first["total"] == total and first["offset"] == 0 and first["limit"] == 5
    assert [e["id"] for e in first["events"]] == [e["id"] for e in raw["episode"]["events"][:5]]
    second = client.get("/api/replay/events", params={"offset": 5, "limit": 5}).json()
    assert second["events"][0]["id"] == raw["episode"]["events"][5]["id"]
    assert set(first["events"][0]) == set(raw["episode"]["events"][0])


def test_events_default_limit_and_cap(client):
    default = client.get("/api/replay/events").json()
    assert default["limit"] == 200 and len(default["events"]) == 200
    big = client.get("/api/replay/events", params={"limit": 10_000_000}).json()
    assert big["limit"] == 1000 and len(big["events"]) == 1000


def test_events_paging_bounds(client):
    total = client.get("/api/replay/events", params={"limit": 1}).json()["total"]
    past_end = client.get("/api/replay/events", params={"offset": total + 50}).json()
    assert past_end["events"] == [] and past_end["total"] == total
    tail = client.get("/api/replay/events", params={"offset": total - 2, "limit": 10}).json()
    assert len(tail["events"]) == 2
    assert client.get("/api/replay/events", params={"offset": -1}).status_code == 422
    assert client.get("/api/replay/events", params={"limit": 0}).status_code == 422
    assert client.get("/api/replay/events", params={"limit": "x"}).status_code == 422


def test_payouts_endpoint_and_seam(raw):
    app = create_app(DEFAULT_REPLAY, web_dist=Path("does-not-exist"))
    c = TestClient(app)
    body = c.get("/api/payouts").json()
    assert body["source"] == "replay" and len(body["payouts"]) == len(raw["payouts"])
    assert {"event_id", "mode", "sender_batch_id", "payout_batch_id", "batch_status", "item_status",
            "amount_usd", "created_at"} <= set(body["payouts"][0])
    # the provider is swappable (tests, alternative stores)
    app.state.payouts_provider = lambda: [{**body["payouts"][0], "mode": "live"}]
    assert c.get("/api/payouts").json()["payouts"][0]["mode"] == "live"


def test_runtime_receipts_and_verified_webhook_reach_the_payouts_endpoint(raw):
    """End to end inside one app: a rail writes a receipt to the shared store, a verified PayPal
    webhook advances it, and /api/payouts lists it first (newest runtime receipts before replay rows)."""
    from payguard.rail import webhook as wh
    from payguard.rail.base import PayoutRequest
    from payguard.rail.sim_rail import SimRail
    from payguard.rail.store import ReceiptStore

    class Verifier:
        def verify_webhook_signature(self, **kw):
            return kw["webhook_id"] == "WH-1"

    store = ReceiptStore()
    app = create_app(DEFAULT_REPLAY, web_dist=Path("does-not-exist"), receipt_store=store)
    app.dependency_overrides[wh.get_webhook_id] = lambda: "WH-1"
    app.dependency_overrides[wh.get_paypal_client] = lambda: Verifier()
    c = TestClient(app)

    receipt = SimRail(store=store).submit(PayoutRequest(7, 3.0, "worker@example.com"))
    event = {"id": "WH-EV-9", "event_type": "PAYMENT.PAYOUTS-ITEM.SUCCEEDED",
             "resource": {"payout_batch_id": receipt.payout_batch_id, "sender_batch_id": receipt.sender_batch_id,
                          "transaction_status": "SUCCESS"}}
    sig = {"PAYPAL-AUTH-ALGO": "SHA256withRSA", "PAYPAL-CERT-URL": "https://api.sandbox.paypal.com/cert",
           "PAYPAL-TRANSMISSION-ID": "tx-9", "PAYPAL-TRANSMISSION-SIG": "sig",
           "PAYPAL-TRANSMISSION-TIME": "2026-11-07T12:00:00Z"}
    r = c.post("/webhooks/paypal", json=event, headers=sig)
    assert r.status_code == 200 and r.json()["status"] == "processed"

    body = c.get("/api/payouts").json()
    assert body["source"] == "runtime+replay"
    assert len(body["payouts"]) == 1 + len(raw["payouts"])
    first = body["payouts"][0]
    assert first["event_id"] == 7 and first["mode"] == "sim" and first["item_status"] == "SUCCESS"


def test_unknown_api_path_is_404_json(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and "detail" in r.json()


def test_replay_path_from_env(raw, tmp_path, monkeypatch):
    alt = copy.deepcopy(raw)
    alt["synthetic_fixture"] = False
    monkeypatch.setenv("PAYGUARD_REPLAY_PATH", str(write(tmp_path, alt)))
    c = TestClient(create_app(web_dist=Path("does-not-exist")))
    assert c.get("/api/health").json()["synthetic_fixture"] is False


# ---- web/dist -------------------------------------------------------------------------------------

def test_api_works_without_web_dist(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 404  # no UI mounted, but nothing crashes


def test_serves_built_app_with_spa_fallback(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>SHELL</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    c = TestClient(create_app(DEFAULT_REPLAY, web_dist=dist))
    assert "SHELL" in c.get("/").text
    assert "SHELL" in c.get("/some/client/route").text
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert c.get("/api/health").json()["status"] == "ok"  # API still wins over the fallback
    assert c.get("/api/nope").status_code == 404
    assert "secret" not in c.get("/../secret.txt").text
    assert "secret" not in c.get("/%2e%2e/secret.txt").text


def test_committed_web_dist_is_served_if_present():
    from payguard.api.app import DEFAULT_WEB_DIST
    if not (DEFAULT_WEB_DIST / "index.html").is_file():
        pytest.skip("web/dist not built")
    c = TestClient(create_app(DEFAULT_REPLAY))
    r = c.get("/")
    assert r.status_code == 200 and '<div id="root">' in r.text
