"""Verified-webhook event log, /api/webhooks/recent, store seeding from the replay, additive replay field.
No network: PayPal's signature check is a fake."""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from payguard.api.app import DEFAULT_REPLAY, create_app
from payguard.api.replay import load_replay, parse_replay
from payguard.rail import webhook as wh
from payguard.rail.events import WebhookEventLog

SIG = {
    "PAYPAL-AUTH-ALGO": "SHA256withRSA",
    "PAYPAL-CERT-URL": "https://api.sandbox.paypal.com/cert",
    "PAYPAL-TRANSMISSION-ID": "tx-1",
    "PAYPAL-TRANSMISSION-SIG": "sig",
    "PAYPAL-TRANSMISSION-TIME": "2026-11-07T12:00:00Z",
}
EMAIL = "private.person@example.com"
NO_DIST = Path("does-not-exist")


class Verifier:
    def __init__(self, ok=True):
        self.ok, self.calls = ok, 0

    def verify_webhook_signature(self, **kw):
        self.calls += 1
        return self.ok


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(DEFAULT_REPLAY.read_text(encoding="utf-8"))


def live_replay(tmp_path: Path, raw: dict, *, webhooks: list[dict] | None = None) -> Path:
    data = copy.deepcopy(raw)
    data["payouts"].append({
        "event_id": 40, "mode": "live", "sender_batch_id": "SB-LIVE-1", "payout_batch_id": "PB-LIVE-1",
        "batch_status": "PENDING", "item_status": "PENDING", "amount_usd": 1.0,
        "created_at": "2026-11-07T10:00:00Z"})
    if webhooks is not None:
        data["verified_webhooks"] = webhooks
    p = tmp_path / "live.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def make(replay=DEFAULT_REPLAY, *, webhook_id="WH-1", verifier=None):
    app = create_app(replay, web_dist=NO_DIST)
    verifier = verifier if verifier is not None else Verifier()
    app.dependency_overrides[wh.get_webhook_id] = lambda: webhook_id
    app.dependency_overrides[wh.get_paypal_client] = lambda: verifier
    return TestClient(app), app, verifier


def item_event(eid="EV-1", batch="PB-LIVE-1", status="SUCCESS", typ="PAYMENT.PAYOUTS-ITEM.SUCCEEDED"):
    return {"id": eid, "event_type": typ, "resource": {
        "payout_batch_id": batch, "sender_batch_id": "SB-LIVE-1", "transaction_status": status,
        "payout_item": {"receiver": EMAIL, "payout_item_fee": {"value": "0.25"}},
        "payout_item_id": "ITEM-9", "name": "Private Person"}}


def batch_event(eid="EV-2", batch="PB-LIVE-1", status="PROCESSING", typ="PAYMENT.PAYOUTSBATCH.PROCESSING"):
    return {"id": eid, "event_type": typ, "resource": {"batch_header": {
        "payout_batch_id": batch, "batch_status": status,
        "sender_batch_header": {"sender_batch_id": "SB-LIVE-1", "email_subject": "x"}}}}


# ---- WebhookEventLog -------------------------------------------------------------------------------
def test_log_is_bounded_newest_first_and_oldest_dropped():
    log = WebhookEventLog(max_events=3)
    for i in range(5):
        log.record(event_id=f"E{i}", event_type="PAYMENT.PAYOUTSBATCH.SUCCESS", payout_batch_id=f"PB{i}",
                   outcome="processed")
    assert len(log) == 3
    assert [e["event_id"] for e in log.recent(10)] == ["E4", "E3", "E2"]
    assert [e["event_id"] for e in log.recent(2)] == ["E4", "E3"]
    assert log.recent(0) == []
    assert WebhookEventLog().recent() == []
    with pytest.raises(ValueError):
        WebhookEventLog(max_events=0)
    with pytest.raises(ValueError):
        log.record(event_id="x", event_type="y", outcome="bogus")


def test_log_default_capacity_is_200():
    log = WebhookEventLog()
    for i in range(250):
        log.record(event_id=f"E{i}", event_type="T", outcome="processed")
    assert len(log) == 200 and log.recent(1)[0]["event_id"] == "E249"
    assert log.recent(300)[-1]["event_id"] == "E50"


def test_log_keeps_only_whitelisted_fields_and_drops_non_strings():
    log = WebhookEventLog()
    rec = log.record(event_id="E1", event_type="T", payout_batch_id={"email": EMAIL}, sender_batch_id="S" * 500,
                     batch_status=None, item_status="SUCCESS", outcome="pending_receipt")
    assert set(rec.__dataclass_fields__) == {"event_id", "event_type", "payout_batch_id", "sender_batch_id",
                                             "batch_status", "item_status", "received_at", "outcome"}
    d = log.recent(1)[0]
    assert d["payout_batch_id"] is None and len(d["sender_batch_id"]) == 128
    assert d["received_at"].endswith("Z") and d["outcome"] == "pending_receipt"


def test_log_is_thread_safe():
    log = WebhookEventLog(max_events=50)

    def work(k):
        for i in range(100):
            log.record(event_id=f"{k}-{i}", event_type="T", outcome="processed")

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(log) == 50


# ---- recording rules in the webhook ----------------------------------------------------------------
def test_verified_event_is_logged_with_ids_and_statuses_only(tmp_path, raw):
    c, app, _ = make(live_replay(tmp_path, raw))
    assert c.post("/webhooks/paypal", json=item_event(), headers=SIG).json()["status"] == "processed"
    (rec,) = app.state.event_log.recent(10)
    assert rec["event_id"] == "EV-1" and rec["event_type"] == "PAYMENT.PAYOUTS-ITEM.SUCCEEDED"
    assert rec["payout_batch_id"] == "PB-LIVE-1" and rec["sender_batch_id"] == "SB-LIVE-1"
    assert rec["item_status"] == "SUCCESS" and rec["batch_status"] is None and rec["outcome"] == "processed"
    blob = json.dumps(c.get("/api/webhooks/recent").json())
    for leaked in (EMAIL, "Private Person", "ITEM-9", "payout_item", "0.25", "resource"):
        assert leaked not in blob


def test_batch_event_and_pending_receipt_outcome():
    c, app, _ = make()
    r = c.post("/webhooks/paypal", json=batch_event(batch="PB-UNKNOWN"), headers=SIG)
    assert r.json()["status"] == "pending_receipt"
    (rec,) = app.state.event_log.recent(10)
    assert rec["outcome"] == "pending_receipt" and rec["batch_status"] == "PROCESSING"


def test_invalid_signature_is_not_logged():
    c, app, v = make(verifier=Verifier(ok=False))
    assert c.post("/webhooks/paypal", json=item_event(), headers=SIG).status_code == 401
    assert v.calls == 1 and len(app.state.event_log) == 0
    # not configured / missing headers / bad body: nothing logged either
    c2, app2, _ = make(webhook_id=None)
    assert c2.post("/webhooks/paypal", json=item_event(), headers=SIG).status_code == 503
    c3, app3, _ = make()
    assert c3.post("/webhooks/paypal", json=item_event()).status_code == 400
    assert len(app2.state.event_log) == 0 and len(app3.state.event_log) == 0


def test_duplicate_unhandled_and_unreferenced_events_are_not_logged(tmp_path, raw):
    c, app, _ = make(live_replay(tmp_path, raw))
    assert c.post("/webhooks/paypal", json=item_event(), headers=SIG).json()["status"] == "processed"
    assert c.post("/webhooks/paypal", json=item_event(), headers=SIG).json()["status"] == "duplicate"
    for ev in ({"id": "U1", "event_type": "PAYMENT.SALE.COMPLETED", "resource": {}},
               {"id": "U2", "event_type": "PAYMENT.PAYOUTS-ITEM.SUCCEEDED", "resource": {}},
               {"id": "U3", "event_type": "PAYMENT.PAYOUTS-ITEM.SUCCEEDED", "resource": "oops"},
               {"event_type": "PAYMENT.PAYOUTSBATCH.SUCCESS"}):
        assert c.post("/webhooks/paypal", json=ev, headers=SIG).json()["status"] == "ignored"
    assert [e["event_id"] for e in app.state.event_log.recent(10)] == ["EV-1"]


# ---- GET /api/webhooks/recent ----------------------------------------------------------------------
def test_recent_empty_and_configured_flag():
    c, _, _ = make(webhook_id="WH-1")
    assert c.get("/api/webhooks/recent").json() == {"configured": True, "events": []}
    c, _, _ = make(webhook_id=None)
    assert c.get("/api/webhooks/recent").json() == {"configured": False, "events": []}


def test_configured_flag_reads_the_environment_by_default(monkeypatch):
    monkeypatch.setattr("payguard.config.load_dotenv", lambda *a, **k: None)
    c = TestClient(create_app(DEFAULT_REPLAY, web_dist=NO_DIST))
    monkeypatch.delenv("PAYPAL_WEBHOOK_ID", raising=False)
    assert c.get("/api/webhooks/recent").json()["configured"] is False
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-ENV")
    assert c.get("/api/webhooks/recent").json()["configured"] is True


def test_recent_merges_runtime_and_replay_dedupes_and_orders_newest_first(tmp_path, raw):
    replay_events = [
        {"event_id": "OLD-1", "event_type": "PAYMENT.PAYOUTSBATCH.PROCESSING", "received_at": "2026-11-06T10:00:00Z",
         "payout_batch_id": "PB-X", "batch_status": "PROCESSING"},
        {"event_id": "EV-1", "event_type": "PAYMENT.PAYOUTS-ITEM.SUCCEEDED", "received_at": "2026-11-06T10:05:00Z",
         "payout_batch_id": "PB-LIVE-1", "item_status": "PENDING"},  # same id as the runtime one
        {"event_id": "FUTURE", "event_type": "PAYMENT.PAYOUTSBATCH.SUCCESS", "received_at": "2999-01-01T00:00:00Z",
         "payout_batch_id": "PB-X", "batch_status": "SUCCESS"},
    ]
    c, _, _ = make(live_replay(tmp_path, raw, webhooks=replay_events))
    r = c.get("/api/webhooks/recent").json()
    assert [(e["event_id"], e["source"]) for e in r["events"]] == [
        ("FUTURE", "replay"), ("EV-1", "replay"), ("OLD-1", "replay")]  # before any runtime event
    c.post("/webhooks/paypal", json=item_event("EV-1"), headers=SIG)
    c.post("/webhooks/paypal", json=batch_event("EV-2"), headers=SIG)
    events = c.get("/api/webhooks/recent").json()["events"]
    ids = [e["event_id"] for e in events]
    assert ids == ["FUTURE", "EV-2", "EV-1", "OLD-1"]  # newest first, one EV-1
    by_id = {e["event_id"]: e for e in events}
    assert by_id["EV-1"]["source"] == "runtime" and by_id["EV-1"]["item_status"] == "SUCCESS"  # runtime wins
    assert by_id["EV-1"]["outcome"] == "processed"
    assert by_id["OLD-1"]["source"] == "replay" and by_id["OLD-1"]["outcome"] is None
    assert set(by_id["OLD-1"]) == {"event_id", "event_type", "received_at", "payout_batch_id", "sender_batch_id",
                                   "batch_status", "item_status", "outcome", "source"}


def test_recent_limit_is_clamped(tmp_path, raw):
    many = [{"event_id": f"R{i:03d}", "event_type": "PAYMENT.PAYOUTSBATCH.SUCCESS",
             "received_at": f"2026-11-06T10:{i // 60:02d}:{i % 60:02d}Z"} for i in range(230)]
    c, _, _ = make(live_replay(tmp_path, raw, webhooks=many))
    get = lambda **p: c.get("/api/webhooks/recent", params=p).json()["events"]  # noqa: E731
    assert len(get()) == 50
    assert len(get(limit=7)) == 7
    assert len(get(limit=0)) == 1 and len(get(limit=-5)) == 1
    assert len(get(limit=10_000)) == 200
    assert get(limit=2)[0]["event_id"] == "R229"
    assert c.get("/api/webhooks/recent", params={"limit": "x"}).status_code == 422


def test_unparsable_timestamps_sort_oldest_without_crashing(tmp_path, raw):
    ev = [{"event_id": "A", "event_type": "T", "received_at": "garbage"},
          {"event_id": "B", "event_type": "T", "received_at": "2026-11-06T10:00:00"}]  # naive -> UTC
    c, _, _ = make(live_replay(tmp_path, raw, webhooks=ev))
    assert [e["event_id"] for e in c.get("/api/webhooks/recent").json()["events"]] == ["B", "A"]


# ---- seeding the store from the replay -------------------------------------------------------------
def test_seed_then_webhook_updates_exactly_one_row(tmp_path, raw):
    c, app, _ = make(live_replay(tmp_path, raw))
    assert app.state.receipt_store.find(payout_batch_id="PB-LIVE-1").item_status == "PENDING"
    before = c.get("/api/payouts").json()
    assert before["source"] == "replay"  # seeded rows alone are not "runtime"
    assert len(before["payouts"]) == len(raw["payouts"]) + 1

    assert c.post("/webhooks/paypal", json=item_event(), headers=SIG).json()["status"] == "processed"
    after = c.get("/api/payouts").json()
    rows = [p for p in after["payouts"] if p["payout_batch_id"] == "PB-LIVE-1"]
    assert len(rows) == 1 and rows[0]["item_status"] == "SUCCESS" and rows[0]["mode"] == "live"
    assert after["source"] == "runtime+replay"
    assert len(after["payouts"]) == len(before["payouts"])  # no duplicates
    sender_ids = [p["sender_batch_id"] for p in after["payouts"]]
    assert len(sender_ids) == len(set(sender_ids))
    assert after["payouts"][0]["payout_batch_id"] == "PB-LIVE-1"  # store rows first


def test_seeding_covers_live_rows_only_and_keeps_replay_order_for_the_rest(raw):
    app = create_app(DEFAULT_REPLAY, web_dist=NO_DIST)
    store = app.state.receipt_store
    live = [p for p in raw["payouts"] if p["mode"] == "live"]
    assert len(store) == len(live) and live  # the fixture has a live row; sim rows are not seeded
    got = TestClient(app).get("/api/payouts").json()["payouts"]
    assert len(got) == len(raw["payouts"])
    assert got[0]["mode"] == "live" and [p["sender_batch_id"] for p in got[1:]] == [
        p["sender_batch_id"] for p in raw["payouts"] if p["mode"] == "sim"]
    seeded = store.get(live[0]["event_id"])
    assert seeded.updated_at == seeded.created_at == live[0]["created_at"]


def test_a_webhook_that_arrives_after_a_restart_finds_the_seeded_receipt(tmp_path, raw):
    path = live_replay(tmp_path, raw)
    c1, _, _ = make(path)
    c1.post("/webhooks/paypal", json=batch_event(status="SUCCESS", typ="PAYMENT.PAYOUTSBATCH.SUCCESS"),
            headers=SIG)
    c2, app2, _ = make(path)  # "restart": fresh store and log, same replay
    assert len(app2.state.event_log) == 0
    r = c2.post("/webhooks/paypal", json=batch_event("EV-3", status="SUCCESS", typ="PAYMENT.PAYOUTSBATCH.SUCCESS"),
                headers=SIG)
    assert r.json()["status"] == "processed"  # not parked as an orphan


# ---- replay contract: additive field ---------------------------------------------------------------
def test_old_replay_without_verified_webhooks_still_loads(raw, tmp_path):
    assert "verified_webhooks" not in raw  # the committed fixture predates the field
    p = tmp_path / "old.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    r = load_replay(p)
    assert r.schema_version == 1 and r.verified_webhooks == []
    c = TestClient(create_app(p, web_dist=NO_DIST))
    assert c.get("/api/replay/summary").json()["verified_webhooks"] == []
    assert c.get("/api/webhooks/recent").json()["events"] == []


def test_replay_with_verified_webhooks_round_trips_and_validates(raw):
    data = copy.deepcopy(raw)
    data["verified_webhooks"] = [{"event_id": "E1", "event_type": "T", "received_at": "2026-11-07T10:00:00Z",
                                  "payout_batch_id": "PB1"}]
    r = parse_replay(data)
    assert r.verified_webhooks[0].sender_batch_id is None and r.verified_webhooks[0].payout_batch_id == "PB1"
    assert r.schema_version == 1
    bad = copy.deepcopy(data)
    del bad["verified_webhooks"][0]["event_id"]
    with pytest.raises(Exception, match=r"verified_webhooks\[0\]\.event_id"):
        parse_replay(bad)
