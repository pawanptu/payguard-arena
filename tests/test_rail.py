"""Offline tests for the payout rails and the webhook receiver. No network: PayPal is faked."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from payguard.paypal_client import PayPalError
from payguard.rail import LiveGuardError, LiveRail, PayoutRequest, ReceiptStore, SimRail
from payguard.rail import webhook as wh
from payguard.rail.base import parse_iso

T0 = datetime(2026, 11, 7, 12, 0, 0, tzinfo=timezone.utc)
EMAIL = "sb-receiver@example.com"


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += timedelta(seconds=s)


# ---------------------------------------------------------------- sim ----
def test_sim_lifecycle_pending_processing_success():
    clock = Clock()
    rail = SimRail(clock=clock, processing_after_s=3, success_after_s=20)
    r = rail.submit(PayoutRequest(7, 12.5, EMAIL))
    assert (r.mode, r.batch_status, r.item_status) == ("sim", "PENDING", "PENDING")
    assert r.created_at == r.updated_at and r.created_at.endswith("Z")
    clock.advance(2.9)
    assert rail.refresh(r).batch_status == "PENDING"
    clock.advance(0.2)
    p = rail.refresh(r)
    assert (p.batch_status, p.item_status) == ("PROCESSING", "PENDING")
    assert parse_iso(p.updated_at) > parse_iso(p.created_at)
    clock.advance(17)
    s = rail.refresh(p)
    assert (s.batch_status, s.item_status) == ("SUCCESS", "SUCCESS")
    assert s.amount_usd == 12.5
    assert rail.refresh(s) is s  # settled receipts are stable


def test_sim_is_deterministic_and_ids_are_obviously_fake():
    a, b = SimRail(clock=Clock()), SimRail(clock=Clock())
    ra, rb = a.submit(PayoutRequest(1, 3.0, EMAIL)), b.submit(PayoutRequest(1, 3.0, EMAIL))
    assert ra == rb
    assert ra.payout_batch_id.startswith("SIM-") and ra.sender_batch_id.startswith("SIM-")
    assert a.submit(PayoutRequest(1, 99.0, EMAIL)) is ra  # idempotent per event id
    assert a.submit(PayoutRequest(2, 3.0, EMAIL)).payout_batch_id != ra.payout_batch_id


def test_sim_writes_to_store_and_validates_config():
    store, clock = ReceiptStore(), Clock()
    rail = SimRail(clock=clock, store=store)
    r = rail.submit(PayoutRequest(5, 1.0, EMAIL))
    assert store.get(5) == r
    clock.advance(25)
    assert rail.refresh(r).batch_status == "SUCCESS"
    assert store.get(5).batch_status == "SUCCESS"
    with pytest.raises(ValueError):
        SimRail(processing_after_s=30, success_after_s=20)


# --------------------------------------------------------------- live ----
class FakeClient:
    def __init__(self):
        self.created = []
        self.fail_with = None
        self.get_response = None

    def create_payout(self, *, sender_batch_id, items, email_subject="x"):
        if self.fail_with:
            raise self.fail_with
        self.created.append((sender_batch_id, items))
        return {"batch_header": {"payout_batch_id": f"PB{len(self.created)}", "batch_status": "PENDING"}}

    def get_payout(self, payout_batch_id):
        return self.get_response


def live(client=None, **kw):
    kw.setdefault("allowed_receivers", {EMAIL})
    return LiveRail(client or FakeClient(), clock=Clock(), **kw)


def test_live_happy_path_builds_correct_payout():
    c = FakeClient()
    rail = live(c)
    r = rail.submit(PayoutRequest(3, 4.999, EMAIL.upper(), "tip"))
    sbid, items = c.created[0]
    assert items[0]["amount"] == {"value": "5.00", "currency": "USD"}
    assert items[0]["receiver"] == EMAIL and items[0]["note"] == "tip"
    assert (r.mode, r.payout_batch_id, r.sender_batch_id) == ("live", "PB1", sbid)
    assert r.batch_status == "PENDING" and rail.spent_usd == 5.0


def test_live_per_payout_cap_refuses_and_does_not_clamp():
    c = FakeClient()
    rail = live(c, max_payout_usd=5.0)
    with pytest.raises(LiveGuardError) as e:
        rail.submit(PayoutRequest(1, 5.01, EMAIL))
    assert e.value.reason == "cap" and not c.created and rail.spent_usd == 0
    rail.submit(PayoutRequest(2, 5.00, EMAIL))  # exactly at the cap is fine


def test_live_default_limits_are_5_and_50(monkeypatch):
    rail = live()
    assert (float(rail.max_payout), float(rail.budget)) == (5.0, 50.0)


def test_live_budget_is_enforced_across_payouts():
    c = FakeClient()
    rail = live(c, max_payout_usd=5, budget_usd=12)
    rail.submit(PayoutRequest(1, 5, EMAIL))
    rail.submit(PayoutRequest(2, 5, EMAIL))
    with pytest.raises(LiveGuardError) as e:
        rail.submit(PayoutRequest(3, 2.01, EMAIL))
    assert e.value.reason == "budget" and len(c.created) == 2
    rail.submit(PayoutRequest(4, 2.00, EMAIL))  # exactly exhausts the budget
    assert rail.remaining_usd == 0
    with pytest.raises(LiveGuardError):
        rail.submit(PayoutRequest(5, 0.01, EMAIL))


def test_live_allowlist_blocks_other_receivers():
    c = FakeClient()
    rail = live(c)
    with pytest.raises(LiveGuardError) as e:
        rail.submit(PayoutRequest(1, 1, "someone-else@example.com"))
    assert e.value.reason == "allowlist" and not c.created
    with pytest.raises(ValueError):
        LiveRail(c, allowed_receivers=set())


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_live_rejects_non_positive_or_non_finite_amounts(bad):
    with pytest.raises(LiveGuardError) as e:
        live().submit(PayoutRequest(1, bad, EMAIL))
    assert e.value.reason == "amount"


def test_live_sender_batch_id_is_idempotent_and_replay_does_not_respend():
    c = FakeClient()
    rail = live(c)
    r1 = rail.submit(PayoutRequest(9, 2, EMAIL))
    r2 = rail.submit(PayoutRequest(9, 2, EMAIL))
    assert r1 is r2 and len(c.created) == 1 and rail.spent_usd == 2
    from payguard.paypal_client import make_sender_batch_id

    assert r1.sender_batch_id == make_sender_batch_id("ev9")
    assert live(c).submit(PayoutRequest(10, 1, EMAIL)).sender_batch_id != r1.sender_batch_id


def test_live_budget_released_on_definite_rejection_kept_on_ambiguous_failure():
    c = FakeClient()
    rail = live(c)
    c.fail_with = PayPalError(422, "{}", "dbg")
    with pytest.raises(PayPalError):
        rail.submit(PayoutRequest(1, 3, EMAIL))
    assert rail.spent_usd == 0
    c.fail_with = PayPalError(503, "{}", "dbg")
    with pytest.raises(PayPalError):
        rail.submit(PayoutRequest(2, 3, EMAIL))
    assert rail.spent_usd == 3  # outcome unknown: stay conservative


def test_live_refresh_reads_batch_and_item_status_and_updates_store():
    c, store = FakeClient(), ReceiptStore()
    rail = LiveRail(c, allowed_receivers={EMAIL}, clock=Clock(), store=store)
    r = rail.submit(PayoutRequest(1, 2, EMAIL))
    c.get_response = {
        "batch_header": {"payout_batch_id": "PB1", "batch_status": "SUCCESS"},
        "items": [{"transaction_status": "SUCCESS"}],
    }
    s = rail.refresh(r)
    assert (s.batch_status, s.item_status) == ("SUCCESS", "SUCCESS")
    assert store.get(1) == s


def test_live_from_env_reads_guard_variables(monkeypatch):
    for k, v in {
        "PAYPAL_CLIENT_ID": "cid", "PAYPAL_CLIENT_SECRET": "sec",
        "PAYPAL_TEST_RECEIVER_EMAIL": EMAIL, "PAYGUARD_LIVE_MAX_PAYOUT_USD": "2.50",
        "PAYGUARD_LIVE_BUDGET_USD": "7", "PAYPAL_BASE_URL": "https://api-m.sandbox.paypal.com",
    }.items():
        monkeypatch.setenv(k, v)
    rail = LiveRail.from_env()
    assert (float(rail.max_payout), float(rail.budget)) == (2.5, 7.0)
    assert rail.allowed == {EMAIL}
    monkeypatch.setenv("PAYPAL_BASE_URL", "https://api-m.paypal.com")
    with pytest.raises(ValueError):  # the client still refuses production
        LiveRail.from_env()


# ------------------------------------------------------------ webhook ----
SIG = {
    "PAYPAL-AUTH-ALGO": "SHA256withRSA",
    "PAYPAL-CERT-URL": "https://api.sandbox.paypal.com/cert",
    "PAYPAL-TRANSMISSION-ID": "tx-1",
    "PAYPAL-TRANSMISSION-SIG": "sig",
    "PAYPAL-TRANSMISSION-TIME": "2026-11-07T12:00:00Z",
}


class FakeVerifier:
    def __init__(self, result=True, raises=None):
        self.result, self.raises, self.calls = result, raises, []

    def verify_webhook_signature(self, **kw):
        self.calls.append(kw)
        if self.raises:
            raise self.raises
        return self.result


def make_app(webhook_id="WH-1", verifier=None, store=None):
    app = FastAPI()
    app.include_router(wh.router)
    store = store if store is not None else ReceiptStore()
    verifier = verifier if verifier is not None else FakeVerifier()
    app.dependency_overrides[wh.get_receipt_store] = lambda: store
    app.dependency_overrides[wh.get_webhook_id] = lambda: webhook_id
    app.dependency_overrides[wh.get_paypal_client] = lambda: verifier
    return TestClient(app), store, verifier


def seeded_store():
    store = ReceiptStore()
    SimRail(clock=Clock(), store=store).submit(PayoutRequest(1, 3.0, EMAIL))
    return store  # payout_batch_id SIM-PB-1, sender_batch_id SIM-SB-1


def batch_event(eid="WH-EV-1", typ="PAYMENT.PAYOUTSBATCH.PROCESSING", status="PROCESSING"):
    return {"id": eid, "event_type": typ, "resource": {"batch_header": {
        "payout_batch_id": "SIM-PB-1", "batch_status": status,
        "sender_batch_header": {"sender_batch_id": "SIM-SB-1"}}}}


def item_event(eid="WH-EV-2", typ="PAYMENT.PAYOUTS-ITEM.SUCCEEDED", status="SUCCESS"):
    return {"id": eid, "event_type": typ, "resource": {
        "payout_batch_id": "SIM-PB-1", "sender_batch_id": "SIM-SB-1", "transaction_status": status}}


def test_webhook_503_without_webhook_id_and_accepts_nothing():
    client, store, verifier = make_app(webhook_id=None, store=seeded_store())
    r = client.post("/webhooks/paypal", json=batch_event(), headers=SIG)
    assert r.status_code == 503
    assert verifier.calls == [] and store.get(1).batch_status == "PENDING"


def test_webhook_503_without_verifier_credentials():
    app = FastAPI()
    app.include_router(wh.router)
    app.dependency_overrides[wh.get_webhook_id] = lambda: "WH-1"
    app.dependency_overrides[wh.get_paypal_client] = lambda: None
    assert TestClient(app).post("/webhooks/paypal", json=batch_event(), headers=SIG).status_code == 503


def test_webhook_rejected_when_verification_fails():
    client, store, verifier = make_app(verifier=FakeVerifier(False), store=seeded_store())
    r = client.post("/webhooks/paypal", json=batch_event(), headers=SIG)
    assert r.status_code == 401
    assert store.get(1).batch_status == "PENDING"
    # a rejected event is not remembered, so a later genuine redelivery is still processed
    assert store.mark_event_seen("WH-EV-1") is True


def test_webhook_verification_outage_is_503_not_accept():
    client, store, _ = make_app(verifier=FakeVerifier(raises=PayPalError(500, "x", None)),
                                store=seeded_store())
    assert client.post("/webhooks/paypal", json=batch_event(), headers=SIG).status_code == 503
    assert store.get(1).batch_status == "PENDING"


def test_webhook_missing_signature_headers_and_bad_json():
    client, store, verifier = make_app(store=seeded_store())
    assert client.post("/webhooks/paypal", json=batch_event()).status_code == 400
    assert client.post("/webhooks/paypal", content=b"{nope", headers=SIG).status_code == 400
    assert client.post("/webhooks/paypal", content=b"[1]", headers=SIG).status_code == 400
    assert verifier.calls == []


def test_webhook_accepted_updates_store_and_passes_verification_inputs():
    client, store, verifier = make_app(store=seeded_store())
    r = client.post("/webhooks/paypal", json=batch_event(), headers=SIG)
    assert r.status_code == 200 and r.json()["status"] == "processed"
    assert store.get(1).batch_status == "PROCESSING"
    call = verifier.calls[0]
    assert call["webhook_id"] == "WH-1" and call["transmission_id"] == "tx-1"
    assert call["auth_algo"] == "SHA256withRSA" and call["webhook_event"]["id"] == "WH-EV-1"

    r = client.post("/webhooks/paypal", json=item_event(), headers=SIG)
    assert r.json()["status"] == "processed" and store.get(1).item_status == "SUCCESS"
    r = client.post("/webhooks/paypal", json=batch_event("WH-EV-3", "PAYMENT.PAYOUTSBATCH.SUCCESS", "SUCCESS"),
                    headers=SIG)
    assert store.get(1).batch_status == "SUCCESS"


def test_webhook_duplicate_event_is_ignored():
    client, store, verifier = make_app(store=seeded_store())
    assert client.post("/webhooks/paypal", json=batch_event(), headers=SIG).json()["status"] == "processed"
    first = store.get(1)
    # same id again (PayPal retry), even carrying different content, changes nothing
    dup = batch_event(typ="PAYMENT.PAYOUTSBATCH.SUCCESS", status="SUCCESS")
    r = client.post("/webhooks/paypal", json=dup, headers=SIG)
    assert r.status_code == 200 and r.json()["status"] == "duplicate"
    assert store.get(1) == first


def test_webhook_unknown_or_malformed_events_are_ignored_safely():
    client, store, _ = make_app(store=seeded_store())
    cases = [
        {"id": "U1", "event_type": "PAYMENT.SALE.COMPLETED", "resource": {}},
        {"id": "U2", "event_type": "PAYMENT.PAYOUTS-ITEM.SUCCEEDED", "resource": "oops"},
        {"id": "U3", "event_type": "PAYMENT.PAYOUTS-ITEM.SUCCEEDED", "resource": {}},
        {"event_type": "PAYMENT.PAYOUTSBATCH.SUCCESS"},
        {"id": "U5"},
    ]
    for ev in cases:
        r = client.post("/webhooks/paypal", json=ev, headers=SIG)
        assert r.status_code == 200 and r.json()["status"] == "ignored", ev
    assert store.get(1).batch_status == "PENDING"


def test_webhook_for_unknown_receipt_is_parked_then_applied_when_receipt_arrives():
    client, store, _ = make_app()
    r = client.post("/webhooks/paypal", json=batch_event(), headers=SIG)
    assert r.status_code == 200 and r.json()["status"] == "pending_receipt"
    SimRail(clock=Clock(), store=store).submit(PayoutRequest(1, 3.0, EMAIL))
    assert store.get(1).batch_status == "PROCESSING"


def test_webhook_item_event_without_transaction_status_uses_event_type():
    client, store, _ = make_app(store=seeded_store())
    ev = item_event(typ="PAYMENT.PAYOUTS-ITEM.HELD")
    del ev["resource"]["transaction_status"]
    client.post("/webhooks/paypal", json=ev, headers=SIG)
    assert store.get(1).item_status == "ONHOLD"


def test_out_of_order_webhooks_never_regress_a_settled_receipt():
    client, store, _ = make_app(store=seeded_store())
    client.post("/webhooks/paypal", json=batch_event("a", "PAYMENT.PAYOUTSBATCH.SUCCESS", "SUCCESS"), headers=SIG)
    client.post("/webhooks/paypal", json=batch_event("b"), headers=SIG)  # late PROCESSING
    assert store.get(1).batch_status == "SUCCESS"


# -------------------------------------------------------------- store ----
def mk_receipt(i):
    return SimRail(clock=Clock()).submit(PayoutRequest(i, 1.0, EMAIL))


def test_store_is_bounded_and_recent_is_newest_first():
    store = ReceiptStore(max_receipts=3)
    for i in range(1, 6):
        store.upsert(mk_receipt(i))
    assert len(store) == 3 and store.get(1) is None and store.get(2) is None
    assert [r.event_id for r in store.recent(2)] == [5, 4]
    assert store.find(payout_batch_id="SIM-PB-1") is None  # evicted ids leave the index too
    assert store.find(payout_batch_id="SIM-PB-5").event_id == 5


def test_store_is_thread_safe_under_concurrent_upserts():
    import threading

    store = ReceiptStore(max_receipts=50)

    def work(base):
        for i in range(200):
            store.upsert(mk_receipt(base * 1000 + i))
            store.recent(5)

    ts = [threading.Thread(target=work, args=(b,)) for b in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(store) == 50 and len(store._by_batch) == 50
