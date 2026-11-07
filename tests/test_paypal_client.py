"""Offline tests for the PayPal client: no network, requests are checked by a mock transport."""
import json

import httpx
import pytest

from payguard.paypal_client import (
    SANDBOX_BASE,
    PayPalClient,
    PayPalError,
    make_sender_batch_id,
    payout_item,
)


def make_client(handler) -> PayPalClient:
    http = httpx.Client(base_url=SANDBOX_BASE, transport=httpx.MockTransport(handler))
    return PayPalClient("cid", "secret", http=http)


def test_refuses_non_sandbox_host():
    with pytest.raises(ValueError):
        PayPalClient("cid", "secret", base_url="https://api-m.paypal.com")


def test_token_is_cached_and_payout_body_is_correct():
    calls = {"token": 0}
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth2/token":
            calls["token"] += 1
            assert request.headers["authorization"].startswith("Basic ")
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 32400})
        assert request.headers["authorization"] == "Bearer tok"
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201, json={"batch_header": {"payout_batch_id": "B1", "batch_status": "PENDING"}}
        )

    client = make_client(handler)
    item = payout_item("w@example.com", 12.5, sender_item_id="i-1")
    client.create_payout(sender_batch_id="b-1", items=[item])
    client.create_payout(sender_batch_id="b-2", items=[item])

    assert calls["token"] == 1
    assert seen["body"]["sender_batch_header"]["sender_batch_id"] == "b-2"
    assert seen["body"]["items"][0]["amount"] == {"value": "12.50", "currency": "USD"}
    assert seen["body"]["items"][0]["recipient_type"] == "EMAIL"


def test_error_carries_debug_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        return httpx.Response(
            422, json={"name": "VALIDATION_ERROR"}, headers={"paypal-debug-id": "dbg123"}
        )

    with pytest.raises(PayPalError) as exc:
        make_client(handler).get_payout("B1")
    assert exc.value.status == 422
    assert exc.value.debug_id == "dbg123"


def test_payout_item_rounds_half_up_to_cents():
    assert payout_item("a@b.co", 1.005, sender_item_id="x")["amount"]["value"] == "1.01"
    assert payout_item("a@b.co", 7, sender_item_id="x")["amount"]["value"] == "7.00"


def test_sender_batch_ids_are_distinct_per_label_and_short():
    ids = {make_sender_batch_id(f"ep{i}") for i in range(1000)}
    assert len(ids) == 1000
    assert all(0 < len(i) <= 64 for i in ids)
