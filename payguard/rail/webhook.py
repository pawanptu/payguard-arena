"""PayPal webhook receiver: POST /webhooks/paypal.

Every event is verified with PayPal's verify-webhook-signature API before anything is accepted:
  POST /v1/notifications/verify-webhook-signature
  https://developer.paypal.com/api/rest/webhooks/rest/
  https://developer.paypal.com/api/webhooks/v1/schema.yaml
Event names: https://developer.paypal.com/api/rest/webhooks/event-names/
Payout statuses: https://developer.paypal.com/api/payments.payouts-batch/v1/schema.yaml

The app owner wires this up by including `router` and, if desired, overriding the three
dependencies (`get_receipt_store`, `get_webhook_id`, `get_paypal_client`) via
`app.dependency_overrides`.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ..config import MissingConfig, optional, require
from ..paypal_client import PayPalClient
from .base import BATCH_STATUSES, ITEM_STATUSES
from .store import ReceiptStore

__all__ = ["router", "ReceiptStore", "get_receipt_store", "get_webhook_id", "get_paypal_client"]

log = logging.getLogger("payguard.rail.webhook")
router = APIRouter()

MAX_BODY_BYTES = 256 * 1024

# PAYMENT.PAYOUTSBATCH.* -> batch_status  (PayPal only emits these three)
BATCH_EVENTS = {
    "PAYMENT.PAYOUTSBATCH.DENIED": "DENIED",
    "PAYMENT.PAYOUTSBATCH.PROCESSING": "PROCESSING",
    "PAYMENT.PAYOUTSBATCH.SUCCESS": "SUCCESS",
}
# PAYMENT.PAYOUTS-ITEM.* -> item transaction_status. CANCELED maps to RETURNED because the
# reference says RETURNED covers funds returned "because the payout went unclaimed or was canceled".
ITEM_EVENTS = {
    "PAYMENT.PAYOUTS-ITEM.SUCCEEDED": "SUCCESS",
    "PAYMENT.PAYOUTS-ITEM.FAILED": "FAILED",
    "PAYMENT.PAYOUTS-ITEM.BLOCKED": "BLOCKED",
    "PAYMENT.PAYOUTS-ITEM.CANCELED": "RETURNED",
    "PAYMENT.PAYOUTS-ITEM.HELD": "ONHOLD",
    "PAYMENT.PAYOUTS-ITEM.REFUNDED": "REFUNDED",
    "PAYMENT.PAYOUTS-ITEM.RETURNED": "RETURNED",
    "PAYMENT.PAYOUTS-ITEM.UNCLAIMED": "UNCLAIMED",
}

_SIG_HEADERS = {
    "auth_algo": "paypal-auth-algo",
    "cert_url": "paypal-cert-url",
    "transmission_id": "paypal-transmission-id",
    "transmission_sig": "paypal-transmission-sig",
    "transmission_time": "paypal-transmission-time",
}

# ---- dependencies (override with app.dependency_overrides) ------------------------------------
_store = ReceiptStore()
_client: PayPalClient | None = None
_client_lock = threading.Lock()


def get_receipt_store() -> ReceiptStore:
    return _store


def get_webhook_id() -> str | None:
    """PAYPAL_WEBHOOK_ID from the environment; None (-> HTTP 503) when not configured."""
    return optional("PAYPAL_WEBHOOK_ID", "") or None


def get_paypal_client() -> PayPalClient | None:
    """Lazily built sandbox client; None (-> HTTP 503) when credentials are not configured."""
    global _client
    with _client_lock:
        if _client is None:
            try:
                env = require("PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET")
            except MissingConfig:
                return None
            _client = PayPalClient(env["PAYPAL_CLIENT_ID"], env["PAYPAL_CLIENT_SECRET"])
        return _client


# ---- handler ---------------------------------------------------------------------------------
def _resp(code: int, status: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"status": status, **extra}, status_code=code)


def _extract(event_type: str, resource: dict) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (payout_batch_id, sender_batch_id, batch_status, item_status) for a payouts event.

    Field names follow PayPal's payout sample payloads (batch events nest under `batch_header`;
    item events carry payout_batch_id / sender_batch_id / transaction_status at top level, with
    the original item under `payout_item`). Extraction is deliberately tolerant.
    """
    header = resource.get("batch_header") if isinstance(resource.get("batch_header"), dict) else {}
    sender_hdr = header.get("sender_batch_header") if isinstance(header.get("sender_batch_header"), dict) else {}
    payout_batch_id = resource.get("payout_batch_id") or header.get("payout_batch_id")
    sender_batch_id = resource.get("sender_batch_id") or sender_hdr.get("sender_batch_id")
    batch_status = item_status = None
    if event_type in BATCH_EVENTS:
        reported = header.get("batch_status") or resource.get("batch_status")
        batch_status = reported if reported in BATCH_STATUSES else BATCH_EVENTS[event_type]
    elif event_type in ITEM_EVENTS:
        reported = resource.get("transaction_status")
        item_status = reported if reported in ITEM_STATUSES else ITEM_EVENTS[event_type]
    return payout_batch_id, sender_batch_id, batch_status, item_status


@router.post("/webhooks/paypal")
async def paypal_webhook(
    request: Request,
    store: ReceiptStore = Depends(get_receipt_store),
    webhook_id: str | None = Depends(get_webhook_id),
    client: PayPalClient | None = Depends(get_paypal_client),
) -> JSONResponse:
    if not webhook_id:
        return _resp(503, "webhook_not_configured")  # accept nothing without a webhook id
    if client is None:
        return _resp(503, "verifier_not_configured")

    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return _resp(413, "too_large")
    try:
        event = json.loads(raw)
    except ValueError:
        return _resp(400, "invalid_json")
    if not isinstance(event, dict):
        return _resp(400, "invalid_json")

    sig = {k: request.headers.get(h) for k, h in _SIG_HEADERS.items()}
    if not all(sig.values()):
        return _resp(400, "missing_signature_headers")

    try:
        ok = await run_in_threadpool(
            client.verify_webhook_signature, webhook_id=webhook_id, webhook_event=event, **sig
        )
    except Exception as exc:  # PayPal/network failure: cannot verify, so accept nothing; PayPal retries
        log.warning("webhook verification call failed: %s", type(exc).__name__)
        return _resp(503, "verification_unavailable")
    if not ok:
        return _resp(401, "signature_invalid")

    # ---- authentic from here on ----
    event_id = event.get("id")
    event_type = event.get("event_type")
    if not isinstance(event_id, str) or not event_id or not isinstance(event_type, str):
        return _resp(200, "ignored", reason="malformed_event")
    if not store.mark_event_seen(event_id):
        return _resp(200, "duplicate")
    if event_type not in BATCH_EVENTS and event_type not in ITEM_EVENTS:
        return _resp(200, "ignored", reason="unhandled_event_type")
    resource = event.get("resource")
    if not isinstance(resource, dict):
        return _resp(200, "ignored", reason="malformed_event")

    payout_batch_id, sender_batch_id, batch_status, item_status = _extract(event_type, resource)
    if not (payout_batch_id or sender_batch_id):
        return _resp(200, "ignored", reason="no_batch_reference")
    receipt = store.apply_status(
        payout_batch_id=payout_batch_id,
        sender_batch_id=sender_batch_id,
        batch_status=batch_status,
        item_status=item_status,
    )
    if receipt is None:
        return _resp(200, "pending_receipt")  # parked; applied if the receipt is stored later
    return _resp(200, "processed", event_id=receipt.event_id)
