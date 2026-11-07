"""Payout rail contract shared by the simulated and live (PayPal sandbox) adapters.

Status vocabularies below are PayPal's own names, taken from the Payouts v1 API reference
(schema: https://developer.paypal.com/api/payments.payouts-batch/v1/schema.yaml, overview:
https://developer.paypal.com/docs/api/payments.payouts-batch/v1/):
  batch_status        PENDING | PROCESSING | SUCCESS | DENIED | CANCELED
  transaction_status  SUCCESS | FAILED | PENDING | UNCLAIMED | RETURNED | ONHOLD | BLOCKED
                      | REFUNDED | REVERSED   (payout item)
Webhook event names (PAYMENT.PAYOUTSBATCH.*, PAYMENT.PAYOUTS-ITEM.*):
  https://developer.paypal.com/api/rest/webhooks/event-names/
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

BATCH_STATUSES = ("PENDING", "PROCESSING", "SUCCESS", "DENIED", "CANCELED")
ITEM_STATUSES = (
    "SUCCESS", "FAILED", "PENDING", "UNCLAIMED", "RETURNED", "ONHOLD", "BLOCKED", "REFUNDED",
    "REVERSED",
)
# Batch statuses PayPal will not move out of, and item statuses that are no longer in flight.
BATCH_TERMINAL = frozenset({"SUCCESS", "DENIED", "CANCELED"})
ITEM_TERMINAL = frozenset(set(ITEM_STATUSES) - {"PENDING"})


def utc_iso(dt: datetime) -> str:
    """ISO-8601 UTC string, e.g. 2026-11-07T12:00:00.000000Z (naive datetimes are taken as UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PayoutRequest:
    event_id: int
    amount_usd: float
    receiver: str
    note: str = ""


@dataclass(frozen=True)
class PayoutReceipt:
    event_id: int
    mode: str  # "sim" | "live"
    sender_batch_id: str
    payout_batch_id: str
    batch_status: str
    item_status: str
    amount_usd: float
    created_at: str  # ISO-8601 UTC
    updated_at: str  # ISO-8601 UTC


class PayoutRail(Protocol):
    mode: str

    def submit(self, req: PayoutRequest) -> PayoutReceipt: ...

    def refresh(self, receipt: PayoutReceipt) -> PayoutReceipt: ...


class RailError(RuntimeError):
    """Base class for rail-level refusals."""


class LiveGuardError(RailError):
    """A live-payout guard refused the request.

    `reason` is one of: amount | cap | budget | allowlist.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
