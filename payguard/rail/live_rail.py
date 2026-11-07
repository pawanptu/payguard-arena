"""Live rail: real payouts to the PayPal SANDBOX through `PayPalClient` (which itself refuses any
non-sandbox host). Every guard below raises `LiveGuardError`; nothing is silently clamped.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..config import optional, require
from ..paypal_client import PayPalClient, PayPalError, make_sender_batch_id, payout_item
from .base import (
    LiveGuardError,
    PayoutReceipt,
    PayoutRequest,
    RailError,
    utc_iso,
    utc_now,
)
from .store import ReceiptStore

DEFAULT_MAX_PAYOUT_USD = "5.00"
DEFAULT_BUDGET_USD = "50.00"
_CENT = Decimal("0.01")


def _cents(x: float | Decimal | str) -> Decimal:
    return Decimal(str(x)).quantize(_CENT, rounding=ROUND_HALF_UP)


def _norm_email(s: str) -> str:
    return s.strip().lower()


class LiveRail:
    mode = "live"

    def __init__(
        self,
        client: PayPalClient | Any,
        *,
        max_payout_usd: float | str = DEFAULT_MAX_PAYOUT_USD,
        budget_usd: float | str = DEFAULT_BUDGET_USD,
        allowed_receivers: set[str] | frozenset[str] | list[str] | tuple[str, ...],
        clock: Callable[[], datetime] = utc_now,
        store: ReceiptStore | None = None,
    ):
        self._client = client
        self.max_payout = _cents(max_payout_usd)
        self.budget = _cents(budget_usd)
        self.allowed = frozenset(_norm_email(a) for a in allowed_receivers if a and a.strip())
        if self.max_payout <= 0 or self.budget <= 0:
            raise ValueError("max_payout_usd and budget_usd must be positive")
        if not self.allowed:
            raise ValueError("LiveRail needs a non-empty receiver allowlist")
        self._clock = clock
        self._store = store
        self._lock = threading.Lock()
        self._spent = Decimal("0.00")
        self._issued: dict[int, PayoutReceipt] = {}

    @classmethod
    def from_env(cls, *, store: ReceiptStore | None = None) -> "LiveRail":
        """Build from PAYPAL_CLIENT_ID/SECRET/BASE_URL, PAYPAL_TEST_RECEIVER_EMAIL and the
        PAYGUARD_LIVE_* guard variables. Raises MissingConfig (names only) if credentials are unset."""
        env = require("PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET", "PAYPAL_TEST_RECEIVER_EMAIL")
        client = PayPalClient(
            env["PAYPAL_CLIENT_ID"],
            env["PAYPAL_CLIENT_SECRET"],
            base_url=optional("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com"),
        )
        extra = [e for e in optional("PAYGUARD_LIVE_RECEIVER_ALLOWLIST", "").split(",") if e.strip()]
        return cls(
            client,
            max_payout_usd=optional("PAYGUARD_LIVE_MAX_PAYOUT_USD", DEFAULT_MAX_PAYOUT_USD),
            budget_usd=optional("PAYGUARD_LIVE_BUDGET_USD", DEFAULT_BUDGET_USD),
            allowed_receivers={env["PAYPAL_TEST_RECEIVER_EMAIL"], *extra},
            store=store,
        )

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return float(self._spent)

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return float(self.budget - self._spent)

    def submit(self, req: PayoutRequest) -> PayoutReceipt:
        # Serialized on purpose: live payouts are a small sample, and this makes the budget
        # check + reserve + idempotent replay trivially race-free.
        with self._lock:
            existing = self._issued.get(req.event_id)
            if existing is not None:
                return existing  # same event never pays twice (and never re-spends budget)

            if isinstance(req.amount_usd, bool) or not math.isfinite(float(req.amount_usd)):
                raise LiveGuardError("amount", f"invalid payout amount {req.amount_usd!r}")
            amount = _cents(req.amount_usd)
            if amount <= 0:
                raise LiveGuardError("amount", f"payout amount must be positive, got {amount}")
            receiver = _norm_email(req.receiver)
            if receiver not in self.allowed:
                raise LiveGuardError("allowlist", "receiver is not in the live-payout allowlist")
            if amount > self.max_payout:
                raise LiveGuardError(
                    "cap", f"payout {amount} USD exceeds per-payout cap {self.max_payout} USD"
                )
            if self._spent + amount > self.budget:
                raise LiveGuardError(
                    "budget",
                    f"payout {amount} USD would exceed the live budget {self.budget} USD "
                    f"(spent {self._spent})",
                )

            sender_batch_id = make_sender_batch_id(f"ev{req.event_id}")
            item = payout_item(
                receiver,
                amount,
                sender_item_id=f"item-{req.event_id}",
                note=(req.note or "PayGuard Arena sandbox payout")[:200],
            )
            self._spent += amount  # reserve first; released only if PayPal definitively refused
            try:
                body = self._client.create_payout(sender_batch_id=sender_batch_id, items=[item])
            except PayPalError as exc:
                if exc.status < 500:  # rejected outright -> nothing was paid
                    self._spent -= amount
                raise  # 5xx is ambiguous, so the reservation stays (as for timeouts etc.)

            header = (body or {}).get("batch_header") or {}
            payout_batch_id = header.get("payout_batch_id")
            if not payout_batch_id:
                raise RailError("PayPal create_payout response had no payout_batch_id")
            now = utc_iso(self._clock())
            receipt = PayoutReceipt(
                event_id=req.event_id,
                mode=self.mode,
                sender_batch_id=sender_batch_id,
                payout_batch_id=payout_batch_id,
                batch_status=header.get("batch_status") or "PENDING",
                item_status="PENDING",
                amount_usd=float(amount),
                created_at=now,
                updated_at=now,
            )
            self._issued[req.event_id] = receipt
        if self._store is not None:
            self._store.upsert(receipt)
        return receipt

    def refresh(self, receipt: PayoutReceipt) -> PayoutReceipt:
        """Poll PayPal for the batch and its first item (one item per batch here)."""
        body = self._client.get_payout(receipt.payout_batch_id)
        header = body.get("batch_header") or {}
        items = body.get("items") or []
        item_status = receipt.item_status
        if items:
            item_status = items[0].get("transaction_status") or item_status
        batch_status = header.get("batch_status") or receipt.batch_status
        if (batch_status, item_status) == (receipt.batch_status, receipt.item_status):
            return receipt
        updated = replace(
            receipt,
            batch_status=batch_status,
            item_status=item_status,
            updated_at=utc_iso(self._clock()),
        )
        if self._store is not None:
            self._store.upsert(updated)
        return updated
