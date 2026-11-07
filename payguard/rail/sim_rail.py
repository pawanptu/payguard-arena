"""Deterministic in-memory payout rail. Mimics the lifecycle seen on the real sandbox on Day 0:
PENDING -> PROCESSING -> SUCCESS in about 20 s. No network, no randomness; time comes from an
injectable clock so tests (and replays) never sleep."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from .base import (
    BATCH_TERMINAL,
    PayoutReceipt,
    PayoutRequest,
    parse_iso,
    utc_iso,
    utc_now,
)
from .store import ReceiptStore

SIM_PREFIX = "SIM-"  # real PayPal ids never start with this


class SimRail:
    mode = "sim"

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = utc_now,
        processing_after_s: float = 3.0,
        success_after_s: float = 20.0,
        store: ReceiptStore | None = None,
    ):
        if not 0 <= processing_after_s <= success_after_s:
            raise ValueError("need 0 <= processing_after_s <= success_after_s")
        self._clock = clock
        self._processing_after = processing_after_s
        self._success_after = success_after_s
        self._store = store
        self._lock = threading.Lock()
        self._issued: dict[int, PayoutReceipt] = {}

    def submit(self, req: PayoutRequest) -> PayoutReceipt:
        """Idempotent per event_id, like the live rail: a repeat returns the first receipt."""
        with self._lock:
            existing = self._issued.get(req.event_id)
            if existing is not None:
                return existing
            now = utc_iso(self._clock())
            receipt = PayoutReceipt(
                event_id=req.event_id,
                mode=self.mode,
                sender_batch_id=f"{SIM_PREFIX}SB-{req.event_id}",
                payout_batch_id=f"{SIM_PREFIX}PB-{req.event_id}",
                batch_status="PENDING",
                item_status="PENDING",
                amount_usd=float(req.amount_usd),
                created_at=now,
                updated_at=now,
            )
            self._issued[req.event_id] = receipt
        if self._store is not None:
            self._store.upsert(receipt)
        return receipt

    def refresh(self, receipt: PayoutReceipt) -> PayoutReceipt:
        """Advance the receipt to what its age implies. Settled receipts are returned as is."""
        if receipt.batch_status in BATCH_TERMINAL:
            return receipt
        now = self._clock()
        age = (now - parse_iso(receipt.created_at)).total_seconds()
        if age >= self._success_after:
            batch, item = "SUCCESS", "SUCCESS"
        elif age >= self._processing_after:
            batch, item = "PROCESSING", "PENDING"
        else:
            batch, item = "PENDING", "PENDING"
        if (batch, item) == (receipt.batch_status, receipt.item_status):
            return receipt
        updated = replace(receipt, batch_status=batch, item_status=item, updated_at=utc_iso(now))
        if self._store is not None:
            self._store.upsert(updated)
        return updated
