"""Thread-safe, bounded in-memory receipt store (re-exported from `payguard.rail.webhook`)."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import replace

from .base import BATCH_TERMINAL, ITEM_TERMINAL, PayoutReceipt, utc_iso, utc_now


class ReceiptStore:
    """Receipts keyed by event_id; evicts the least recently updated beyond `max_receipts`.

    Also indexes receipts by PayPal batch ids (webhooks carry no event_id, so this is how they
    find their receipt), remembers recent webhook event ids for dedupe, and parks status updates
    that arrive before the receipt is stored (a webhook racing the submit response).
    """

    def __init__(self, max_receipts: int = 10_000, max_seen: int = 20_000, max_orphans: int = 1_000):
        self._lock = threading.RLock()
        self._receipts: OrderedDict[int, PayoutReceipt] = OrderedDict()
        self._by_batch: dict[str, int] = {}
        self._by_sender: dict[str, int] = {}
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._orphans: OrderedDict[str, dict] = OrderedDict()
        self._max_receipts = max_receipts
        self._max_seen = max_seen
        self._max_orphans = max_orphans

    def __len__(self) -> int:
        with self._lock:
            return len(self._receipts)

    # ---- receipts -------------------------------------------------------------------------
    def upsert(self, receipt: PayoutReceipt) -> PayoutReceipt:
        with self._lock:
            old = self._receipts.pop(receipt.event_id, None)
            if old is not None:
                self._by_batch.pop(old.payout_batch_id, None)
                self._by_sender.pop(old.sender_batch_id, None)
            # A webhook that arrived before this receipt is applied now.
            for key in (receipt.payout_batch_id, receipt.sender_batch_id):
                pending = self._orphans.pop(key, None)
                if pending:
                    receipt = _apply(receipt, **pending)
            self._receipts[receipt.event_id] = receipt  # most recent last
            self._by_batch[receipt.payout_batch_id] = receipt.event_id
            self._by_sender[receipt.sender_batch_id] = receipt.event_id
            while len(self._receipts) > self._max_receipts:
                _, evicted = self._receipts.popitem(last=False)
                self._by_batch.pop(evicted.payout_batch_id, None)
                self._by_sender.pop(evicted.sender_batch_id, None)
            return receipt

    def get(self, event_id: int) -> PayoutReceipt | None:
        with self._lock:
            return self._receipts.get(event_id)

    def recent(self, n: int = 20) -> list[PayoutReceipt]:
        """Up to `n` receipts, most recently updated first."""
        with self._lock:
            return list(reversed(self._receipts.values()))[: max(0, n)]

    def find(
        self, *, payout_batch_id: str | None = None, sender_batch_id: str | None = None
    ) -> PayoutReceipt | None:
        with self._lock:
            ev = self._by_batch.get(payout_batch_id or "")
            if ev is None:
                ev = self._by_sender.get(sender_batch_id or "")
            return self._receipts.get(ev) if ev is not None else None

    # ---- webhook helpers ------------------------------------------------------------------
    def mark_event_seen(self, webhook_event_id: str) -> bool:
        """Record a webhook event id. True if new, False if it is a duplicate."""
        with self._lock:
            if webhook_event_id in self._seen:
                return False
            self._seen[webhook_event_id] = None
            while len(self._seen) > self._max_seen:
                self._seen.popitem(last=False)
            return True

    def apply_status(
        self,
        *,
        payout_batch_id: str | None,
        sender_batch_id: str | None,
        batch_status: str | None = None,
        item_status: str | None = None,
    ) -> PayoutReceipt | None:
        """Apply a webhook-derived status. Returns the receipt, or None when none matches yet
        (the update is parked and applied if the receipt is stored later)."""
        with self._lock:
            rec = self.find(payout_batch_id=payout_batch_id, sender_batch_id=sender_batch_id)
            if rec is None:
                key = payout_batch_id or sender_batch_id
                if key:
                    cur = self._orphans.pop(key, {})
                    if batch_status:
                        cur["batch_status"] = batch_status
                    if item_status:
                        cur["item_status"] = item_status
                    self._orphans[key] = cur
                    while len(self._orphans) > self._max_orphans:
                        self._orphans.popitem(last=False)
                return None
            new = _apply(rec, batch_status=batch_status, item_status=item_status)
            if new is not rec:
                self._receipts[rec.event_id] = new
                self._receipts.move_to_end(rec.event_id)
            return new


def _apply(
    rec: PayoutReceipt, *, batch_status: str | None = None, item_status: str | None = None
) -> PayoutReceipt:
    """Never move backwards: a terminal batch status is final, and a settled item never returns
    to PENDING (PayPal retries webhooks, so deliveries can arrive out of order)."""
    changes: dict[str, str] = {}
    if batch_status and batch_status != rec.batch_status and rec.batch_status not in BATCH_TERMINAL:
        if not (batch_status == "PENDING" and rec.batch_status == "PROCESSING"):
            changes["batch_status"] = batch_status
    if item_status and item_status != rec.item_status:
        if not (item_status == "PENDING" and rec.item_status in ITEM_TERMINAL):
            changes["item_status"] = item_status
    if not changes:
        return rec
    return replace(rec, updated_at=utc_iso(utc_now()), **changes)
