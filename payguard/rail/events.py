"""Bounded, thread-safe log of VERIFIED PayPal payouts webhook events.

An event is recorded only after its signature verified, it passed the duplicate check and it is a
handled payouts event that references a batch (see `payguard.rail.webhook`). The log keeps ids and
statuses only: never the raw resource, receiver emails, names or any other personal data.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass

from .base import utc_iso, utc_now

DEFAULT_MAX_EVENTS = 200
_MAX_FIELD_LEN = 128
OUTCOMES = ("processed", "pending_receipt")


def _text(v: object) -> str | None:
    """A short string, or None. Anything that is not a plain string (dicts, lists, numbers) is dropped."""
    return v[:_MAX_FIELD_LEN] if isinstance(v, str) and v else None


@dataclass(frozen=True)
class WebhookEventRecord:
    event_id: str
    event_type: str
    payout_batch_id: str | None
    sender_batch_id: str | None
    batch_status: str | None
    item_status: str | None
    received_at: str  # ISO-8601 UTC
    outcome: str  # "processed" | "pending_receipt"


class WebhookEventLog:
    def __init__(self, max_events: int = DEFAULT_MAX_EVENTS):
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._lock = threading.Lock()
        self._events: deque[WebhookEventRecord] = deque(maxlen=max_events)  # oldest dropped first

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def record(
        self,
        *,
        event_id: str,
        event_type: str,
        payout_batch_id: object = None,
        sender_batch_id: object = None,
        batch_status: object = None,
        item_status: object = None,
        outcome: str,
    ) -> WebhookEventRecord:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        rec = WebhookEventRecord(
            event_id=event_id[:_MAX_FIELD_LEN],
            event_type=event_type[:_MAX_FIELD_LEN],
            payout_batch_id=_text(payout_batch_id),
            sender_batch_id=_text(sender_batch_id),
            batch_status=_text(batch_status),
            item_status=_text(item_status),
            received_at=utc_iso(utc_now()),
            outcome=outcome,
        )
        with self._lock:
            self._events.append(rec)
        return rec

    def recent(self, n: int = 50) -> list[dict]:
        """Up to `n` events as plain dicts, newest first."""
        with self._lock:
            items = list(self._events)
        return [asdict(e) for e in reversed(items)][: max(0, n)]
