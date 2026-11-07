"""Thin PayPal REST client: OAuth + Payouts. SANDBOX ONLY by construction.

Endpoints used (PayPal Payouts v1):
  POST /v1/oauth2/token              client-credentials token
  POST /v1/payments/payouts          create a payout batch
  GET  /v1/payments/payouts/{id}     batch + item status
"""
from __future__ import annotations

import os
import re
import time
import uuid
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import urlparse

import httpx

SANDBOX_BASE = "https://api-m.sandbox.paypal.com"
_ALLOWED_HOSTS = {"api-m.sandbox.paypal.com"}


class PayPalError(RuntimeError):
    """HTTP error from PayPal. Carries the debug id PayPal support asks for."""

    def __init__(self, status: int, body: str, debug_id: str | None):
        super().__init__(f"PayPal HTTP {status} (debug_id={debug_id}): {body[:500]}")
        self.status = status
        self.body = body
        self.debug_id = debug_id


# One id per process run (override with PAYGUARD_RUN_ID for a reproducible run). PayPal requires
# `sender_batch_id` to be unique per sender for 30 days, so separate runs must never collide.
_RUN_ID = os.environ.get("PAYGUARD_RUN_ID") or uuid.uuid4().hex[:8]
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def make_sender_batch_id(label: str = "") -> str:
    """Return the `sender_batch_id` for one payout batch (always under 64 chars).

    Same `label` within one run -> same id, so a retry of that batch is rejected by PayPal as a
    duplicate instead of paying twice. Different runs get different prefixes, so they never
    collide. With no label the id is random (never idempotent).
    """
    if not label:
        return f"pg-{_RUN_ID}-{uuid.uuid4().hex[:12]}"
    return f"pg-{_RUN_ID}-{_UNSAFE.sub('-', label)[:40]}"


def payout_item(
    receiver_email: str,
    amount: Decimal | float | int | str,
    *,
    sender_item_id: str,
    currency: str = "USD",
    note: str = "PayGuard Arena sandbox payout",
) -> dict:
    """Build one Payouts item. Amounts are rounded half-up to cents, sent as strings."""
    value = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {
        "recipient_type": "EMAIL",
        "receiver": receiver_email,
        "amount": {"value": f"{value}", "currency": currency},
        "note": note,
        "sender_item_id": sender_item_id,
    }


class PayPalClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        base_url: str = SANDBOX_BASE,
        http: httpx.Client | None = None,
    ):
        host = urlparse(base_url).hostname
        if host not in _ALLOWED_HOSTS:
            raise ValueError(
                f"Refusing non-sandbox PayPal host {host!r}; this project is sandbox only."
            )
        self._id = client_id
        self._secret = client_secret
        self._http = http or httpx.Client(base_url=base_url, timeout=20.0)
        self._token: str | None = None
        self._expires_at = 0.0

    @staticmethod
    def _check(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            raise PayPalError(resp.status_code, resp.text, resp.headers.get("paypal-debug-id"))

    def _access_token(self) -> str:
        # Sandbox tokens last hours; refresh a minute early.
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        resp = self._http.post(
            "/v1/oauth2/token",
            auth=(self._id, self._secret),
            data={"grant_type": "client_credentials"},
            headers={"Accept": "application/json"},
        )
        self._check(resp)
        body = resp.json()
        self._token = body["access_token"]
        self._expires_at = time.time() + int(body.get("expires_in", 0))
        return self._token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/json",
        }

    def create_payout(
        self,
        *,
        sender_batch_id: str,
        items: list[dict],
        email_subject: str = "PayGuard Arena sandbox payout",
    ) -> dict:
        body = {
            "sender_batch_header": {
                "sender_batch_id": sender_batch_id,
                "email_subject": email_subject,
            },
            "items": items,
        }
        resp = self._http.post("/v1/payments/payouts", json=body, headers=self._headers())
        self._check(resp)
        return resp.json()

    def get_payout(self, payout_batch_id: str) -> dict:
        resp = self._http.get(
            f"/v1/payments/payouts/{payout_batch_id}", headers=self._headers()
        )
        self._check(resp)
        return resp.json()
