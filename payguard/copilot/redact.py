"""Recursive redaction of tool data before it reaches the LLM or a recorded session.

Rule: a key whose normalised name contains a personal-data token has its whole value replaced by "[redacted]"
(so the model still sees that the field existed). Any string value is scanned for e-mail addresses. Ids,
statuses, amounts, currency codes and dates pass through. HATEOAS `links` are dropped (noise, not evidence).
"""
from __future__ import annotations

import re
from typing import Any

REDACTED = "[redacted]"
EMAIL_MASK = "[redacted-email]"

_EMAIL = re.compile(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

# Substring tokens on the lower-cased key. Covers: email, email_address, payer_email, name (exact) and *_name,
# given_name, surname, full_name, alternate_full_name, address, address_line_1, postal_code, phone, phone_number,
# payer_id, account_id, payer_info, shipping, shipping_info, receiver, recipient (the payout item's receiver).
_TOKENS = (
    "email", "surname", "given_name", "full_name", "address", "postal", "phone",
    "payer_id", "account_id", "account_number", "payer_info", "shipping", "receiver", "recipient",
)
# Keys that match a token but carry no personal data (enums, counts, ids of the batch itself).
_KEEP = {"recipient_type", "email_subject", "receiver_type", "receiver_count"}


def _sensitive_key(key: str) -> bool:
    k = key.strip().lower()
    if k in _KEEP:
        return False
    if k == "name" or k.endswith("_name"):
        return True
    return any(t in k for t in _TOKENS)


def redact(value: Any) -> Any:
    """Return a redacted deep copy of `value` (dicts, lists, strings; everything else passes through)."""
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            if str(k).lower() == "links":
                continue
            out[k] = REDACTED if _sensitive_key(str(k)) else redact(v)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _EMAIL.sub(EMAIL_MASK, value)
    return value
