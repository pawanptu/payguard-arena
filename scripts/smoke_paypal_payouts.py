"""Day-0 gate: does a real sandbox Payouts batch work from this machine?

Run from the repo root:  python -m scripts.smoke_paypal_payouts

Exit codes: 0 = batch accepted and reached SUCCESS, 1 = PayPal rejected it,
2 = config missing, 3 = accepted but still not final after polling (inconclusive).
Never prints the token or the secret.
"""
from __future__ import annotations

import sys
import time

from payguard.config import MissingConfig, optional, require
from payguard.paypal_client import (
    SANDBOX_BASE,
    PayPalClient,
    PayPalError,
    make_sender_batch_id,
    payout_item,
)

TERMINAL = {"SUCCESS", "DENIED", "CANCELED"}


def main() -> int:
    try:
        cfg = require("PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET", "PAYPAL_TEST_RECEIVER_EMAIL")
    except MissingConfig as exc:
        print(f"FAIL config: {exc}")
        return 2

    client = PayPalClient(
        cfg["PAYPAL_CLIENT_ID"],
        cfg["PAYPAL_CLIENT_SECRET"],
        base_url=optional("PAYPAL_BASE_URL", SANDBOX_BASE),
    )
    batch_key = make_sender_batch_id("smoke")
    item = payout_item(
        cfg["PAYPAL_TEST_RECEIVER_EMAIL"], "1.00", sender_item_id=f"{batch_key}-1"
    )

    try:
        created = client.create_payout(sender_batch_id=batch_key, items=[item])
    except PayPalError as exc:
        print(f"FAIL create_payout: HTTP {exc.status} debug_id={exc.debug_id}\n{exc.body}")
        return 1

    header = created.get("batch_header", {})
    batch_id = header.get("payout_batch_id")
    print(f"ACCEPTED payout_batch_id={batch_id} batch_status={header.get('batch_status')}")

    status = header.get("batch_status")
    detail: dict = {}
    for _ in range(10):
        if status in TERMINAL:
            break
        time.sleep(3)
        try:
            detail = client.get_payout(batch_id)
        except PayPalError as exc:
            print(f"FAIL get_payout: HTTP {exc.status} debug_id={exc.debug_id}\n{exc.body}")
            return 1
        status = detail.get("batch_header", {}).get("batch_status")
        print(f"  poll batch_status={status}")

    for it in detail.get("items", []):
        err = it.get("errors")
        print(f"  item transaction_status={it.get('transaction_status')} errors={err}")

    if status == "SUCCESS":
        print("PASS: Payouts batch reached SUCCESS in the sandbox.")
        return 0
    if status in TERMINAL:
        print(f"FAIL: batch ended as {status}. Kill criterion applies (see docs/IDEATION.md s8).")
        return 1
    print("INCONCLUSIVE: batch accepted but not final yet; re-check with get_payout later.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
