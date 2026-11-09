"""Redaction of tool data (hand-written fixtures in the shape of PayPal's documented responses; no network)."""
import json

from payguard.copilot.redact import EMAIL_MASK, REDACTED, redact

PAYOUT_BATCH = {
    "batch_header": {
        "payout_batch_id": "U2UPGB9BXLFRY",
        "batch_status": "SUCCESS",
        "time_created": "2026-11-08T02:58:53Z",
        "time_completed": "2026-11-08T02:58:57Z",
        "sender_batch_header": {
            "sender_batch_id": "pg-e2cdff20-ev1823",
            "email_subject": "PayGuard Arena sandbox payout",
            "recipient_type": "EMAIL",
        },
        "amount": {"currency": "USD", "value": "1.00"},
        "fees": {"currency": "USD", "value": "0.00"},
    },
    "items": [
        {
            "payout_item_id": "7WQJ5TGBTK8KJ",
            "transaction_id": "2A304751AM7442345",
            "transaction_status": "SUCCESS",
            "payout_item_fee": {"currency": "USD", "value": "0.00"},
            "payout_batch_id": "U2UPGB9BXLFRY",
            "payout_item": {
                "recipient_type": "EMAIL",
                "amount": {"value": "1.00", "currency": "USD"},
                "note": "PayGuard Arena sandbox payout. Contact jane.doe@example.com for questions",
                "receiver": "sb-buyer123@personal.example.com",
                "sender_item_id": "ev1823",
                "recipient_wallet": "PAYPAL",
            },
            "time_processed": "2026-11-08T02:58:56Z",
            "links": [{"href": "https://api-m.sandbox.paypal.com/v1/payments/payouts-item/7WQJ5TGBTK8KJ", "rel": "item"}],
        }
    ],
    "links": [{"href": "https://api-m.sandbox.paypal.com/v1/payments/payouts/U2UPGB9BXLFRY", "rel": "self"}],
}

TRANSACTIONS = {
    "transaction_details": [
        {
            "transaction_info": {
                "paypal_account_id": "XYZ123ABC456D",
                "transaction_id": "5TY05013RG002845M",
                "transaction_event_code": "T0001",
                "transaction_initiation_date": "2026-11-08T02:58:54+0000",
                "transaction_amount": {"currency_code": "USD", "value": "-1.00"},
                "transaction_status": "S",
            },
            "payer_info": {
                "account_id": "XYZ123ABC456D",
                "email_address": "buyer@example.com",
                "address_status": "Y",
                "payer_name": {"given_name": "Jane", "surname": "Doe", "alternate_full_name": "Jane Doe"},
                "country_code": "US",
            },
            "shipping_info": {
                "name": "Jane Doe",
                "address": {"line1": "1 Main St", "city": "San Jose", "postal_code": "95131", "country_code": "US"},
            },
            "cart_info": {"item_details": [{"item_name": "Widget", "item_quantity": "1"}]},
        }
    ],
    "account_number": "ABCDEFGHIJKLM",
    "total_items": 1,
    "page": 1,
}


def test_payout_batch_keeps_ids_statuses_amounts_dates():
    out = redact(PAYOUT_BATCH)
    assert out["batch_header"]["payout_batch_id"] == "U2UPGB9BXLFRY"
    assert out["batch_header"]["batch_status"] == "SUCCESS"
    assert out["batch_header"]["amount"] == {"currency": "USD", "value": "1.00"}
    assert out["batch_header"]["time_created"] == "2026-11-08T02:58:53Z"
    item = out["items"][0]
    assert item["transaction_id"] == "2A304751AM7442345"
    assert item["transaction_status"] == "SUCCESS"
    assert item["payout_item"]["amount"] == {"value": "1.00", "currency": "USD"}
    assert item["payout_item"]["sender_item_id"] == "ev1823"
    assert item["payout_item"]["recipient_type"] == "EMAIL"  # an enum, not personal data
    assert out["batch_header"]["sender_batch_header"]["email_subject"] == "PayGuard Arena sandbox payout"


def test_payout_batch_masks_receiver_and_emails_in_free_text_and_drops_links():
    out = redact(PAYOUT_BATCH)
    item = out["items"][0]["payout_item"]
    assert item["receiver"] == REDACTED
    assert EMAIL_MASK in item["note"] and "jane.doe@example.com" not in item["note"]
    assert "links" not in out and "links" not in out["items"][0]
    blob = json.dumps(out)
    assert "@" not in blob.replace(EMAIL_MASK, "")


def test_transaction_search_masks_personal_blocks_keeps_money():
    out = redact(TRANSACTIONS)
    d = out["transaction_details"][0]
    assert d["payer_info"] == REDACTED
    assert d["shipping_info"] == REDACTED
    assert d["transaction_info"]["paypal_account_id"] == REDACTED
    assert d["transaction_info"]["transaction_id"] == "5TY05013RG002845M"
    assert d["transaction_info"]["transaction_amount"] == {"currency_code": "USD", "value": "-1.00"}
    assert d["transaction_info"]["transaction_status"] == "S"
    assert d["transaction_info"]["transaction_initiation_date"] == "2026-11-08T02:58:54+0000"
    assert out["account_number"] == REDACTED
    assert out["total_items"] == 1
    blob = json.dumps(out)
    for secret in ("Jane", "Doe", "buyer@example.com", "1 Main St", "95131", "San Jose"):
        assert secret not in blob


def test_individual_key_families():
    data = {"email": "a@b.co", "name": "N", "given_name": "G", "surname": "S", "full_name": "F",
            "address_line_1": "x", "postal_code": "1", "phone_number": {"national_number": "555"},
            "payer_id": "P", "account_id": "A", "payee_name": "Shop", "case_id": "keep", "risk": 0.7}
    out = redact(data)
    assert {k for k, v in out.items() if v == REDACTED} == {
        "email", "name", "given_name", "surname", "full_name", "address_line_1", "postal_code",
        "phone_number", "payer_id", "account_id", "payee_name"}
    assert out["case_id"] == "keep" and out["risk"] == 0.7


def test_email_pattern_in_any_string_including_lists_and_input_not_mutated():
    data = {"memo": "pay x.y+z@mail.example.org now", "tags": ["ok", "bob@x.io"], "n": 3, "flag": True, "none": None}
    snapshot = json.dumps(data)
    out = redact(data)
    assert out["memo"] == f"pay {EMAIL_MASK} now"
    assert out["tags"] == ["ok", EMAIL_MASK]
    assert out["n"] == 3 and out["flag"] is True and out["none"] is None
    assert json.dumps(data) == snapshot
