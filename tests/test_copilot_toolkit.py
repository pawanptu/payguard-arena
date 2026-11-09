"""Agent Toolkit adapter: selection, allowlist, dispatch, 403 handling, Toolkit-bug workarounds. No network:
the real Toolkit code runs against a fake HTTP object."""
import json
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import BaseModel, Field

from payguard.copilot import toolkit_tools as tt
from payguard.copilot.tooling import ToolError

NOW = datetime(2026, 11, 9, 7, 36, 24, tzinfo=timezone.utc)
ENV = {"PAYPAL_CLIENT_ID": "id-test", "PAYPAL_CLIENT_SECRET": "secret-test"}


# ------------------------------------------------------------------ fakes
class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeHTTP:
    """Stands in for `requests`: routes by URL substring, records every call."""

    def __init__(self, routes):
        self.routes = routes
        self.get_urls = []
        self.token_calls = 0

    def post(self, url, **kw):
        assert "/v1/oauth2/token" in url
        self.token_calls += 1
        return FakeResp(200, {"access_token": "tok-SECRET", "expires_in": 3600})

    def get(self, url, **kw):
        self.get_urls.append(url)
        assert kw.get("timeout")  # every request carries a timeout
        for pat, (status, body) in self.routes.items():
            if pat in url:
                return FakeResp(status, body)
        return FakeResp(404, {"name": "RESOURCE_NOT_FOUND"})


def real_dispatcher(http):
    pytest.importorskip("paypal_agent_toolkit")
    return tt.ToolkitDispatcher.from_env(ENV, http=http, now=lambda: NOW)


def query(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query, keep_blank_values=True).items()}


class FakeAPI:
    def __init__(self, result=None, exc=None):
        self.calls = []
        self.result, self.exc = result, exc

    def run(self, method, params):
        self.calls.append((method, params))
        if self.exc:
            raise self.exc
        return self.result


# ------------------------------------------------------------------ selection
class P(BaseModel):
    x: str | None = Field(None, description="an optional thing")
    kind: str = Field("a", description="kind")


def _entry(method, actions):
    return {"method": method, "name": method, "description": f"{method} does things", "args_schema": P,
            "actions": actions, "execute": lambda c, p: {}}


def _fake_is_allowed(tool, cfg):  # same rule as the Toolkit's is_tool_allowed
    return any(cfg.actions.get(prod, {}).get(act, False)
               for prod, acts in tool.get("actions", {}).items() for act in acts)


class Cfg:
    actions = tt.READ_ONLY_ACTIONS


def test_filter_registry_needs_both_toolkit_permission_and_allowlist():
    registry = [
        _entry("list_disputes", {"disputes": {"list": True}}),          # allowed + allowlisted
        _entry("create_order", {"orders": {"create": True}}),           # not permitted
        _entry("pay_order", {"orders": {"capture": True}}),             # not permitted
        _entry("list_products", {"transactions": {"list": True}}),      # permitted action, but not allowlisted
        _entry("get_invoice", {"invoices": {"get": True}}),
        _entry("list_invoices", {"invoices": {"create": True}}),        # allowlisted name, action not permitted
    ]
    got = [t["method"] for t in tt.filter_registry(registry, is_allowed=_fake_is_allowed, configuration=Cfg)]
    assert got == ["list_disputes", "get_invoice"]


def test_real_registry_enables_exactly_the_six_read_only_tools():
    pytest.importorskip("paypal_agent_toolkit")
    enabled = tt.enabled_entries()
    names = {t["method"] for t in enabled}
    assert names == {"list_transactions", "list_disputes", "get_dispute", "list_invoices", "get_invoice",
                     "get_order_details"}
    assert names == set(tt.ALLOWED_METHODS)
    bad = re.compile(r"create|send|capture|accept|cancel|delete|update|pay_|record|activate|setup|generate")
    assert not [n for n in names if bad.search(n)]
    # And the whole registry really contains write tools that we are excluding.
    from paypal_agent_toolkit.shared.tools import tools

    assert {"create_order", "pay_order", "send_invoice", "accept_dispute_claim"} <= {t["method"] for t in tools}


def test_toolkit_tools_from_fake_registry_have_simplified_schema_and_dispatch():
    api = FakeAPI(result=json.dumps({"ok": 1}))
    pytest.importorskip("paypal_agent_toolkit")
    registry = [_entry("get_dispute", {"disputes": {"get": True}}), _entry("create_order", {"orders": {"create": True}})]
    tools = tt.toolkit_tools(tt.ToolkitDispatcher(api), registry)
    assert [t.name for t in tools] == ["get_dispute"]
    t = tools[0]
    assert t.source == "paypal-agent-toolkit"
    assert t.schema["properties"]["x"] == {"type": "string", "description": "an optional thing"}
    assert "title" not in t.schema["properties"]["x"]
    assert t.fn({"x": "1"}) == {"ok": 1}
    assert api.calls == [("get_dispute", {"x": "1"})]


# ------------------------------------------------------------------ environment guard
def test_credentials_from_env_only_and_sandbox_host_only():
    assert tt.sandbox_credentials(ENV) == ("id-test", "secret-test")
    assert tt.sandbox_credentials({**ENV, "PAYPAL_BASE_URL": "https://api-m.sandbox.paypal.com"})
    with pytest.raises(tt.ToolkitConfigError, match="sandbox only"):
        tt.sandbox_credentials({**ENV, "PAYPAL_BASE_URL": "https://api-m.paypal.com"})
    with pytest.raises(tt.ToolkitConfigError) as exc:
        tt.sandbox_credentials({"PAYPAL_CLIENT_ID": "x"})
    assert "PAYPAL_CLIENT_SECRET" in str(exc.value) and "PAYPAL_CLIENT_ID" not in str(exc.value)


def test_install_hint_names_both_commands():
    assert "pip install -r requirements-copilot.txt" in tt.INSTALL_HINT
    assert "pip install --no-deps paypal-agent-toolkit==1.11.0" in tt.INSTALL_HINT


# ------------------------------------------------------------------ dispatcher
@pytest.mark.parametrize("method", ["create_order", "pay_order", "send_invoice", "accept_dispute_claim",
                                    "delete_invoice", "get_merchant_insights", "", "list_products"])
def test_dispatcher_refuses_methods_outside_the_allowlist(method):
    api = FakeAPI(result="{}")
    with pytest.raises(ToolError, match="not an enabled read-only tool"):
        tt.ToolkitDispatcher(api).call(method, {})
    assert api.calls == []  # never reached PayPalAPI.run


def test_dispatcher_normalises_json_string_and_dict_results():
    assert tt.ToolkitDispatcher(FakeAPI(result='{"items": []}')).call("list_disputes", {}) == {"items": []}
    assert tt.ToolkitDispatcher(FakeAPI(result={"items": [1]})).call("list_disputes", {}) == {"items": [1]}
    assert tt.ToolkitDispatcher(FakeAPI(result="not json")).call("list_disputes", {}) == {"text": "not json"}


def test_get_order_details_message_bug_is_dropped():
    raw = json.dumps({"message": "The PayPal order X has been successfully captured.", "status": "CREATED",
                      "amount": "N/A", "raw": {}})
    out = tt.ToolkitDispatcher(FakeAPI(result=raw)).call("get_order_details", {"order_id": "A" * 17})
    assert "message" not in out and out["status"] == "CREATED"


def test_403_becomes_a_clean_tool_error():
    api = FakeAPI(exc=tt.PayPalHTTPError(403, "NOT_AUTHORIZED"))
    with pytest.raises(ToolError) as exc:
        tt.ToolkitDispatcher(api).call("list_invoices", {})
    assert str(exc.value) == "PayPal sandbox: NOT_AUTHORIZED (this app lacks the permission for this API)"


def test_other_errors_are_short_and_safe():
    assert tt.describe_error(tt.PayPalHTTPError(404, "RESOURCE_NOT_FOUND")) == "PayPal sandbox: HTTP 404 RESOURCE_NOT_FOUND"
    assert "401" in tt.describe_error(tt.PayPalHTTPError(401))
    assert tt.describe_error(RuntimeError("Bearer abc.def SECRET")) == "tool failed (RuntimeError)"
    try:
        P.model_validate({"kind": 5})
    except Exception as exc:
        assert tt.describe_error(exc).startswith("Invalid arguments: kind:")


def test_search_months_is_clamped():
    api = FakeAPI(result="{}")
    tt.ToolkitDispatcher(api, now=lambda: NOW).call("list_transactions", {"search_months": 999})
    assert api.calls[0][1]["search_months"] == tt.MAX_SEARCH_MONTHS


# ------------------------------------------------------------------ real Toolkit code against a fake HTTP
def test_real_toolkit_403_on_transactions_and_invoices_is_clean():
    http = FakeHTTP({"/v1/reporting/transactions": (403, {"name": "NOT_AUTHORIZED", "message": "m"}),
                     "/v2/invoicing/invoices": (403, {"name": "NOT_AUTHORIZED"})})
    d = real_dispatcher(http)
    for method in ("list_transactions", "list_invoices"):
        with pytest.raises(ToolError) as exc:
            d.call(method, {})
        assert str(exc.value) == tt.NOT_AUTHORIZED_TEXT


def test_real_toolkit_403_on_search_by_transaction_id_is_not_reported_as_not_found():
    # The Toolkit swallows every error in its search-by-id loop and would answer "not found".
    http = FakeHTTP({"/v1/reporting/transactions": (403, {"name": "NOT_AUTHORIZED"})})
    d = real_dispatcher(http)
    with pytest.raises(ToolError) as exc:
        d.call("list_transactions", {"transaction_id": "5TY05013RG002845M"})
    assert str(exc.value) == tt.NOT_AUTHORIZED_TEXT
    assert len(http.get_urls) == 1  # no pointless repeats after "not authorised"


def test_real_toolkit_list_transactions_workarounds_none_param_and_dates():
    body = {"transaction_details": [{"transaction_info": {
        "transaction_id": "5TY05013RG002845M", "transaction_event_code": "T0001",
        "transaction_initiation_date": "2026-11-08T02:58:54+0000",
        "transaction_amount": {"currency_code": "USD", "value": "-1.00"}, "transaction_status": "S"},
        "payer_info": {"email_address": "x@y.com"}}], "total_items": 1, "page": 1}
    http = FakeHTTP({"/v1/reporting/transactions": (200, body)})
    out = real_dispatcher(http).call("list_transactions", {})
    q = query(http.get_urls[0])
    assert "transaction_id" not in q                       # workaround 1: no transaction_id=None
    assert "None" not in http.get_urls[0]
    assert q["end_date"] == "2026-11-09T07:36:24Z"         # workaround 2: explicit UTC, last 30 days
    assert q["start_date"] == "2026-10-10T07:36:24Z"
    assert out["transactions"][0]["transaction_amount"] == {"currency_code": "USD", "value": "-1.00"}
    assert "payer_info" not in json.dumps(out)             # compacted before it can reach the model
    assert http.token_calls == 1


def test_real_toolkit_search_by_id_still_works_and_sends_the_id():
    tx = {"transaction_info": {"transaction_id": "5TY05013RG002845M", "transaction_status": "S"}}
    http = FakeHTTP({"/v1/reporting/transactions": (200, {"transaction_details": [tx]})})
    out = real_dispatcher(http).call("list_transactions", {"transaction_id": "5TY05013RG002845M"})
    assert query(http.get_urls[0])["transaction_id"] == "5TY05013RG002845M"
    assert out["found"] is True and out["transactions"][0]["transaction_id"] == "5TY05013RG002845M"


def test_real_toolkit_token_is_cached_across_calls():
    http = FakeHTTP({"/v1/customer/disputes": (200, {"items": []})})
    d = real_dispatcher(http)
    d.call("list_disputes", {})
    d.call("list_disputes", {})
    assert http.token_calls == 1 and len(http.get_urls) == 2


def test_read_only_client_cannot_write():
    http = FakeHTTP({})
    d = real_dispatcher(http)
    for verb in ("post", "put", "delete"):
        with pytest.raises(PermissionError):
            getattr(d.client, verb)("/v2/checkout/orders", {})
    assert http.get_urls == []


# ------------------------------------------------------------------ pure helpers
def test_clean_uri_drops_none_and_merges_payload():
    assert tt.clean_uri("/v1/customer/disputes?transaction_id=None&page_size=10") == "/v1/customer/disputes?page_size=10"
    assert tt.clean_uri("/v1/x", {"a": "1", "b": None}) == "/v1/x?a=1"
    assert tt.clean_uri("/v1/reporting/transactions?start_date=2026-10-01T00:00:00&end_date=2026-10-02T00:00:00Z") == (
        "/v1/reporting/transactions?start_date=2026-10-01T00%3A00%3A00Z&end_date=2026-10-02T00%3A00%3A00Z")


def test_prepare_transaction_args_dates():
    f = tt.prepare_transaction_args
    out = f({}, NOW)
    assert (out["start_date"], out["end_date"]) == ("2026-10-10T07:36:24Z", "2026-11-09T07:36:24Z")
    # no timezone -> read as UTC, "Z" appended; fractional seconds dropped
    out = f({"start_date": "2026-10-20T01:02:03", "end_date": "2026-10-25T04:05:06.789"}, NOW)
    assert (out["start_date"], out["end_date"]) == ("2026-10-20T01:02:03Z", "2026-10-25T04:05:06Z")
    # an offset is converted to UTC
    assert f({"start_date": "2026-10-20T01:00:00+05:30", "end_date": "2026-10-21T00:00:00Z"}, NOW)["start_date"] == "2026-10-19T19:30:00Z"
    # date only
    assert f({"start_date": "2026-10-20", "end_date": "2026-10-21"}, NOW)["start_date"] == "2026-10-20T00:00:00Z"
    # only start -> end 30 days later, capped at now
    assert f({"start_date": "2026-10-20T00:00:00Z"}, NOW)["end_date"] == "2026-11-09T07:36:24Z"
    assert f({"start_date": "2026-09-01T00:00:00Z"}, NOW)["end_date"] == "2026-10-01T00:00:00Z"
    # only end -> start 30 days earlier
    assert f({"end_date": "2026-10-31T00:00:00Z"}, NOW)["start_date"] == "2026-10-01T00:00:00Z"
    # more than 31 days -> start moved to end - 31 days
    out = f({"start_date": "2026-01-01T00:00:00Z", "end_date": "2026-11-01T00:00:00Z"}, NOW)
    assert out["start_date"] == "2026-10-01T00:00:00Z"
    # does not mutate its input and keeps other args
    src = {"page_size": 5}
    assert f(src, NOW)["page_size"] == 5 and src == {"page_size": 5}


def test_prepare_transaction_args_rejects_bad_dates():
    with pytest.raises(ToolError, match="ISO 8601"):
        tt.prepare_transaction_args({"start_date": "yesterday"}, NOW)
    with pytest.raises(ToolError, match="before"):
        tt.prepare_transaction_args({"start_date": "2026-10-02T00:00:00Z", "end_date": "2026-10-01T00:00:00Z"}, NOW)


def test_compact_transactions_keeps_evidence_drops_personal_blocks():
    data = {"transaction_details": [{"transaction_info": {"transaction_id": "T1", "transaction_amount": {"value": "-1.00"},
                                                          "bank_reference_id": "zzz"},
                                     "payer_info": {"email_address": "a@b.c"}, "shipping_info": {}}],
            "total_items": 1, "account_number": "ACC", "links": []}
    out = tt.compact_transactions(data)
    assert out == {"total_items": 1, "returned": 1,
                   "transactions": [{"transaction_id": "T1", "transaction_amount": {"value": "-1.00"}}]}
    assert tt.compact_transactions({"items": []}) == {"items": []}
