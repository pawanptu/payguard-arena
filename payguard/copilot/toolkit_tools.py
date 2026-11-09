"""Read-only tools from the official PayPal Agent Toolkit (`paypal_agent_toolkit.shared`, version 1.11.0).

Only the framework-agnostic `shared` layer is used (the openai/langchain/crewai adapters are never imported).

Safety layers, outermost first:
  1. Environment: credentials come from PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET only, and PAYPAL_BASE_URL must be
     unset or the sandbox host. Every call runs with Context(sandbox=True).
  2. Selection: the Toolkit's own `is_tool_allowed` with READ_ONLY_ACTIONS, intersected with a hard-coded
     allowlist of six method names.
  3. Dispatch: only `PayPalAPI.run` is used, and `ToolkitDispatcher.call` refuses any method outside the allowlist
     even if the LLM asks for one.
  4. HTTP: the client we hand to the Toolkit is a subclass whose post/put/delete raise, so a state-changing request
     cannot be sent even if a handler tried.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping
from urllib.parse import parse_qsl, urlencode, urlparse

from .tooling import SOURCE_TOOLKIT, Tool, ToolError

INSTALL_HINT = (
    "The PayPal Agent Toolkit is not installed. Run: pip install -r requirements-copilot.txt  and then  "
    "pip install --no-deps paypal-agent-toolkit==1.11.0"
)

READ_ONLY_ACTIONS: dict[str, dict[str, bool]] = {
    "transactions": {"list": True},
    "disputes": {"list": True, "get": True},
    "invoices": {"list": True, "get": True},
    "orders": {"get": True},
}
ALLOWED_METHODS = frozenset(
    {"list_transactions", "list_disputes", "get_dispute", "list_invoices", "get_invoice", "get_order_details"}
)
SANDBOX_HOST = "api-m.sandbox.paypal.com"
SANDBOX_BASE = f"https://{SANDBOX_HOST}"
SOURCE_NAME = "payguard-arena-copilot"  # shown in the User-Agent the Toolkit sends
HTTP_TIMEOUT_S = 20.0
MAX_SEARCH_MONTHS = 12
_DESC_MAX = 420

NOT_AUTHORIZED_TEXT = "PayPal sandbox: NOT_AUTHORIZED (this app lacks the permission for this API)"


class ToolkitUnavailable(RuntimeError):
    """paypal_agent_toolkit (or requests) is not installed."""


class ToolkitConfigError(RuntimeError):
    """Credentials missing or a non-sandbox host requested. Names variables, never values."""


class PayPalHTTPError(Exception):
    """HTTP error from the sandbox, reduced to the status and PayPal's short error name (never headers or body)."""

    def __init__(self, status: int, name: str = ""):
        super().__init__(f"HTTP {status} {name}".strip())
        self.status = status
        self.name = name


# ---------------------------------------------------------------- toolkit import and selection

def import_toolkit() -> dict[str, Any]:
    """Lazy import of the shared layer; raises ToolkitUnavailable with the install command."""
    try:
        from paypal_agent_toolkit.shared.api import PayPalAPI
        from paypal_agent_toolkit.shared.configuration import Configuration, Context, is_tool_allowed
        from paypal_agent_toolkit.shared.tools import tools
    except ImportError as exc:
        raise ToolkitUnavailable(f"{INSTALL_HINT} (import failed: {exc.name or type(exc).__name__})") from None
    return {
        "PayPalAPI": PayPalAPI,
        "Configuration": Configuration,
        "Context": Context,
        "is_tool_allowed": is_tool_allowed,
        "registry": tools,
    }


def filter_registry(
    registry: list[dict],
    *,
    is_allowed: Callable[[dict, Any], bool],
    configuration: Any,
) -> list[dict]:
    """Registry entries that the Toolkit's own permission check allows AND whose method is on our allowlist."""
    return [t for t in registry if t.get("method") in ALLOWED_METHODS and is_allowed(t, configuration)]


def enabled_entries(registry: list[dict] | None = None) -> list[dict]:
    """The enabled registry entries (real Toolkit registry unless one is passed in)."""
    tk = import_toolkit()
    configuration = tk["Configuration"](actions=READ_ONLY_ACTIONS, context=tk["Context"](sandbox=True))
    return filter_registry(
        registry if registry is not None else tk["registry"],
        is_allowed=tk["is_tool_allowed"],
        configuration=configuration,
    )


def toolkit_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("paypal-agent-toolkit")
    except PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------- environment

def sandbox_credentials(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """(client_id, secret) from the environment. Refuses unless PAYPAL_BASE_URL is unset or the sandbox host."""
    env = os.environ if env is None else env
    base = (env.get("PAYPAL_BASE_URL") or "").strip()
    if base and urlparse(base).hostname != SANDBOX_HOST:
        raise ToolkitConfigError(f"Refusing PAYPAL_BASE_URL host {urlparse(base).hostname!r}: sandbox only.")
    cid = (env.get("PAYPAL_CLIENT_ID") or "").strip()
    sec = (env.get("PAYPAL_CLIENT_SECRET") or "").strip()
    missing = [n for n, v in (("PAYPAL_CLIENT_ID", cid), ("PAYPAL_CLIENT_SECRET", sec)) if not v]
    if missing:
        raise ToolkitConfigError("Missing environment variables: " + ", ".join(missing) + " (see .env.example)")
    return cid, sec


# ---------------------------------------------------------------- the HTTP client handed to the Toolkit

_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def clean_uri(uri: str, payload: Mapping[str, Any] | None = None) -> str:
    """Query-string clean-up for the Toolkit's GET requests (WORKAROUND for Toolkit 1.11.0 bugs).

    Bug 1: `list_transactions` builds its query with `urlencode(validated.dict(...))`, so an unset optional
    `transaction_id` is sent as the literal `transaction_id=None`. Any parameter whose value is None or the literal
    string "None" is dropped here.
    Bug 2 (unverified, the sandbox app currently gets 403 on this API): the default `start_date`/`end_date` carry no
    UTC offset, but the Transaction Search API documents Internet date-time (RFC 3339) values, so "Z" is appended
    to the two date parameters of /v1/reporting/transactions when they have no offset.
    `payload`, when given (the Toolkit's search-by-id path passes its query as `payload=`), is merged into the URI.
    """
    parts = urlparse(uri)
    params = list(parse_qsl(parts.query, keep_blank_values=True))
    if payload:
        params += [(k, str(v)) for k, v in payload.items()]
    params = [(k, v) for k, v in params if v not in ("None", "")]
    if parts.path.rstrip("/") == "/v1/reporting/transactions":
        params = [(k, v if k not in ("start_date", "end_date") or _TZ_SUFFIX.search(v) else v + "Z")
                  for k, v in params]
    query = urlencode(params)
    return parts.path + (f"?{query}" if query else "")


def _error_name(resp: Any) -> str:
    try:
        name = resp.json().get("name", "")
    except Exception:
        return ""
    return name if isinstance(name, str) and re.fullmatch(r"[A-Z0-9_]{1,64}", name) else ""


def build_client_class() -> type:
    """Subclass of the Toolkit's PayPalClient: read-only, with timeouts, a token cache and the clean_uri fix.

    Defined lazily because the base class lives in the (optionally installed) Toolkit.
    """
    try:
        from paypal_agent_toolkit.shared.paypal_client import PayPalClient as Base
    except ImportError as exc:
        raise ToolkitUnavailable(f"{INSTALL_HINT} (import failed: {exc.name or type(exc).__name__})") from None

    class ReadOnlySandboxClient(Base):  # type: ignore[misc, valid-type]
        """GET only. The Toolkit's own client logs PayPal error headers on the root logger and sends requests
        without a timeout; this one does neither. `errors`/`ok_calls` let the dispatcher see failures that a
        Toolkit handler swallows (its search-by-transaction-id loop catches every exception)."""

        def __init__(self, client_id, secret, context, *, http=None, timeout: float = HTTP_TIMEOUT_S):
            super().__init__(client_id, secret, context)
            if urlparse(self.base_url).hostname != SANDBOX_HOST:
                raise ToolkitConfigError("Refusing a non-sandbox PayPal host: sandbox only.")
            if http is None:
                try:
                    import requests as http  # type: ignore[no-redef]
                except ImportError:
                    raise ToolkitUnavailable(INSTALL_HINT) from None
            self._http = http
            self._timeout = timeout
            self._token: str | None = None
            self._expires_at = 0.0
            self.errors: list[Exception] = []
            self.ok_calls = 0

        def begin(self) -> None:
            self.errors = []
            self.ok_calls = 0

        def get_access_token(self):
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            resp = self._http.post(
                f"{self.base_url}/v1/oauth2/token",
                headers={"Accept": "application/json"},
                data={"grant_type": "client_credentials"},
                auth=(self.client_id, self.secret),
                timeout=self._timeout,
            )
            if resp.status_code >= 400:
                raise PayPalHTTPError(resp.status_code, _error_name(resp) or "TOKEN_REQUEST_FAILED")
            body = resp.json()
            token = body.get("access_token") if isinstance(body, dict) else None
            if not token:
                raise PayPalHTTPError(resp.status_code, "NO_ACCESS_TOKEN")
            self._token = token
            self._expires_at = time.time() + int(body.get("expires_in", 0) or 0)
            return token

        def get(self, uri, payload=None):
            if self.errors and getattr(self.errors[-1], "status", 0) in (401, 403):
                # The Toolkit's search-by-id loop repeats the same request up to 12 times after a failure; once
                # the sandbox has said "not authorised", repeating cannot help.
                raise self.errors[-1]
            url = f"{self.base_url}{clean_uri(uri, payload)}"
            try:
                resp = self._http.get(url, headers=self.build_headers(), timeout=self._timeout)
                if resp.status_code >= 400:
                    raise PayPalHTTPError(resp.status_code, _error_name(resp))
                try:
                    data = resp.json()
                except ValueError:
                    data = {}
            except Exception as exc:
                self.errors.append(exc)
                raise
            self.ok_calls += 1
            # The Toolkit's search-by-id path (payload=...) runs json.loads() on the reply, every other path
            # expects a dict. Toolkit 1.11.0 is inconsistent, so serve each caller what it parses.
            return json.dumps(data) if payload is not None else data

        def _refuse(self, *_a, **_k):
            raise PermissionError("read-only client: PayPal write requests are disabled in the copilot")

        post = put = delete = _refuse

    return ReadOnlySandboxClient


# ---------------------------------------------------------------- list_transactions argument preparation

MAX_RANGE_DAYS = 31
DEFAULT_RANGE_DAYS = 30


def _parse_utc(value: Any, field: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ToolError(f"{field} is empty")
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        raise ToolError(f"{field} must be an ISO 8601 date-time such as 2026-10-10T07:36:24Z") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # a date without a timezone is read as UTC
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def prepare_transaction_args(args: dict, now: datetime | None = None) -> dict:
    """Explicit UTC dates for list_transactions (WORKAROUND for Toolkit 1.11.0 bug).

    The Toolkit's own defaults (Pydantic default_factory, isoformat() with no timezone) are rejected by PayPal with
    400 INVALID_REQUEST "Invalid date passed". So we always pass start_date/end_date as `YYYY-MM-DDTHH:MM:SSZ`:
    none given -> the last 30 days ending now; a date with no timezone -> read as UTC (Z appended); only one end
    given -> the other is 30 days away (end capped at now); range longer than 31 days -> start moved to end - 31 days.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    out = dict(args)
    start = _parse_utc(out["start_date"], "start_date") if out.get("start_date") else None
    end = _parse_utc(out["end_date"], "end_date") if out.get("end_date") else None
    if start is None and end is None:
        end = now
        start = end - timedelta(days=DEFAULT_RANGE_DAYS)
    elif end is None:
        end = min(start + timedelta(days=DEFAULT_RANGE_DAYS), now)
    elif start is None:
        start = end - timedelta(days=DEFAULT_RANGE_DAYS)
    if end < start:
        raise ToolError("end_date is before start_date")
    if end - start > timedelta(days=MAX_RANGE_DAYS):
        start = end - timedelta(days=MAX_RANGE_DAYS)
    out["start_date"], out["end_date"] = _fmt(start), _fmt(end)
    return out


_TX_INFO_KEEP = ("transaction_id", "paypal_reference_id", "transaction_event_code", "transaction_initiation_date",
                 "transaction_updated_date", "transaction_amount", "fee_amount", "transaction_status",
                 "transaction_subject", "transaction_note", "invoice_id", "custom_field")


def compact_transactions(data: Any) -> Any:
    """Keep the evidence from a Transaction Search reply (id, event code, date, amount, fee, status), drop the rest.

    34 full records overflow the model's observation budget; this keeps enough to reconcile payouts by amount and time.
    Personal-data keys are removed later by redact().
    """
    if not isinstance(data, dict) or not isinstance(data.get("transaction_details"), list):
        return data
    rows = []
    for d in data["transaction_details"]:
        info = d.get("transaction_info", {}) if isinstance(d, dict) else {}
        rows.append({k: info[k] for k in _TX_INFO_KEEP if k in info})
    head = {k: data[k] for k in ("start_date", "end_date", "page", "total_items", "total_pages", "message",
                                 "found") if k in data}
    return {**head, "returned": len(rows), "transactions": rows}


# ---------------------------------------------------------------- dispatch

def describe_error(exc: BaseException) -> str:
    """Short, safe, user-facing text for a failed Toolkit call (no headers, bodies or secrets)."""
    if isinstance(exc, PayPalHTTPError):
        if exc.status == 403 or exc.name == "NOT_AUTHORIZED":
            return NOT_AUTHORIZED_TEXT
        if exc.status == 401:
            return "PayPal sandbox: authentication failed (HTTP 401)"
        return f"PayPal sandbox: HTTP {exc.status}" + (f" {exc.name}" if exc.name else "")
    if type(exc).__name__ == "ValidationError":  # pydantic: the Toolkit validates arguments itself
        try:
            items = exc.errors()  # type: ignore[attr-defined]
            detail = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}" for e in items[:3])
        except Exception:
            detail = "arguments rejected"
        return f"Invalid arguments: {detail}"[:300]
    if type(exc).__module__.startswith("requests"):
        return f"PayPal sandbox: network error ({type(exc).__name__})"
    if isinstance(exc, PermissionError):
        return "refused: read-only copilot"
    return f"tool failed ({type(exc).__name__})"


class ToolkitDispatcher:
    """The only way the copilot reaches the Toolkit. `api` is a PayPalAPI (or a test double with .run)."""

    def __init__(self, api: Any, client: Any | None = None, *, now: Callable[[], datetime] | None = None):
        self._api = api
        self._client = client
        self._now = now or (lambda: datetime.now(timezone.utc))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, http: Any | None = None,
                 now: Callable[[], datetime] | None = None) -> "ToolkitDispatcher":
        cid, secret = sandbox_credentials(env)
        tk = import_toolkit()
        context = tk["Context"](sandbox=True, source=SOURCE_NAME)
        api = tk["PayPalAPI"](cid, secret, context)
        client = build_client_class()(cid, secret, context, http=http)
        api._paypal_client = client  # swap the Toolkit's default client for the read-only one
        return cls(api, client, now=now)

    @property
    def client(self) -> Any | None:
        return self._client

    def call(self, method: str, args: dict) -> Any:
        if method not in ALLOWED_METHODS:
            raise ToolError(f"refused: {method!r} is not an enabled read-only tool")
        args = dict(args)
        if method == "list_transactions":
            if isinstance(args.get("search_months"), int):
                args["search_months"] = max(1, min(args["search_months"], MAX_SEARCH_MONTHS))
            args = prepare_transaction_args(args, self._now())
        if self._client is not None:
            self._client.begin()
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # the Toolkit print()s per-month search errors
                raw = self._api.run(method, args)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(describe_error(exc)) from None
        if self._client is not None and self._client.errors and not self._client.ok_calls:
            # A handler swallowed every HTTP failure (list_transactions by id): surface it instead of "not found".
            raise ToolError(describe_error(self._client.errors[0]))
        return normalize_result(method, raw)


def normalize_result(method: str, raw: Any) -> Any:
    """Handlers return dicts or JSON strings; always hand back parsed JSON."""
    data: Any = raw
    if isinstance(raw, (str, bytes)):
        try:
            data = json.loads(raw)
        except ValueError:
            data = {"text": raw if isinstance(raw, str) else raw.decode("utf-8", "replace")}
    if method == "list_transactions":
        data = compact_transactions(data)
    if method == "get_order_details" and isinstance(data, dict):
        # Toolkit bug: its get_order_details always says "has been successfully captured", even for an
        # uncaptured order. Keep only the real fields.
        data = {k: v for k, v in data.items() if k != "message"}
    return data


# ---------------------------------------------------------------- tool objects for the agent

def simplify_schema(model_schema: dict) -> dict:
    """Pydantic model_json_schema() -> small JSON schema for the LLM prompt (type, description, enum, required)."""
    props: dict[str, dict] = {}
    for name, spec in (model_schema.get("properties") or {}).items():
        spec = dict(spec)
        if "anyOf" in spec:  # Optional[X] -> X (the description stays on the outer level)
            inner = next((s for s in spec["anyOf"] if s.get("type") != "null"), {})
            spec = {**inner, **{k: v for k, v in spec.items() if k != "anyOf"}}
        slim = {k: spec[k] for k in ("type", "description", "enum", "minimum", "maximum") if k in spec}
        slim.setdefault("type", "string")
        props[name] = slim
    out: dict[str, Any] = {"type": "object", "properties": props}
    if model_schema.get("required"):
        out["required"] = list(model_schema["required"])
    return out


def _short(text: str, limit: int = _DESC_MAX) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def toolkit_tools(dispatcher: ToolkitDispatcher, registry: list[dict] | None = None) -> list[Tool]:
    """One Tool per enabled registry entry. `registry` is only for tests; None means the real Toolkit registry."""
    out = []
    for entry in enabled_entries(registry):
        method = entry["method"]
        out.append(
            Tool(
                name=method,
                source=SOURCE_TOOLKIT,
                description="Read-only PayPal sandbox call (official Agent Toolkit). " + _short(entry.get("description", "")),
                schema=simplify_schema(entry["args_schema"].model_json_schema()),
                fn=lambda args, _m=method: dispatcher.call(_m, args),
            )
        )
    return out
