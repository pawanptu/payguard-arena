"""Our own read-only PayPal REST call. The Agent Toolkit has no Payouts tools, so payout status comes from here.

`get_payout_batch` is `GET /v1/payments/payouts/{id}` through payguard.paypal_client (sandbox-host guard
included). It reads; it never creates a batch.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping

from ..paypal_client import PayPalClient, PayPalError
from .tooling import SOURCE_REST, Tool, ToolError
from .toolkit_tools import NOT_AUTHORIZED_TEXT, sandbox_credentials

_BATCH_ID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_NETWORK_ERRORS = {"ConnectError", "ReadTimeout", "ConnectTimeout", "TimeoutException", "RemoteProtocolError",
                   "NetworkError", "PoolTimeout", "WriteTimeout"}

DESCRIPTION = (
    "Read-only. Status of one PayPal Payouts batch and its items (batch_status, item status, amount, fees). "
    "The PayPal Agent Toolkit has no Payouts tools, so this is our own read-only REST call "
    "(GET /v1/payments/payouts/{id}) to the PayPal sandbox. Get payout_batch_id from list_live_payouts."
)
SCHEMA = {
    "type": "object",
    "properties": {"payout_batch_id": {"type": "string", "description": "PayPal payout batch id, e.g. U2UPGB9BXLFRY"}},
    "required": ["payout_batch_id"],
}


def _describe(exc: PayPalError) -> str:
    name = ""
    try:
        body = json.loads(exc.body)
        if isinstance(body, dict) and re.fullmatch(r"[A-Z0-9_]{1,64}", str(body.get("name", ""))):
            name = body["name"]
    except ValueError:
        pass
    if exc.status == 403 or name == "NOT_AUTHORIZED":
        return NOT_AUTHORIZED_TEXT
    return f"PayPal sandbox: HTTP {exc.status}" + (f" {name}" if name else "")


def get_payout_batch_tool(
    client: PayPalClient | None = None,
    *,
    env: Mapping[str, str] | None = None,
    client_factory: Callable[[], PayPalClient] | None = None,
) -> Tool:
    """Tool for `get_payout_batch`. The client is built lazily on first use (listing tools needs no network)."""
    holder: dict[str, PayPalClient] = {}
    if client is not None:
        holder["c"] = client

    def _client() -> PayPalClient:
        if "c" not in holder:
            if client_factory is not None:
                holder["c"] = client_factory()
            else:
                cid, secret = sandbox_credentials(env)  # also enforces the sandbox host
                holder["c"] = PayPalClient(cid, secret)
        return holder["c"]

    def run(args: dict) -> Any:
        batch_id = args.get("payout_batch_id")
        if not isinstance(batch_id, str) or not _BATCH_ID.match(batch_id.strip()):
            raise ToolError("payout_batch_id must be a PayPal batch id (letters, digits, - or _)")
        try:
            return _client().get_payout(batch_id.strip())
        except PayPalError as exc:
            raise ToolError(_describe(exc)) from None
        except Exception as exc:
            if type(exc).__name__ in _NETWORK_ERRORS:
                raise ToolError(f"PayPal sandbox: network error ({type(exc).__name__})") from None
            raise

    return Tool(name="get_payout_batch", source=SOURCE_REST, description=DESCRIPTION, schema=SCHEMA, fn=run)
