"""Minimal Sarvam chat-completions client with a JSON-schema contract.

Request format is the one verified by scripts/smoke_sarvam.py (2026-11-07). Retries: up to `retries`
extra attempts with exponential backoff on 429, 5xx and network errors; any other 4xx fails at once.
A reply that is not valid JSON or violates the schema raises LLMSchemaError and is NOT retried (the
call was already billed). The API key is only ever placed in request headers; it is scrubbed from
every error message and hidden from repr().
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

from ..config import optional
from .errors import LLMConfigError, LLMHTTPError, LLMSchemaError, LLMTransportError

DEFAULT_MODEL = "sarvam-105b-conversations"
DEFAULT_BASE_URL = "https://api.sarvam.ai"
RETRY_STATUS = {429, 500, 502, 503, 504}

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def validate_schema(value: Any, schema: dict) -> None:
    """Check `value` against the subset of JSON Schema we use: type, required, properties, enum,
    minimum/maximum, non-empty strings via minLength. Unknown extra properties are tolerated (the
    reply was already paid for; callers read only the fields they asked for). Raises LLMSchemaError."""
    stype = schema.get("type")
    if stype:
        ok = isinstance(value, _TYPES[stype]) and not (stype in ("number", "integer") and isinstance(value, bool))
        if not ok:
            raise LLMSchemaError(f"expected {stype}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise LLMSchemaError(f"value {value!r} not in enum {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise LLMSchemaError(f"{value} below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise LLMSchemaError(f"{value} above maximum {schema['maximum']}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise LLMSchemaError("string shorter than minLength")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                raise LLMSchemaError(f"missing required field {name!r}")
        for name, sub in schema.get("properties", {}).items():
            if name in value:
                try:
                    validate_schema(value[name], sub)
                except LLMSchemaError as exc:
                    raise LLMSchemaError(f"{name}: {exc}") from None


class SarvamClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 30.0,
        retries: int = 2,
        backoff: float = 0.5,
        max_tokens: int = 300,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        key = (api_key if api_key is not None else optional("SARVAM_API_KEY", "")).strip()
        if not key:
            raise LLMConfigError("Missing environment variable: SARVAM_API_KEY (see .env.example)")
        self._key = key
        self.model = model or optional("SARVAM_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or optional("SARVAM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.retries = retries
        self.backoff = backoff
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._sleep = sleep
        self.requests_made = 0  # real HTTP requests sent, retries included
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def __repr__(self) -> str:
        return f"SarvamClient(model={self.model!r}, base_url={self.base_url!r})"

    def close(self) -> None:
        self._http.close()

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "***")

    def _post(self, body: dict) -> dict:
        headers = {  # the docs list both; sending both avoids guessing which is enforced (see smoke script)
            "api-subscription-key": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(self.backoff * 2 ** (attempt - 1))
            self.requests_made += 1
            try:
                resp = self._http.post("/v1/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as exc:
                last = LLMTransportError(self._scrub(f"{type(exc).__name__}: {exc}"))
                continue
            if resp.status_code in RETRY_STATUS:
                last = LLMHTTPError(resp.status_code, self._scrub(resp.text[:200]))
                continue
            if resp.status_code >= 400:
                raise LLMHTTPError(resp.status_code, self._scrub(resp.text[:200]))
            try:
                return resp.json()
            except ValueError:
                raise LLMSchemaError("response body was not JSON") from None
        assert last is not None
        raise last

    def complete_json(self, messages: list[dict], schema: dict, *, name: str = "defender_verdict") -> dict:
        """Chat completion constrained to `schema`; returns the parsed, validated JSON object."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
        }
        data = self._post(body)
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMSchemaError("response had no choices[0].message") from None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise LLMSchemaError("reply content was not valid JSON") from None
        validate_schema(parsed, schema)
        return parsed
