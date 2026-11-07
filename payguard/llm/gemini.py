"""Minimal Google Gemini (generateContent REST) client with a JSON-schema contract; same interface as SarvamClient.

Request: POST {base}/v1beta/models/{model}:generateContent, API key in the `x-goog-api-key` header only (never in
the URL or query string). System prompt -> `systemInstruction`, the rest -> `contents`; JSON output via
`generationConfig.responseMimeType = "application/json"` + `responseJsonSchema` (JSON Schema, lowercase types).
Sources (read 2026-11-08, see the worker report for what is UNVERIFIED):
  https://ai.google.dev/api/generate-content  (-> https://aistudio.google.com/api-reference/generate-content)
  https://ai.google.dev/gemini-api/docs/structured-output  (-> https://aistudio.google.com/docs/structured-output)

Free-tier limits are unknown, so the client throttles itself: at least `min_interval_s` between two requests
(env GEMINI_MIN_INTERVAL_S, default 6.5 s ~ 9 per minute), retries included. Retries: up to `retries` extra
attempts on 429/500/502/503/504 and network errors; the wait honours `Retry-After` or Google's RetryInfo
`retryDelay`, capped at 60 s, else exponential backoff. A 429 that signals an exhausted DAILY / project quota
raises LLMQuotaError at once. A reply with no candidate text (safety block, empty) or that is not valid JSON /
violates the schema raises LLMSchemaError and is not retried. The key is scrubbed from every error message,
hidden from repr(), and neither the key nor any response body is ever put in an exception message.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable

import httpx

from ..config import optional
from .client import validate_schema
from .errors import LLMConfigError, LLMHTTPError, LLMQuotaError, LLMSchemaError, LLMTransportError

# A model on the free tier: gemini-3.5-flash-lite is listed as stable with "Free of charge" input/output on
# https://ai.google.dev/gemini-api/docs/pricing (-> https://aistudio.google.com/docs/pricing) and as stable on
# https://ai.google.dev/gemini-api/docs/models. Its default thinking level is "minimal" (docs/thinking), so no
# thinking config is sent. Override with GEMINI_MODEL.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MIN_INTERVAL_S = 6.5
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_WAIT_S = 60.0

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*$")
_DAILY_HINT = re.compile(r"per.?day|daily", re.IGNORECASE)


def _to_gemini_schema(schema: Any) -> Any:
    """JSON Schema accepted by `responseJsonSchema`. `additionalProperties` is dropped (our validator tolerates
    extras anyway, and not every Gemini model documents it); everything else the judge uses (type, properties,
    required, enum) is supported as-is."""
    if isinstance(schema, dict):
        return {k: _to_gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_to_gemini_schema(v) for v in schema]
    return schema


def _parse_duration(value: Any) -> float | None:
    """'28s', '0.5s', '250ms', 12 -> seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    if isinstance(value, str):
        m = _DURATION.match(value)
        if m:
            x = float(m.group(1))
            return x / 1000.0 if m.group(2) == "ms" else x
    return None


def _error_info(resp: httpx.Response) -> dict:
    """The `error` object of a Google error body, or {} when the body is not that shape."""
    try:
        err = resp.json().get("error", {})
    except (ValueError, AttributeError):
        return {}
    return err if isinstance(err, dict) else {}


def _details(err: dict) -> list[dict]:
    d = err.get("details", [])
    return [x for x in d if isinstance(x, dict)] if isinstance(d, list) else []


def _is_quota_exhausted(err: dict) -> bool:
    """429 that will not clear by waiting seconds: a per-day (or project) quota is used up.

    Recognised (UNVERIFIED against live responses, see report): google.rpc QuotaFailure violations whose quotaId
    mentions a per-day quota (e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier) or whose quotaValue is "0"
    (no free quota for that model); and the newer string codes `quota_exceeded`."""
    status = str(err.get("status", "")) + " " + str(err.get("code", ""))
    if "quota_exceeded" in status.lower():
        return True
    for d in _details(err):
        if str(d.get("@type", "")).endswith("QuotaFailure"):
            for v in d.get("violations", []) if isinstance(d.get("violations"), list) else []:
                if not isinstance(v, dict):
                    continue
                if _DAILY_HINT.search(str(v.get("quotaId", ""))) or str(v.get("quotaValue", "")) == "0":
                    return True
    return False


def _retry_delay(resp: httpx.Response, err: dict) -> float | None:
    """Seconds the server asks us to wait: `Retry-After` header first, else RetryInfo.retryDelay."""
    ra = _parse_duration(resp.headers.get("retry-after", "").strip().rstrip("s"))
    if ra is not None:
        return ra
    for d in _details(err):
        if str(d.get("@type", "")).endswith("RetryInfo"):
            got = _parse_duration(d.get("retryDelay"))
            if got is not None:
                return got
    return None


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 2.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        min_interval_s: float | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        key = (api_key if api_key is not None else optional("GEMINI_API_KEY", "")).strip()
        if not key:
            raise LLMConfigError("Missing environment variable: GEMINI_API_KEY (see .env.example)")
        self._key = key
        self.model = (model or optional("GEMINI_MODEL", DEFAULT_MODEL)).removeprefix("models/")
        self.base_url = (base_url or optional("GEMINI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        if min_interval_s is None:
            try:
                min_interval_s = float(optional("GEMINI_MIN_INTERVAL_S", str(DEFAULT_MIN_INTERVAL_S)))
            except ValueError:
                raise LLMConfigError("GEMINI_MIN_INTERVAL_S must be a number of seconds") from None
        self.min_interval_s = max(0.0, min_interval_s)
        self.retries = retries
        self.backoff = backoff
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self._throttle_lock = threading.Lock()
        self.requests_made = 0  # real HTTP requests sent, retries included
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def __repr__(self) -> str:
        return f"GeminiClient(model={self.model!r}, base_url={self.base_url!r})"

    def close(self) -> None:
        self._http.close()

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "***")

    def _throttle(self) -> None:
        """Block until at least min_interval_s has passed since the previous request was sent."""
        with self._throttle_lock:
            if self._last_request_at is not None:
                wait = self.min_interval_s - (self._clock() - self._last_request_at)
                if wait > 0:
                    self._sleep(wait)
            self._last_request_at = self._clock()

    def _http_error(self, resp: httpx.Response, err: dict, cls: type[LLMHTTPError] = LLMHTTPError) -> LLMHTTPError:
        # Only the short status enum (e.g. RESOURCE_EXHAUSTED) is surfaced, never the body or message text.
        status = err.get("status")
        label = status if isinstance(status, str) and _SAFE_TOKEN.match(status) else "error"
        return cls(resp.status_code, self._scrub(label))

    def _post(self, path: str, body: dict) -> dict:
        headers = {"x-goog-api-key": self._key, "Content-Type": "application/json"}
        last: Exception | None = None
        wait = 0.0
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(min(wait, MAX_WAIT_S))
            self._throttle()
            self.requests_made += 1
            wait = self.backoff * 2 ** attempt
            try:
                resp = self._http.post(path, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last = LLMTransportError(self._scrub(f"{type(exc).__name__}: {exc}"))
                continue
            if resp.status_code < 400:
                try:
                    return resp.json()
                except ValueError:
                    raise LLMSchemaError("response body was not JSON") from None
            err = _error_info(resp)
            if resp.status_code == 429 and _is_quota_exhausted(err):
                raise self._http_error(resp, err, LLMQuotaError)
            if resp.status_code in RETRY_STATUS:
                last = self._http_error(resp, err)
                hint = _retry_delay(resp, err)
                if hint is not None:
                    wait = hint
                continue
            raise self._http_error(resp, err)
        assert last is not None
        raise last

    @staticmethod
    def _to_request(messages: list[dict]) -> tuple[dict | None, list[dict]]:
        system: list[str] = []
        contents: list[dict] = []
        for m in messages:
            text = str(m.get("content", ""))
            role = m.get("role")
            if role == "system":
                system.append(text)
            else:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]})
        instruction = {"parts": [{"text": "\n\n".join(system)}]} if system else None
        return instruction, contents

    def complete_json(self, messages: list[dict], schema: dict) -> dict:
        """One generateContent call constrained to `schema`; returns the parsed, validated JSON object."""
        instruction, contents = self._to_request(messages)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": _to_gemini_schema(schema),
            },
        }
        if instruction is not None:
            body["systemInstruction"] = instruction
        data = self._post(f"/v1beta/models/{self.model}:generateContent", body)
        text = self._candidate_text(data)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            raise LLMSchemaError("reply content was not valid JSON") from None
        validate_schema(parsed, schema)
        return parsed

    @staticmethod
    def _candidate_text(data: Any) -> str:
        """Text of the first candidate (non-thought parts joined). Raises LLMSchemaError if there is none."""
        def reason(value: Any) -> str:
            return value if isinstance(value, str) and _SAFE_TOKEN.match(value) else "unknown"

        try:
            cand = data["candidates"][0]
            parts = cand["content"]["parts"]
            text = "".join(p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought"))
        except (KeyError, IndexError, TypeError, AttributeError):
            fb = data.get("promptFeedback") if isinstance(data, dict) else None
            block = fb.get("blockReason") if isinstance(fb, dict) else None
            finish = None
            if isinstance(data, dict) and isinstance(data.get("candidates"), list) and data["candidates"]:
                c0 = data["candidates"][0]
                finish = c0.get("finishReason") if isinstance(c0, dict) else None
            raise LLMSchemaError(
                f"response had no candidate text (blockReason={reason(block)}, finishReason={reason(finish)})"
            ) from None
        if not text.strip():
            raise LLMSchemaError(f"response had no candidate text (finishReason={reason(cand.get('finishReason'))})")
        return text
