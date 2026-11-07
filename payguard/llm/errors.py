"""Typed errors for the LLM layer. Messages never contain the API key."""
from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for every failure of an LLM call."""


class LLMConfigError(LLMError):
    """The client cannot be built (for example no API key). Names the variable, never a value."""


class LLMTransportError(LLMError):
    """Network failure or timeout after all retries."""


class LLMHTTPError(LLMError):
    """Non-success HTTP status (4xx immediately, 429/5xx after the retries)."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class LLMSchemaError(LLMError):
    """The reply was not JSON or did not satisfy the schema's required fields, types or enums."""
