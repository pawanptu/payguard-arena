"""Tool abstraction shared by the three tool sources (Toolkit, our REST client, local console data)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

SOURCE_TOOLKIT = "paypal-agent-toolkit"
SOURCE_REST = "payguard-rest"
SOURCE_CONSOLE = "payguard-console"


class ToolError(Exception):
    """A tool failed in an expected way. The message is short, safe to show and to record (no secrets)."""


@dataclass(frozen=True)
class Tool:
    name: str
    source: str
    description: str
    schema: dict  # simplified JSON schema of the arguments (type/properties/required), for the LLM prompt
    fn: Callable[[dict], Any]  # takes the parsed args dict, returns JSON-able data or raises ToolError
