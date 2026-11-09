"""Assemble the copilot's tool list from the three sources."""
from __future__ import annotations

from typing import Mapping

from .console_tools import ConsoleTools
from .rest_tools import get_payout_batch_tool
from .tooling import Tool
from .toolkit_tools import ToolkitDispatcher, toolkit_tools


def build_tools(env: Mapping[str, str] | None = None, *, dispatcher: ToolkitDispatcher | None = None,
                console: ConsoleTools | None = None) -> list[Tool]:
    """Toolkit tools (real registry) + get_payout_batch + console tools. Raises ToolkitUnavailable/ToolkitConfigError."""
    dispatcher = dispatcher or ToolkitDispatcher.from_env(env)
    return toolkit_tools(dispatcher) + [get_payout_batch_tool(env=env)] + (console or ConsoleTools()).tools()
