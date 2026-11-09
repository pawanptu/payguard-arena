"""Tool-using loop over `complete_json(messages, schema)`, so it works with SarvamClient and GeminiClient alike.

Each step the model returns one JSON object:
  {"thought": str, "action": "call_tool" | "final_answer", "tool": str, "args_json": str, "answer": str}
The loop parses `args_json` strictly, runs the tool, redacts and truncates the result, and feeds it back as the next
user message. Protocol slips (bad JSON, unknown tool, empty answer) become an error observation and the loop
continues; they are counted in `protocol_errors` but are not tool calls, so they are not contract-C turns.
After `max_steps` model calls without a final answer, one more call is forced to answer from what it has seen.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .redact import redact
from .tooling import Tool, ToolError

DEFAULT_MAX_STEPS = 6
DEFAULT_MAX_RESULT_CHARS = 8000

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["call_tool", "final_answer"]},
        "tool": {"type": "string"},
        "args_json": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["thought", "action", "tool", "args_json", "answer"],
}
FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["final_answer"]},
        "tool": {"type": "string"},
        "args_json": {"type": "string"},
        "answer": {"type": "string", "minLength": 1},
    },
    "required": ["thought", "action", "tool", "args_json", "answer"],
}

SYSTEM_PROMPT = """You are the ops copilot of a payout-fraud console. You help a human reviewer by answering questions.

Rules:
- You are READ-ONLY. You can only call the tools listed below. You can never release, hold, send, refund, cancel or change anything. If asked to, say you cannot and what the reviewer can do instead.
- The payouts, scores and cases in the console are SIMULATED replay data. Only calls to the PayPal sandbox (sources paypal-agent-toolkit and payguard-rest) return real sandbox data.
- Base every fact on a tool result and say which tool it came from, for example "(get_hold_case)". Never invent ids, numbers or statuses. If you did not look something up, say so.
- If a tool result says a PayPal API is not authorised (NOT_AUTHORIZED), say so plainly: this sandbox app lacks the permission for that API. Do not guess what the data would have shown.
- Personal data (names, emails, addresses) is removed from tool results. Do not try to recover it.
- To reconcile payouts with PayPal's records: list_live_payouts gives our payouts, list_transactions gives the sandbox ledger (pass dates as 2026-10-10T07:36:24Z, at most 31 days apart). A payout shows up there as event code T0001 with a negative amount of the same size a few seconds after its created_at, and its `custom` field is item-<event_id> (for example item-736 for event 736). T1503 rows are the matching holds and T1105 rows their releases. Say which payouts you could and could not match, and how.
- Keep the final answer short, in plain language: what you found, what it means, what the reviewer should check next.

Each turn, reply with ONE JSON object with exactly these keys:
  "thought": one short sentence on what you will do or what you learned
  "action": "call_tool" or "final_answer"
  "tool": the tool name when action is "call_tool", else ""
  "args_json": the tool arguments as a JSON object written as a string, e.g. "{\\"event_id\\": 24}"; "{}" when none
  "answer": the final answer when action is "final_answer", else ""
Call one tool per turn. When you have enough information, give the final answer.

Tools:
"""


class LLM(Protocol):
    def complete_json(self, messages: list[dict], schema: dict) -> dict: ...


def render_tools(tools: list[Tool]) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name} [source: {t.source}]: {t.description}\n  args schema: "
                     f"{json.dumps(t.schema, separators=(',', ':'))}")
    return "\n".join(lines)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"


@dataclass
class CopilotResult:
    question: str
    answer: str
    turns: list[dict]  # contract C turns
    steps: int  # model calls made
    protocol_errors: int = 0
    forced_final: bool = False
    tool_names: list[str] = field(default_factory=list)


class Copilot:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        clock=time.perf_counter,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.max_result_chars = max_result_chars
        self._clock = clock
        self.system_prompt = SYSTEM_PROMPT + render_tools(tools)

    # ------------------------------------------------------------------ one tool call
    def _run_tool(self, tool: Tool, args: dict) -> tuple[bool, Any, int]:
        t0 = self._clock()
        try:
            data = redact(tool.fn(dict(args)))
            ok, payload = True, data
        except ToolError as exc:
            ok, payload = False, str(exc)
        except Exception as exc:  # never let a tool crash the loop; type name only, no message (could hold secrets)
            ok, payload = False, f"tool failed ({type(exc).__name__})"
        ms = int(round((self._clock() - t0) * 1000))
        return ok, payload, ms

    def _observation(self, step: int, tool: str, ok: bool, payload: Any) -> str:
        body = json.dumps({"ok": ok, ("data" if ok else "error"): payload}, ensure_ascii=False, default=str)
        return (f"OBSERVATION step {step} tool {tool}: {truncate(body, self.max_result_chars)}\n"
                "Call another tool or give the final answer.")

    # ------------------------------------------------------------------ the loop
    def ask(self, question: str) -> CopilotResult:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": question}]
        turns: list[dict] = [{"role": "user", "text": question}]
        step = 0  # number of tool calls so far
        protocol_errors = 0
        calls = 0

        def protocol_error(out: dict, text: str) -> None:
            nonlocal protocol_errors
            protocol_errors += 1
            messages.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"ERROR: {text} Reply again with one valid JSON object."})

        while calls < self.max_steps:
            calls += 1
            out = self.llm.complete_json(messages, STEP_SCHEMA)
            action = out.get("action")
            if action == "final_answer":
                answer = str(out.get("answer") or "").strip()
                if answer:
                    turns.append({"role": "assistant", "text": answer})
                    return CopilotResult(question, answer, turns, calls, protocol_errors, False, list(self.tools))
                protocol_error(out, "final_answer needs a non-empty \"answer\".")
                continue
            name = str(out.get("tool") or "").strip()
            tool = self.tools.get(name)
            if action != "call_tool" or tool is None:
                known = ", ".join(self.tools)
                protocol_error(out, f"unknown tool or action ({name!r}). Allowed tools: {known}.")
                continue
            raw = str(out.get("args_json") or "").strip() or "{}"
            try:
                args = json.loads(raw)
            except ValueError:
                args = None
            if not isinstance(args, dict):
                protocol_error(out, "args_json must be a JSON object written as a string.")
                continue
            step += 1
            safe_args = redact(args)
            turns.append({"role": "tool_call", "step": step, "tool": name, "source": tool.source, "args": safe_args})
            ok, payload, ms = self._run_tool(tool, args)
            result_turn: dict[str, Any] = {"role": "tool_result", "step": step, "tool": name, "ok": ok, "ms": ms}
            result_turn["data" if ok else "error"] = payload
            turns.append(result_turn)
            messages.append({"role": "assistant", "content": json.dumps(
                {"thought": out.get("thought", ""), "action": "call_tool", "tool": name,
                 "args_json": json.dumps(safe_args), "answer": ""}, ensure_ascii=False)})
            messages.append({"role": "user", "content": self._observation(step, name, ok, payload)})

        # Step budget used up: force a final answer from what has been observed.
        messages.append({"role": "user", "content": (
            f"Step limit reached ({self.max_steps}). Give your final answer now, using only the observations "
            "above. Say what you could not check. Set action to final_answer.")})
        out = self.llm.complete_json(messages, FINAL_SCHEMA)
        answer = str(out.get("answer") or "").strip() or (
            f"I could not finish within {self.max_steps} steps. See the tool results above.")
        turns.append({"role": "assistant", "text": answer})
        return CopilotResult(question, answer, turns, calls + 1, protocol_errors, True, list(self.tools))
