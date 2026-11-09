"""Ops copilot CLI (read-only, PayPal sandbox).

  python -m scripts.copilot check
  python -m scripts.copilot tools
  python -m scripts.copilot ask "Why was payout #24 held?" [--provider gemini|sarvam] [--max-steps 6]
         [--record ID --title "..." --order N] [--force]
  (global option before the subcommand: --env-file PATH)

Needs `pip install -r requirements-copilot.txt` and `pip install --no-deps paypal-agent-toolkit==1.11.0`.
Credentials are read from the environment / a local .env (PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET, and
GEMINI_API_KEY or SARVAM_API_KEY). Secrets and tokens are never printed.
Exit codes: 0 ok, 2 setup problem (missing install or variables), 1 other failure.
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from payguard.copilot.agent import DEFAULT_MAX_STEPS, Copilot
from payguard.copilot.record import RecordError, build_session, validate_slug, write_session
from payguard.copilot.registry import build_tools
from payguard.copilot.tooling import SOURCE_TOOLKIT
from payguard.copilot.toolkit_tools import (
    ToolkitConfigError,
    ToolkitDispatcher,
    ToolkitUnavailable,
    toolkit_version,
)
from payguard.llm.errors import LLMConfigError, LLMError


def _quiet_logging() -> None:
    # The Toolkit logs PayPal error bodies and response headers with logging.error on the root logger.
    logging.disable(logging.CRITICAL)


def _make_llm(provider: str):
    if provider == "gemini":
        from payguard.llm.gemini import GeminiClient

        return GeminiClient(max_tokens=1500)
    from payguard.llm.client import SarvamClient

    return SarvamClient(max_tokens=1200)


def _print_transcript(turns: list[dict]) -> None:
    for t in turns:
        role = t["role"]
        if role == "user":
            print(f"\nQUESTION: {t['text']}\n")
        elif role == "tool_call":
            print(f"[step {t['step']}] CALL   {t['tool']}  source={t['source']}  args={t['args']}")
        elif role == "tool_result":
            if t["ok"]:
                text = str(t["data"])
                shown = text if len(text) <= 300 else text[:300] + f"... ({len(text)} chars)"
                print(f"[step {t['step']}] RESULT {t['tool']}  ok=True  {t['ms']} ms  {shown}")
            else:
                print(f"[step {t['step']}] RESULT {t['tool']}  ok=False  {t['ms']} ms  error={t['error']}")
        elif role == "assistant":
            print(f"\nANSWER:\n{t['text']}")


def cmd_check(args: argparse.Namespace) -> int:
    try:
        dispatcher = ToolkitDispatcher.from_env()
        dispatcher.client.get_access_token()
    except (ToolkitUnavailable, ToolkitConfigError) as exc:
        print(f"setup problem: {exc}")
        return 2
    except Exception as exc:
        status = getattr(exc, "status", None)
        print(f"sandbox auth FAILED: {type(exc).__name__}" + (f" (HTTP {status})" if status else ""))
        return 1
    print("sandbox auth OK")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    try:
        tools = build_tools()
    except (ToolkitUnavailable, ToolkitConfigError) as exc:
        print(f"setup problem: {exc}")
        return 2
    for t in tools:
        print(f"{t.name:<20} {t.source}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if args.record:
        try:
            validate_slug(args.record)
        except RecordError as exc:
            print(f"error: {exc}")
            return 2
    try:
        tools = build_tools()
        llm = _make_llm(args.provider)
    except (ToolkitUnavailable, ToolkitConfigError, LLMConfigError) as exc:
        print(f"setup problem: {exc}")
        return 2
    try:
        result = Copilot(llm, tools, max_steps=args.max_steps).ask(args.question)
    except LLMError as exc:
        print(f"LLM failed: {type(exc).__name__}: {exc}")
        return 1
    _print_transcript(result.turns)
    print(f"\n({result.steps} model calls, {result.protocol_errors} protocol errors"
          + (", final answer was forced at the step limit" if result.forced_final else "") + ")")
    if args.record:
        session = build_session(
            session_id=args.record,
            title=args.title or args.question[:120],
            order=args.order,
            turns=result.turns,
            llm_provider=args.provider,
            llm_model=getattr(llm, "model", "unknown"),
            toolkit_version=toolkit_version(),
            enabled_tools=[t.name for t in tools if t.source == SOURCE_TOOLKIT],
        )
        try:
            path = write_session(session, force=args.force)
        except RecordError as exc:
            print(f"not recorded: {exc}")
            return 1
        print(f"recorded {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="scripts.copilot", description="Read-only PayPal sandbox ops copilot")
    p.add_argument("--env-file", default=None, help="path of a .env to load (default: the usual search)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="get a sandbox access token and print only OK or the error name")
    sub.add_parser("tools", help="list the enabled tools and their sources")
    a = sub.add_parser("ask", help="ask the copilot one question")
    a.add_argument("question")
    a.add_argument("--provider", choices=["gemini", "sarvam"], default="gemini")
    a.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    a.add_argument("--record", metavar="ID", help="write eval/copilot/sessions/<ID>.json")
    a.add_argument("--title", help="session title (default: the question)")
    a.add_argument("--order", type=int, default=100, help="sort order on the site (default 100)")
    a.add_argument("--force", action="store_true", help="overwrite an existing session file")
    args = p.parse_args(argv)
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()
    _quiet_logging()
    return {"check": cmd_check, "tools": cmd_tools, "ask": cmd_ask}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
