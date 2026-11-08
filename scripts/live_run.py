"""Recorded live sandbox run: pay a few replay events through the PayPal SANDBOX, then record them.

    python -m scripts.live_run                    dry run: prints the plan, touches nothing
    python -m scripts.live_run --send             really send (sandbox money, small sample)
    python -m scripts.live_run --reattach         after --replay was rebuilt: carry the receipts already in
                                                  --out onto it (sends nothing, needs no credentials)

With --send the script
  1. loads the replay and picks `--count` events (`select_live_events`),
  2. pays each through `LiveRail` (its per-payout cap, budget and receiver allowlist always apply),
  3. polls PayPal until each batch settles,
  4. pulls the signature-verified webhook events for those batches from the deployed app
     (GET {render-url}/api/webhooks/recent), and
  5. writes `--out`: the replay plus the live receipts (mode "live") and `verified_webhooks`.
Deploy that file as the replay so the live payouts and their webhooks show up in the dashboard.

The dry run needs no credentials and makes no network call. No secret is ever printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

from payguard.api.replay import (
    Episode,
    EpisodeEvent,
    Replay,
    ReplayError,
    load_replay,
    parse_replay,
)
from payguard.config import MissingConfig, require
from payguard.paypal_client import PayPalError
from payguard.rail.base import (
    BATCH_TERMINAL,
    ITEM_TERMINAL,
    PayoutReceipt,
    PayoutRequest,
    RailError,
)
from payguard.rail.live_rail import DEFAULT_MAX_PAYOUT_USD

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPLAY = ROOT / "eval" / "replay.json"
DEFAULT_OUT = ROOT / "eval" / "replay_live.json"
SAMPLE_REPLAY = "payguard/api/fixtures/sample_replay.json"
DEFAULT_RENDER_URL = "https://payguard-arena.onrender.com"
MAX_COUNT = 5
DEFAULT_COUNT = 3
DEFAULT_AMOUNT = "1.00"
REFRESH_EVERY_S = 10.0
WEBHOOK_EVERY_S = 15.0
FIRST_WEBHOOK_TIMEOUT_S = 60.0  # the free Render tier may be asleep
WEBHOOK_TIMEOUT_S = 20.0
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


# ---- selection and limits -------------------------------------------------------------------------
def demo_priority(event: EpisodeEvent) -> tuple:
    """Sort key for the demo: among eligible events (RELEASE, paid, not yet live), LOWER keys are sent first.

    Python's sort is stable, so events with equal keys keep their episode order.
    Available per event: id, t (hours since episode start), worker_id, requester_id, amount (USD in the
    simulation; the real sandbox payout is always --amount), is_attack, family ("honest", "ring",
    "structuring", ...), signals (specialist -> risk in [0, 1], plus "memory" in the final replay, or None),
    weights (specialist -> L1 weight, or None).

    Policy (user's choice, D-031): "false alarm overruled". Honest workers first; among them, the payout one
    voice was most worried about but the swarm still released, so the demo shows the swarm weighing its
    specialists instead of blocking on a single alarm. Events without signals count as no worry.
    """
    worry = max(event.signals.values()) if event.signals else 0.0
    return (event.is_attack, -worry)


def select_live_events(
    episode: Episode, count: int, *, exclude_ids: frozenset[int] | set[int] = frozenset()
) -> list[EpisodeEvent]:
    """Which episode events get a real sandbox payout.

    Policy: events whose verdict is RELEASE and that were paid, ordered by `demo_priority` (episode order
    on ties), first `count` of them. `exclude_ids` skips events that already have a live payout in the
    replay, so a re-run never pays the same event twice.
    """
    eligible = [e for e in episode.events
                if e.verdict == "RELEASE" and e.paid and e.id not in exclude_ids]
    return sorted(eligible, key=demo_priority)[:count]


def payout_cap() -> Decimal:
    """The script's own ceiling per payout: LiveRail's default cap, or a lower PAYGUARD_LIVE_MAX_PAYOUT_USD.

    Read from the process environment only (no .env loading), so the dry run touches no credential
    file. LiveRail still enforces its configured cap when sending.
    """
    cap = Decimal(DEFAULT_MAX_PAYOUT_USD)
    raw = os.environ.get("PAYGUARD_LIVE_MAX_PAYOUT_USD", "").strip()
    if raw:
        try:
            cap = min(cap, Decimal(raw))
        except InvalidOperation:
            pass
    return cap


def parse_amount(text: str) -> Decimal:
    try:
        amount = Decimal(text)
    except InvalidOperation:
        raise ValueError(f"--amount {text!r} is not a number") from None
    if not amount.is_finite() or amount <= 0:
        raise ValueError("--amount must be a positive number")
    amount = amount.quantize(Decimal("0.01"))
    cap = payout_cap()
    if amount > cap:
        raise ValueError(f"--amount {amount} exceeds the live payout cap of {cap} USD")
    return amount


# ---- the run --------------------------------------------------------------------------------------
@dataclass
class RunResult:
    receipts: list[PayoutReceipt] = field(default_factory=list)
    webhooks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _describe(exc: Exception) -> str:
    """One safe line for an error: never a response body."""
    if isinstance(exc, PayPalError):
        return f"PayPal HTTP {exc.status} (debug_id={exc.debug_id})"
    if isinstance(exc, (RailError, MissingConfig)):
        return str(exc)
    return type(exc).__name__


def submit_all(rail, events: list[EpisodeEvent], amount: Decimal, receiver: str,
               warnings: list[str]) -> list[PayoutReceipt]:
    """Submit one payout per event. Stops at the first refusal but keeps what was already paid."""
    receipts: list[PayoutReceipt] = []
    for ev in events:
        try:
            receipts.append(rail.submit(PayoutRequest(
                event_id=ev.id, amount_usd=float(amount), receiver=receiver,
                note=f"PayGuard Arena sandbox sample (event {ev.id})")))
        except Exception as exc:  # LiveGuardError, PayPalError, network: stop paying, keep the rest
            warnings.append(f"payout for event {ev.id} failed ({_describe(exc)}); no further payouts sent")
            break
    return receipts


def poll_batches(rail, receipts: list[PayoutReceipt], poll_seconds: float, warnings: list[str], *,
                 sleep: Callable[[float], None], clock: Callable[[], float]) -> list[PayoutReceipt]:
    """Refresh every receipt every 10 s until each batch is terminal or `poll_seconds` is used up."""
    current = list(receipts)
    deadline = clock() + poll_seconds
    while True:
        for i, rec in enumerate(current):
            if rec.batch_status in BATCH_TERMINAL:
                continue
            try:
                current[i] = rail.refresh(rec)
            except Exception as exc:
                warnings.append(f"refresh of {rec.payout_batch_id} failed ({_describe(exc)})")
        left = deadline - clock()
        if all(r.batch_status in BATCH_TERMINAL for r in current) or left <= 0:
            break
        sleep(min(REFRESH_EVERY_S, left))
    for r in current:
        if r.batch_status not in BATCH_TERMINAL:
            warnings.append(f"batch {r.payout_batch_id} still {r.batch_status} after {poll_seconds:g} s")
    return current


def _matches(ev: dict, batch_ids: set[str], sender_ids: set[str]) -> bool:
    return ev.get("payout_batch_id") in batch_ids or ev.get("sender_batch_id") in sender_ids


def _settled(events: list[dict]) -> bool:
    return any(e.get("batch_status") in BATCH_TERMINAL or e.get("item_status") in ITEM_TERMINAL
               for e in events)


def collect_webhooks(http: httpx.Client, render_url: str, receipts: list[PayoutReceipt],
                     wait_seconds: float, warnings: list[str], *,
                     sleep: Callable[[float], None], clock: Callable[[], float]) -> list[dict]:
    """Poll the deployed app for verified webhook events of our batches (every 15 s, up to `wait_seconds`).

    Stops early once every batch has a settled event. Never raises: an unreachable or unconfigured
    server just yields fewer (or zero) events and a warning.
    """
    if not receipts:
        return []
    url = render_url.rstrip("/") + "/api/webhooks/recent"
    deadline = clock() + wait_seconds
    found: dict[str, dict] = {}
    last_problem = "no response"
    first = True
    while True:
        try:
            resp = http.get(url, params={"limit": 200},
                            timeout=FIRST_WEBHOOK_TIMEOUT_S if first else WEBHOOK_TIMEOUT_S)
            resp.raise_for_status()
            body = resp.json()
            if body.get("configured") is False:
                warnings.append("the server has no PAYPAL_WEBHOOK_ID configured; no webhooks can arrive")
                last_problem = ""
                break
            batch_ids = {r.payout_batch_id for r in receipts}
            sender_ids = {r.sender_batch_id for r in receipts}
            for ev in body.get("events") or []:
                if (isinstance(ev, dict) and isinstance(ev.get("event_id"), str)
                        and isinstance(ev.get("event_type"), str)
                        and isinstance(ev.get("received_at"), str) and _matches(ev, batch_ids, sender_ids)):
                    found[ev["event_id"]] = ev
            last_problem = ""
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            last_problem = type(exc).__name__
        first = False
        all_settled = all(
            _settled([e for e in found.values() if _matches(e, {r.payout_batch_id}, {r.sender_batch_id})])
            for r in receipts)
        left = deadline - clock()
        if all_settled or left <= 0:
            break
        sleep(min(WEBHOOK_EVERY_S, left))
    if last_problem:
        warnings.append(f"could not read verified webhooks from {render_url} ({last_problem})")
    return sorted(found.values(), key=lambda e: e["received_at"])


def build_out_replay(replay: Replay, receipts: list[PayoutReceipt], webhooks: list[dict]) -> dict:
    """The input replay plus the live receipts and the matched webhooks, validated before it is returned."""
    data = replay.model_dump(mode="json")
    data["payouts"] += [{
        "event_id": r.event_id, "mode": "live", "sender_batch_id": r.sender_batch_id,
        "payout_batch_id": r.payout_batch_id, "batch_status": r.batch_status,
        "item_status": r.item_status, "amount_usd": r.amount_usd, "created_at": r.created_at,
    } for r in receipts]
    keep = ("event_id", "event_type", "received_at", "payout_batch_id", "sender_batch_id",
            "batch_status", "item_status")
    known = {w["event_id"] for w in data["verified_webhooks"]}
    data["verified_webhooks"] += [{k: e.get(k) for k in keep} for e in webhooks
                                  if e["event_id"] not in known]
    parse_replay(data, "output replay")  # raises ReplayError before anything is written
    return data


# ---- reattach: the replay was rebuilt after the live run ------------------------------------------
def reattach_problems(new: Replay, old: Replay) -> list[str]:
    """Why the live receipts in `old` must NOT be carried onto `new`; an empty list means it is safe.

    `old` is an earlier --out: the replay a live run was built on, plus its live payouts and webhooks.
    `new` is meant to be the same run's replay rebuilt later (for example with an extra result row), so
    its episode is recomputed and its payouts start empty. A receipt only means something next to the
    episode event it paid, so this decides what "the same run" is. Name the offending event id in each
    message; `reattach` joins them into the error.
    """
    problems = []
    for key in ("seed", "split", "defender"):
        was, now = getattr(old.episode, key), getattr(new.episode, key)
        if was != now:
            problems.append(f"episode {key} is {now!r} in the rebuild, {was!r} in the live file")
    live = [p for p in old.payouts if p.mode == "live"]
    if not live:
        problems.append("the live file has no live payouts (wrong --out?)")
    if problems:
        return problems  # another episode: per-event checks would only add noise
    was_events = {e.id: e for e in old.episode.events}
    now_events = {e.id: e for e in new.episode.events}
    for p in live:
        now, was = now_events.get(p.event_id), was_events.get(p.event_id)
        if now is None:
            problems.append(f"event {p.event_id} is not in the rebuilt episode")
        elif now.verdict != "RELEASE" or not now.paid:
            problems.append(f"event {p.event_id} is {now.verdict} (paid={now.paid}) in the rebuild, "
                            "not RELEASE + paid")
        elif was is not None and (was.worker_id, was.amount, was.t) != (now.worker_id, now.amount, now.t):
            # same id, different payout: the rebuild is not the deterministic replay the receipt was paid on
            problems.append(f"event {p.event_id} has another worker, amount or time in the rebuild")
    return problems


def reattach(new: Replay, old: Replay) -> dict:
    """`new` plus the live payouts and verified webhooks recorded in `old`. Pays nothing.

    Idempotent: receipts and webhooks `new` already has are not added twice.
    """
    problems = reattach_problems(new, old)
    if problems:
        raise ReplayError("cannot reattach the live receipts: " + "; ".join(problems))
    data, src = new.model_dump(mode="json"), old.model_dump(mode="json")
    have = {p["event_id"] for p in data["payouts"] if p["mode"] == "live"}
    data["payouts"] += [p for p in src["payouts"] if p["mode"] == "live" and p["event_id"] not in have]
    known = {w["event_id"] for w in data["verified_webhooks"]}
    data["verified_webhooks"] += [w for w in src["verified_webhooks"] if w["event_id"] not in known]
    parse_replay(data, "reattached replay")
    return data


def reattach_main(replay: Replay, replay_path: Path, out: Path) -> int:
    if not out.is_file():
        print(f"ERROR nothing to reattach: {out} does not exist (run --send first)", file=sys.stderr)
        return EXIT_USAGE
    try:
        data = reattach(replay, load_replay(out))
    except ReplayError as exc:  # `out` is left exactly as it was
        print(f"ERROR {exc}", file=sys.stderr)
        return EXIT_USAGE
    write_json_atomic(out, data)
    n = sum(p["mode"] == "live" for p in data["payouts"])
    print(f"Reattached {n} live payout(s) and {len(data['verified_webhooks'])} verified webhook(s) "
          f"onto {replay_path}; wrote {out}. Nothing was sent.")
    return EXIT_OK


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Compact, like the evolution export: the episode alone is about 1 MB, indenting would double it.
    tmp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def run_live(*, rail, receiver: str, http: httpx.Client, events: list[EpisodeEvent],
             amount: Decimal, render_url: str, poll_seconds: float, wait_webhooks: float,
             sleep: Callable[[float], None] = time.sleep,
             clock: Callable[[], float] = time.monotonic) -> RunResult:
    """Pay `events`, wait for them to settle, collect their verified webhooks. Side effects only via
    the injected `rail` and `http`."""
    result = RunResult()
    result.receipts = submit_all(rail, events, amount, receiver, result.warnings)
    result.receipts = poll_batches(rail, result.receipts, poll_seconds, result.warnings,
                                   sleep=sleep, clock=clock)
    result.webhooks = collect_webhooks(http, render_url, result.receipts, wait_webhooks,
                                       result.warnings, sleep=sleep, clock=clock)
    return result


# ---- command line ---------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m scripts.live_run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", type=Path, default=DEFAULT_REPLAY, help="input replay (default eval/replay.json)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output replay (default eval/replay_live.json)")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"payouts to send, 1..{MAX_COUNT}")
    ap.add_argument("--amount", default=DEFAULT_AMOUNT, help="USD per payout, at most the live cap")
    ap.add_argument("--render-url", default=DEFAULT_RENDER_URL)
    ap.add_argument("--poll-seconds", type=float, default=120, help="how long to poll PayPal for batch status")
    ap.add_argument("--wait-webhooks", type=float, default=180, help="how long to wait for verified webhooks")
    ap.add_argument("--send", action="store_true", help="really send (default is a dry run)")
    ap.add_argument("--reattach", action="store_true",
                    help="carry the live receipts already in --out onto a rebuilt --replay; sends nothing")
    return ap


def main(argv: list[str] | None = None, *, rail_factory: Callable[[], object] | None = None,
         receiver: str | None = None, http_factory: Callable[[], httpx.Client] | None = None,
         sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic) -> int:
    """`rail_factory`, `receiver`, `http_factory`, `sleep` and `clock` exist so tests can inject fakes."""
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.reattach and args.send:
        ap.error("--reattach never sends; drop --send")
    if not 1 <= args.count <= MAX_COUNT:
        ap.error(f"--count must be between 1 and {MAX_COUNT} (got {args.count})")
    try:
        amount = parse_amount(args.amount)
    except ValueError as exc:
        ap.error(str(exc))
    if args.poll_seconds < 0 or args.wait_webhooks < 0:
        ap.error("--poll-seconds and --wait-webhooks must not be negative")
    if not args.render_url.startswith(("https://", "http://")):
        ap.error("--render-url must start with https:// (or http://)")

    if not args.replay.is_file():
        print(f"ERROR replay file not found: {args.replay}\n"
              f"       pass --replay, e.g. --replay {SAMPLE_REPLAY}", file=sys.stderr)
        return EXIT_USAGE
    if args.out.resolve() == args.replay.resolve():
        print("ERROR --out must differ from --replay (the input is never overwritten)", file=sys.stderr)
        return EXIT_USAGE
    try:
        replay = load_replay(args.replay)
    except ReplayError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.reattach:
        return reattach_main(replay, args.replay, args.out)

    already_live ={p.event_id for p in replay.payouts if p.mode == "live"}
    events = select_live_events(replay.episode, args.count, exclude_ids=already_live)
    if not events:
        print("ERROR no eligible events (need RELEASE + paid events without a live payout)", file=sys.stderr)
        return EXIT_USAGE

    print(f"{'SEND' if args.send else 'DRY RUN'}: {len(events)} sandbox payout(s) of {amount} USD "
          f"(cap {payout_cap()} USD) from {args.replay}")
    for ev in events:
        print(f"  event {ev.id:>6}  {amount} USD  family={ev.family}")
    if not args.send:
        print("Dry run: nothing was sent and no credential or network was used. Add --send to run it.")
        return EXIT_OK

    # ---- from here on: credentials and network ----
    try:
        if rail_factory is None:
            from payguard.rail import LiveRail, ReceiptStore
            rail = LiveRail.from_env(store=ReceiptStore())
        else:
            rail = rail_factory()
        to = receiver if receiver is not None else require("PAYPAL_TEST_RECEIVER_EMAIL")["PAYPAL_TEST_RECEIVER_EMAIL"]
    except MissingConfig as exc:
        print(f"ERROR config: {exc}", file=sys.stderr)
        return EXIT_USAGE
    http = http_factory() if http_factory is not None else httpx.Client(follow_redirects=False)
    try:
        result = run_live(rail=rail, receiver=to, http=http, events=events, amount=amount,
                          render_url=args.render_url, poll_seconds=args.poll_seconds,
                          wait_webhooks=args.wait_webhooks, sleep=sleep, clock=clock)
    finally:
        http.close()

    if not result.receipts:
        for w in result.warnings:
            print(f"WARNING {w}", file=sys.stderr)
        print("ERROR no payout was sent; nothing written", file=sys.stderr)
        return EXIT_FAIL
    try:
        out = build_out_replay(replay, result.receipts, result.webhooks)
    except ReplayError as exc:  # should not happen; the paid receipts are still shown below
        print(f"ERROR {exc}", file=sys.stderr)
        for r in result.receipts:
            print(f"  PAID but not recorded: event {r.event_id} batch {r.payout_batch_id}", file=sys.stderr)
        return EXIT_FAIL
    write_json_atomic(args.out, out)

    print(f"\nSent {len(result.receipts)} of {len(events)} payout(s):")
    for r in result.receipts:
        print(f"  event {r.event_id:>6}  batch {r.payout_batch_id}  {r.batch_status}/{r.item_status}  "
              f"{r.amount_usd:.2f} USD")
    print(f"Verified webhooks matched: {len(result.webhooks)}")
    for w in result.warnings:
        print(f"WARNING {w}")
    print(f"Wrote {args.out}. Deploy it as the replay (PAYGUARD_REPLAY_PATH) to show these in the dashboard.")
    return EXIT_FAIL if len(result.receipts) < len(events) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
