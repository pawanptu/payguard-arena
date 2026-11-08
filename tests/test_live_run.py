"""scripts/live_run.py with a fake PayPal client behind a real LiveRail and a MockTransport 'Render'.
No network, no credentials, no sleeping."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from payguard.api.app import DEFAULT_REPLAY
from payguard.api.replay import EpisodeEvent, load_replay
from payguard.paypal_client import PayPalError
from payguard.rail import LiveRail, ReceiptStore
from scripts import live_run

EMAIL = "sb-receiver@example.com"
RENDER = "https://render.test"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("PAYGUARD_LIVE_MAX_PAYOUT_USD", raising=False)


class FakeTime:
    """Fake clock: sleeping advances it, so loops finish instantly."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class FakePayPal:
    def __init__(self, fail_after: int | None = None, settle_after_polls: int = 2):
        self.created, self.polls, self.fail_after = [], {}, fail_after
        self.settle_after_polls = settle_after_polls

    def create_payout(self, *, sender_batch_id, items, **_):
        if self.fail_after is not None and len(self.created) >= self.fail_after:
            raise PayPalError(400, '{"secret":"body"}', "dbg-1")
        self.created.append((sender_batch_id, items))
        return {"batch_header": {"payout_batch_id": f"PB{len(self.created)}", "batch_status": "PENDING"}}

    def get_payout(self, batch_id):
        n = self.polls[batch_id] = self.polls.get(batch_id, 0) + 1
        done = n >= self.settle_after_polls
        return {"batch_header": {"batch_status": "SUCCESS" if done else "PROCESSING"},
                "items": [{"transaction_status": "SUCCESS" if done else "PENDING"}]}


def rail_for(paypal, **kw):
    return lambda: LiveRail(paypal, allowed_receivers={EMAIL}, store=ReceiptStore(), **kw)


def webhook_event(i, batch, **kw):
    return {"event_id": f"WH{i}", "event_type": "PAYMENT.PAYOUTSBATCH.SUCCESS",
            "received_at": f"2026-11-07T12:00:{i:02d}Z", "payout_batch_id": batch, "sender_batch_id": None,
            "batch_status": "SUCCESS", "item_status": None, "outcome": "processed", "source": "runtime", **kw}


def render(handler):
    calls = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    return (lambda: httpx.Client(transport=httpx.MockTransport(wrapped))), calls


@pytest.fixture
def replay_file(tmp_path) -> Path:
    p = tmp_path / "replay.json"
    p.write_text(DEFAULT_REPLAY.read_text(encoding="utf-8"), encoding="utf-8")
    return p


def run(argv, replay_file, tmp_path, *, paypal=None, http_factory=None, ft=None, **kw):
    ft = ft or FakeTime()
    out = tmp_path / "out.json"
    code = live_run.main(
        ["--replay", str(replay_file), "--out", str(out), "--render-url", RENDER, *argv],
        rail_factory=rail_for(paypal or FakePayPal()), receiver=EMAIL,
        http_factory=http_factory, sleep=ft.sleep, clock=ft.clock, **kw)
    return code, out


def expected_ids(replay_file, n):
    r = load_replay(replay_file)
    already = {p.event_id for p in r.payouts if p.mode == "live"}  # the fixture's live event is skipped
    return [e.id for e in live_run.select_live_events(r.episode, n, exclude_ids=already)]


# ---- selection -------------------------------------------------------------------------------------
def worry(e) -> float:
    return max(e.signals.values()) if e.signals else 0.0


def test_select_live_events_honest_most_worried_first(replay_file):
    ep = load_replay(replay_file).episode
    eligible = [e for e in ep.events if e.verdict == "RELEASE" and e.paid]
    picked = live_run.select_live_events(ep, 3)
    assert len(picked) == 3
    assert all(e.verdict == "RELEASE" and e.paid for e in picked)
    # Independent restatement of the policy: honest before attack, then most worried, then episode order.
    expect = [e.id for _, e in sorted(enumerate(eligible), key=lambda p: (p[1].is_attack, -worry(p[1]), p[0]))][:3]
    assert [e.id for e in picked] == expect
    assert [e.id for e in live_run.select_live_events(ep, 2, exclude_ids={expect[0]})] == [expect[1], expect[2]]
    assert live_run.select_live_events(ep, 0) == []


def test_demo_priority_honest_first_then_most_worried_then_episode_order():
    def ev(i, attack, signals):
        return EpisodeEvent(id=i, t=float(i), worker_id=1, requester_id=1, amount=10.0, verdict="RELEASE",
                            paid=True, is_attack=attack, family="ring" if attack else "honest", signals=signals)
    events = [ev(1, True, {"graph": 0.9}), ev(2, False, {"graph": 0.1, "velocity": 0.4}), ev(3, False, None),
              ev(4, False, {"graph": 0.4}), ev(5, False, {"policy": 0.7, "memory": 0.2})]
    assert [e.id for e in sorted(events, key=live_run.demo_priority)] == [5, 2, 4, 3, 1]


# ---- dry run and argument limits -------------------------------------------------------------------
def test_dry_run_prints_the_plan_and_touches_nothing(replay_file, tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dry run must not reach for credentials, rails, network or time")

    monkeypatch.setattr(live_run, "require", boom)
    out = tmp_path / "out.json"
    code = live_run.main(["--replay", str(replay_file), "--out", str(out), "--count", "2", "--amount", "1.5"],
                         rail_factory=boom, http_factory=boom, sleep=boom, clock=boom)
    text = capsys.readouterr().out
    assert code == 0 and not out.exists()
    ids = expected_ids(replay_file, 2)
    assert text.count("event ") >= 2 and all(f"{i}" in text for i in ids)
    assert "1.50 USD" in text and "family=" in text and "Dry run" in text


@pytest.mark.parametrize("count", ["6", "0", "-1", "100"])
def test_count_above_five_or_below_one_is_refused(replay_file, tmp_path, count, capsys):
    with pytest.raises(SystemExit) as exc:
        run(["--count", count, "--send"], replay_file, tmp_path)
    assert exc.value.code == 2 and "--count" in capsys.readouterr().err


@pytest.mark.parametrize("amount", ["5.01", "100", "0", "-1", "abc", "nan", "inf"])
def test_amount_above_the_cap_or_invalid_is_refused(replay_file, tmp_path, amount, capsys):
    paypal = FakePayPal()
    with pytest.raises(SystemExit) as exc:
        run(["--amount", amount, "--send"], replay_file, tmp_path, paypal=paypal)
    assert exc.value.code == 2 and "--amount" in capsys.readouterr().err
    assert paypal.created == []


def test_amount_at_the_cap_is_allowed_and_a_lower_env_cap_applies(replay_file, tmp_path, monkeypatch):
    assert run(["--amount", "5.00"], replay_file, tmp_path)[0] == 0  # dry run
    monkeypatch.setenv("PAYGUARD_LIVE_MAX_PAYOUT_USD", "2.00")
    with pytest.raises(SystemExit):
        run(["--amount", "3"], replay_file, tmp_path)


def test_missing_replay_suggests_the_sample(tmp_path, capsys):
    code = live_run.main(["--replay", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o.json")])
    err = capsys.readouterr().err
    assert code == 2 and "not found" in err and "sample_replay.json" in err


def test_broken_replay_and_same_in_out_are_refused(replay_file, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert live_run.main(["--replay", str(bad), "--out", str(tmp_path / "o.json")]) == 2
    assert "schema_version" in capsys.readouterr().err
    assert live_run.main(["--replay", str(replay_file), "--out", str(replay_file)]) == 2
    assert "--out must differ" in capsys.readouterr().err


def test_bad_render_url_is_refused(replay_file, tmp_path):
    with pytest.raises(SystemExit):
        live_run.main(["--replay", str(replay_file), "--render-url", "ftp://x"])


# ---- full fake run ---------------------------------------------------------------------------------
def test_full_run_writes_a_valid_replay_with_live_payouts_and_matched_webhooks(replay_file, tmp_path, capsys):
    paypal = FakePayPal()
    batches = ["PB1", "PB2", "PB3"]
    # call 1: Render only knows an early event for PB1; call 2 onward: every batch is settled
    other = webhook_event(99, "PB-SOMEONE-ELSE")

    def handler(request, n):
        assert request.url.path == "/api/webhooks/recent" and request.url.params["limit"] == "200"
        evs = [other, webhook_event(1, "PB1", batch_status="PROCESSING")] if n == 1 else \
            [other] + [webhook_event(i + 1, b) for i, b in enumerate(batches)]
        return httpx.Response(200, json={"configured": True, "events": evs})

    factory, calls = render(handler)
    ft = FakeTime()
    code, out = run(["--send", "--count", "3"], replay_file, tmp_path, paypal=paypal, http_factory=factory, ft=ft)
    text = capsys.readouterr().out
    assert code == 0 and len(calls) == 2

    original = load_replay(replay_file)
    result = load_replay(out)  # valid against the contract
    assert result.schema_version == 1
    new = result.payouts[len(original.payouts):]
    assert [p.mode for p in new] == ["live"] * 3
    assert [p.event_id for p in new] == expected_ids(replay_file, 3)
    assert [p.payout_batch_id for p in new] == batches
    assert all(p.batch_status == "SUCCESS" and p.item_status == "SUCCESS" and p.amount_usd == 1.0 for p in new)
    assert result.payouts[:len(original.payouts)] == original.payouts
    assert {w.payout_batch_id for w in result.verified_webhooks} == set(batches)
    assert [w.event_id for w in result.verified_webhooks] == ["WH1", "WH2", "WH3"]  # the unrelated one is dropped
    assert all(not hasattr(w, "outcome") for w in result.verified_webhooks)
    # PayPal was polled until settled, never beyond; sender ids are unique
    assert paypal.polls == {"PB1": 2, "PB2": 2, "PB3": 2}
    assert len({sb for sb, _ in paypal.created}) == 3
    assert all(items[0]["amount"]["value"] == "1.00" for _, items in paypal.created)
    # summary: batch ids, statuses, webhook count; no receiver
    assert "PB1" in text and "SUCCESS/SUCCESS" in text and "Verified webhooks matched: 3" in text
    assert EMAIL not in text and EMAIL not in out.read_text(encoding="utf-8")


def test_poll_timeout_is_a_warning_and_the_replay_is_still_written(replay_file, tmp_path, capsys):
    paypal = FakePayPal(settle_after_polls=10_000)  # never settles
    factory, _ = render(lambda r, n: httpx.Response(200, json={"configured": True, "events": []}))
    ft = FakeTime()
    code, out = run(["--send", "--count", "1", "--poll-seconds", "25", "--wait-webhooks", "30"],
                    replay_file, tmp_path, paypal=paypal, http_factory=factory, ft=ft)
    text = capsys.readouterr().out
    assert code == 0 and "still PROCESSING after 25 s" in text
    assert load_replay(out).payouts[-1].batch_status == "PROCESSING"
    assert paypal.polls["PB1"] == 4  # t = 0, 10, 20, 25
    assert ft.sleeps[:3] == [10.0, 10.0, 5.0]


def test_render_unreachable_still_writes_the_replay_with_a_warning(replay_file, tmp_path, capsys):
    def handler(request, n):
        raise httpx.ConnectError("down")

    factory, calls = render(handler)
    code, out = run(["--send", "--count", "2", "--wait-webhooks", "45"], replay_file, tmp_path,
                    http_factory=factory)
    text = capsys.readouterr().out
    assert code == 0 and len(calls) == 4  # t = 0, 15, 30, 45
    result = load_replay(out)
    assert result.verified_webhooks == [] and sum(p.mode == "live" for p in result.payouts) == 3  # 1 + 2 new
    assert "WARNING could not read verified webhooks" in text and "ConnectError" in text
    assert "Verified webhooks matched: 0" in text


def test_server_errors_and_bad_json_from_render_are_survivable(replay_file, tmp_path, capsys):
    def handler(request, n):
        return httpx.Response(502) if n % 2 else httpx.Response(200, content=b"<html>not json</html>")

    factory, _ = render(handler)
    code, out = run(["--send", "--count", "1", "--wait-webhooks", "30"], replay_file, tmp_path, http_factory=factory)
    assert code == 0 and load_replay(out).verified_webhooks == []
    assert "WARNING could not read verified webhooks" in capsys.readouterr().out


def test_unconfigured_server_stops_waiting_immediately(replay_file, tmp_path, capsys):
    factory, calls = render(lambda r, n: httpx.Response(200, json={"configured": False, "events": []}))
    ft = FakeTime()
    code, out = run(["--send", "--count", "1"], replay_file, tmp_path, http_factory=factory, ft=ft)
    text = capsys.readouterr().out
    assert code == 0 and len(calls) == 1 and "PAYPAL_WEBHOOK_ID" in text
    assert load_replay(out).verified_webhooks == []


def test_guard_refusal_midway_keeps_what_was_paid(replay_file, tmp_path, capsys):
    paypal2 = FakePayPal()  # the rail gets a 1.50 USD budget: the first 1.00 payout fits, the second does not
    ft = FakeTime()
    out2 = tmp_path / "out2.json"
    code = live_run.main(["--replay", str(replay_file), "--out", str(out2), "--render-url", RENDER, "--send",
                          "--count", "3", "--wait-webhooks", "0"],
                         rail_factory=rail_for(paypal2, budget_usd="1.50"), receiver=EMAIL,
                         http_factory=lambda: httpx.Client(transport=httpx.MockTransport(
                             lambda r: httpx.Response(200, json={"configured": True, "events": []}))),
                         sleep=ft.sleep, clock=ft.clock)
    text = capsys.readouterr().out
    assert code == 1 and len(paypal2.created) == 1  # partial run reported as a failure
    assert "Sent 1 of 3" in text and "exceed the live budget" in text and "no further payouts" in text
    assert sum(p.mode == "live" for p in load_replay(out2).payouts) == 2  # fixture's 1 + the 1 paid


def test_first_payout_rejected_writes_nothing(replay_file, tmp_path, capsys):
    paypal = FakePayPal(fail_after=0)
    factory, calls = render(lambda r, n: httpx.Response(200, json={"configured": True, "events": []}))
    code, out = run(["--send", "--count", "2"], replay_file, tmp_path, paypal=paypal, http_factory=factory)
    err = capsys.readouterr().err
    assert code == 1 and not out.exists() and calls == []  # nothing paid: Render is never polled
    assert "PayPal HTTP 400 (debug_id=dbg-1)" in err and "secret" not in err  # never the response body


def test_rerun_on_its_own_output_does_not_pay_the_same_events_again(replay_file, tmp_path, capsys):
    factory = lambda: httpx.Client(transport=httpx.MockTransport(  # noqa: E731
        lambda r: httpx.Response(200, json={"configured": True, "events": []})))
    code, out = run(["--send", "--count", "2", "--wait-webhooks", "0"], replay_file, tmp_path, http_factory=factory)
    assert code == 0
    first_ids = expected_ids(replay_file, 2)
    paypal = FakePayPal()
    out2 = tmp_path / "out_again.json"
    ft = FakeTime()
    live_run.main(["--replay", str(out), "--out", str(out2), "--render-url", RENDER, "--send", "--count", "2",
                   "--wait-webhooks", "0"], rail_factory=rail_for(paypal), receiver=EMAIL, http_factory=factory,
                  sleep=ft.sleep, clock=ft.clock)
    paid_again = [int(items[0]["sender_item_id"].split("-")[1]) for _, items in paypal.created]
    assert paid_again and not set(paid_again) & set(first_ids)


def test_missing_credentials_exit_cleanly_without_a_traceback(replay_file, tmp_path, capsys, monkeypatch):
    from payguard.config import MissingConfig

    def no_creds():
        raise MissingConfig("Missing environment variables: PAYPAL_CLIENT_ID (see .env.example)")

    code = live_run.main(["--replay", str(replay_file), "--out", str(tmp_path / "o.json"), "--send"],
                         rail_factory=no_creds, receiver=EMAIL)
    err = capsys.readouterr().err
    assert code == 2 and "PAYPAL_CLIENT_ID" in err and not (tmp_path / "o.json").exists()


def test_output_json_is_plain_replay_json(replay_file, tmp_path):
    factory = lambda: httpx.Client(transport=httpx.MockTransport(  # noqa: E731
        lambda r: httpx.Response(200, json={"configured": True, "events": []})))
    _, out = run(["--send", "--count", "1", "--wait-webhooks", "0"], replay_file, tmp_path, http_factory=factory)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and "verified_webhooks" in data
    assert not list(tmp_path.glob("*.tmp"))


# ---- reattach: carry receipts onto a rebuilt replay, never pay again --------------------------------
def live_output(replay_file, tmp_path) -> Path:
    """A finished fake live run on the fixture (2 new payouts, PB1 and PB2, each with a webhook)."""
    factory, _ = render(lambda r, n: httpx.Response(200, json={
        "configured": True, "events": [webhook_event(1, "PB1"), webhook_event(2, "PB2")]}))
    code, out = run(["--send", "--count", "2"], replay_file, tmp_path, http_factory=factory)
    assert code == 0
    return out


def rebuilt(replay_file, tmp_path, edit=lambda d: None) -> Path:
    """The fixture 'rebuilt' after the live run: one extra result row, no live payouts or webhooks."""
    d = json.loads(replay_file.read_text(encoding="utf-8"))
    d["results"].append({**d["results"][0], "defender": "single_late"})
    d["payouts"] = [p for p in d["payouts"] if p["mode"] != "live"]
    d["verified_webhooks"] = []
    edit(d)
    p = tmp_path / "rebuilt.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def reattach(new_path, out):
    def boom():
        raise AssertionError("--reattach must not touch PayPal or Render")
    return live_run.main(["--replay", str(new_path), "--out", str(out), "--reattach"],
                         rail_factory=boom, http_factory=boom)


def test_reattach_carries_the_receipts_onto_a_rebuilt_replay_and_sends_nothing(replay_file, tmp_path, capsys):
    out = live_output(replay_file, tmp_path)
    old = load_replay(out)
    capsys.readouterr()
    assert reattach(rebuilt(replay_file, tmp_path), out) == 0
    result = load_replay(out)
    assert any(r.defender == "single_late" for r in result.results)  # the rebuilt rows won
    assert [p for p in result.payouts if p.mode == "live"] == [p for p in old.payouts if p.mode == "live"]
    assert result.verified_webhooks == old.verified_webhooks
    assert "Nothing was sent" in capsys.readouterr().out


def test_reattach_twice_changes_nothing(replay_file, tmp_path):
    out = live_output(replay_file, tmp_path)
    new_path = rebuilt(replay_file, tmp_path)
    assert reattach(new_path, out) == 0
    first = out.read_text(encoding="utf-8")
    assert reattach(new_path, out) == 0
    assert out.read_text(encoding="utf-8") == first


@pytest.mark.parametrize("field", ["seed", "split", "defender"])
def test_reattach_refuses_a_replay_of_another_episode(replay_file, tmp_path, capsys, field):
    out = live_output(replay_file, tmp_path)
    before = out.read_text(encoding="utf-8")
    other = (lambda v: v + 1) if field == "seed" else (lambda v: v + "_other")
    new_path = rebuilt(replay_file, tmp_path, lambda d: d["episode"].__setitem__(field, other(d["episode"][field])))
    assert reattach(new_path, out) == 2
    assert "cannot reattach" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == before  # untouched


@pytest.mark.parametrize("change", ["held", "missing"])
def test_reattach_refuses_when_a_paid_event_is_not_released_and_paid_in_the_rebuild(replay_file, tmp_path,
                                                                                    capsys, change):
    out = live_output(replay_file, tmp_path)
    paid = [p.event_id for p in load_replay(out).payouts if p.mode == "live"][-1]

    def edit(d):
        events = d["episode"]["events"]
        if change == "missing":
            d["episode"]["events"] = [e for e in events if e["id"] != paid]
        else:
            for e in events:
                if e["id"] == paid:
                    e.update(verdict="HOLD", paid=False, paid_at=None)

    before = out.read_text(encoding="utf-8")
    assert reattach(rebuilt(replay_file, tmp_path, edit), out) == 2
    err = capsys.readouterr().err
    assert "cannot reattach" in err and str(paid) in err
    assert out.read_text(encoding="utf-8") == before


def test_reattach_without_an_earlier_output_or_with_send_is_refused(replay_file, tmp_path, capsys):
    assert reattach(replay_file, tmp_path / "nope.json") == 2
    assert "nothing to reattach" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        live_run.main(["--replay", str(replay_file), "--out", str(tmp_path / "o.json"), "--reattach", "--send"])
    assert exc.value.code == 2


def test_reattach_refuses_a_paid_event_that_became_another_payout(replay_file, tmp_path, capsys):
    out = live_output(replay_file, tmp_path)
    paid = [p.event_id for p in load_replay(out).payouts if p.mode == "live"][0]

    def edit(d):
        for e in d["episode"]["events"]:
            if e["id"] == paid:
                e["amount"] += 1.0  # same id, still RELEASE + paid, but not the payout the receipt was for

    assert reattach(rebuilt(replay_file, tmp_path, edit), out) == 2
    assert f"event {paid} has another worker, amount or time" in capsys.readouterr().err


def test_reattach_from_a_file_without_live_payouts_is_refused(replay_file, tmp_path, capsys):
    empty = rebuilt(replay_file, tmp_path)  # no live payouts at all
    out = tmp_path / "empty_out.json"
    out.write_text(empty.read_text(encoding="utf-8"), encoding="utf-8")
    assert reattach(empty, out) == 2
    assert "no live payouts" in capsys.readouterr().err
