"""Agent loop with a scripted fake LLM, recording format (contract C), and the CLI's setup paths. No network."""
import json

import pytest

from payguard.copilot.agent import FINAL_SCHEMA, STEP_SCHEMA, Copilot, truncate
from payguard.copilot.record import (
    MAX_DATA_CHARS,
    PREVIEW_CHARS,
    RecordError,
    build_session,
    validate_session,
    validate_slug,
    write_session,
)
from payguard.copilot.tooling import SOURCE_CONSOLE, SOURCE_TOOLKIT, Tool, ToolError


class FakeLLM:
    """Replays scripted replies; records the messages and schema of every call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete_json(self, messages, schema):
        self.calls.append((json.loads(json.dumps(messages)), schema))
        return self.script.pop(0)


def call(tool, args="{}", thought="t"):
    return {"thought": thought, "action": "call_tool", "tool": tool, "args_json": args, "answer": ""}


def final(text):
    return {"thought": "done", "action": "final_answer", "tool": "", "args_json": "{}", "answer": text}


def make_tools():
    seen = []

    def echo(args):
        seen.append(args)
        return {"echo": args, "email": "ops@example.com", "payer_info": {"x": 1}}

    def boom(args):
        raise RuntimeError("secret-token-abc")

    def nope(args):
        raise ToolError("PayPal sandbox: NOT_AUTHORIZED (this app lacks the permission for this API)")

    def big(args):
        return {"rows": ["x" * 100] * 200}

    tools = [Tool("echo", SOURCE_CONSOLE, "echoes", {"type": "object", "properties": {}}, echo),
             Tool("boom", SOURCE_CONSOLE, "crashes", {"type": "object", "properties": {}}, boom),
             Tool("list_transactions", SOURCE_TOOLKIT, "403s", {"type": "object", "properties": {}}, nope),
             Tool("big", SOURCE_CONSOLE, "large", {"type": "object", "properties": {}}, big)]
    return tools, seen


def test_happy_path_turns_follow_contract_c():
    tools, seen = make_tools()
    llm = FakeLLM([call("echo", '{"event_id": 24}'), final("Held because of the graph (echo).")])
    res = Copilot(llm, tools).ask("Why was #24 held?")
    assert res.answer == "Held because of the graph (echo)."
    t = res.turns
    assert [x["role"] for x in t] == ["user", "tool_call", "tool_result", "assistant"]
    assert t[1] == {"role": "tool_call", "step": 1, "tool": "echo", "source": SOURCE_CONSOLE, "args": {"event_id": 24}}
    assert t[2]["step"] == 1 and t[2]["tool"] == "echo" and t[2]["ok"] is True and isinstance(t[2]["ms"], int)
    assert seen == [{"event_id": 24}]
    # result is redacted before it is recorded
    assert t[2]["data"]["email"] == "[redacted]" and t[2]["data"]["payer_info"] == "[redacted]"
    assert res.steps == 2 and res.protocol_errors == 0 and not res.forced_final


def test_model_sees_redacted_observation_and_system_prompt_rules():
    tools, _ = make_tools()
    llm = FakeLLM([call("echo", '{"a": 1}'), final("ok")])
    Copilot(llm, tools).ask("q")
    system = llm.calls[0][0][0]
    assert system["role"] == "system"
    for phrase in ("READ-ONLY", "SIMULATED", "which tool", "NOT_AUTHORIZED", "never release", "echo [source: payguard-console]"):
        assert phrase.lower() in system["content"].lower()
    second = llm.calls[1][0]
    obs = second[-1]
    assert obs["role"] == "user" and obs["content"].startswith("OBSERVATION step 1 tool echo")
    assert "ops@example.com" not in obs["content"] and "[redacted]" in obs["content"]
    assert llm.calls[0][1] is STEP_SCHEMA


def test_tool_error_403_is_a_failed_result_and_the_loop_continues():
    tools, _ = make_tools()
    llm = FakeLLM([call("list_transactions"), final("The sandbox app is not authorised for Transaction Search.")])
    res = Copilot(llm, tools).ask("q")
    r = res.turns[2]
    assert r["ok"] is False and r["error"].startswith("PayPal sandbox: NOT_AUTHORIZED") and "data" not in r
    assert "NOT_AUTHORIZED" in llm.calls[1][0][-1]["content"]


def test_crashing_tool_never_leaks_the_exception_message():
    tools, _ = make_tools()
    llm = FakeLLM([call("boom"), final("failed")])
    res = Copilot(llm, tools).ask("q")
    assert res.turns[2]["ok"] is False and res.turns[2]["error"] == "tool failed (RuntimeError)"
    assert "secret-token-abc" not in json.dumps(res.turns) + json.dumps(llm.calls)


def test_bad_args_json_recovers_and_is_not_a_turn():
    tools, seen = make_tools()
    llm = FakeLLM([call("echo", "{bad json"), call("echo", "[1, 2]"), call("echo", '{"ok": true}'), final("done")])
    res = Copilot(llm, tools).ask("q")
    assert seen == [{"ok": True}]  # only the valid call ran
    assert res.protocol_errors == 2
    assert [x["role"] for x in res.turns] == ["user", "tool_call", "tool_result", "assistant"]
    assert res.turns[1]["step"] == 1
    err_msg = llm.calls[1][0][-1]["content"]
    assert err_msg.startswith("ERROR:") and "args_json" in err_msg


def test_empty_args_json_means_no_arguments():
    tools, seen = make_tools()
    Copilot(FakeLLM([call("echo", ""), final("x")]), tools).ask("q")
    assert seen == [{}]


def test_unknown_tool_recovers_and_lists_allowed_tools():
    tools, seen = make_tools()
    llm = FakeLLM([call("create_order", '{"amount": 5}'), call("", "{}"), final("I cannot create orders.")])
    res = Copilot(llm, tools).ask("create an order")
    assert res.protocol_errors == 2 and seen == []
    assert "echo" in llm.calls[1][0][-1]["content"] and "Allowed tools" in llm.calls[1][0][-1]["content"]
    assert not any(t["role"] == "tool_call" for t in res.turns)
    assert res.answer == "I cannot create orders."


def test_empty_final_answer_is_a_protocol_error():
    tools, _ = make_tools()
    res = Copilot(FakeLLM([final("  "), final("real")]), tools).ask("q")
    assert res.answer == "real" and res.protocol_errors == 1


def test_stops_at_max_steps_and_forces_a_final_answer():
    tools, seen = make_tools()
    llm = FakeLLM([call("echo", '{"n": 1}'), call("echo", '{"n": 2}'), call("echo", '{"n": 3}'), final("Forced summary.")])
    res = Copilot(llm, tools, max_steps=3).ask("q")
    assert len(seen) == 3  # exactly max_steps tool calls, no more
    assert res.forced_final and res.answer == "Forced summary." and res.steps == 4
    assert llm.calls[-1][1] is FINAL_SCHEMA
    assert "Step limit reached (3)" in llm.calls[-1][0][-1]["content"]
    assert res.turns[-1] == {"role": "assistant", "text": "Forced summary."}


def test_forced_final_with_stubborn_model_uses_fallback_text():
    tools, _ = make_tools()
    res = Copilot(FakeLLM([call("echo"), call("echo")]), tools, max_steps=1).ask("q")
    assert res.forced_final and "could not finish within 1 steps" in res.answer


def test_large_results_are_truncated_for_the_model_only():
    tools, _ = make_tools()
    llm = FakeLLM([call("big"), final("ok")])
    res = Copilot(llm, tools, max_result_chars=500).ask("q")
    obs = llm.calls[1][0][-1]["content"]
    assert "[truncated" in obs and len(obs) < 800
    assert len(res.turns[2]["data"]["rows"]) == 200  # the transcript keeps the full redacted data
    assert truncate("abc", 10) == "abc" and truncate("abcdef", 3) == "abc...[truncated 3 chars]"


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError):
        Copilot(FakeLLM([]), [], max_steps=0)


# ------------------------------------------------------------------ recording (contract C)
def _session(turns=None, **over):
    tools, _ = make_tools()
    res = Copilot(FakeLLM([call("echo", '{"event_id": 24}'), call("list_transactions"), final("Answer.")]), tools).ask("Why?")
    kw = dict(session_id="why-held-24", title="Why was payout #24 held?", order=1, turns=turns or res.turns,
              llm_provider="gemini", llm_model="gemini-test", toolkit_version="1.11.0",
              enabled_tools=["list_transactions"])
    kw.update(over)
    return build_session(**kw)


def test_session_matches_contract_c():
    s = _session()
    assert list(s) == ["schema_version", "id", "order", "title", "recorded_at", "environment", "llm", "toolkit",
                       "redaction", "turns"]
    assert s["schema_version"] == 1 and s["environment"] == "paypal-sandbox" and s["order"] == 1
    assert s["llm"] == {"provider": "gemini", "model": "gemini-test"}
    assert s["toolkit"] == {"package": "paypal-agent-toolkit", "version": "1.11.0", "read_only": True,
                            "enabled_tools": ["list_transactions"]}
    assert s["redaction"].startswith("Emails, names, addresses")
    assert s["recorded_at"].endswith("+00:00")
    roles = [t["role"] for t in s["turns"]]
    assert roles == ["user", "tool_call", "tool_result", "tool_call", "tool_result", "assistant"]
    assert s["turns"][4]["ok"] is False and isinstance(s["turns"][4]["error"], str)
    validate_session(s)


def test_write_refuses_overwrite_without_force_and_writes_valid_json(tmp_path):
    s = _session()
    p = write_session(s, sessions_dir=tmp_path)
    assert p == tmp_path / "why-held-24.json"
    assert json.loads(p.read_text(encoding="utf-8")) == s
    with pytest.raises(RecordError, match="already exists"):
        write_session(s, sessions_dir=tmp_path)
    s2 = _session(title="New title")
    write_session(s2, sessions_dir=tmp_path, force=True)
    assert json.loads(p.read_text(encoding="utf-8"))["title"] == "New title"
    assert [f.name for f in tmp_path.iterdir()] == ["why-held-24.json"]  # no temp file left behind


@pytest.mark.parametrize("bad", ["", "Why-Held", "a/b", "../x", "a b", "-x", "x-", "a" * 70, "x.json"])
def test_bad_slugs_are_refused(bad, tmp_path):
    with pytest.raises(RecordError):
        validate_slug(bad)
    with pytest.raises(RecordError):
        _session(session_id=bad)


def test_validate_session_catches_broken_turns():
    s = _session()
    broken = json.loads(json.dumps(s))
    broken["turns"][2]["step"] = 9
    with pytest.raises(RecordError, match="does not match"):
        validate_session(broken)
    broken = json.loads(json.dumps(s))
    broken["turns"][1]["source"] = "somewhere-else"
    with pytest.raises(RecordError, match="source"):
        validate_session(broken)
    broken = json.loads(json.dumps(s))
    broken["turns"].pop()
    with pytest.raises(RecordError, match="end with an assistant"):
        validate_session(broken)


def test_huge_tool_data_is_capped_in_the_recording():
    turns = [{"role": "user", "text": "q"},
             {"role": "tool_call", "step": 1, "tool": "big", "source": SOURCE_CONSOLE, "args": {}},
             {"role": "tool_result", "step": 1, "tool": "big", "ok": True, "ms": 1, "data": {"r": "x" * (MAX_DATA_CHARS + 5)}},
             {"role": "assistant", "text": "a"}]
    s = _session(turns=turns)
    d = s["turns"][2]["data"]
    assert d["truncated"] is True and len(d["preview"]) == PREVIEW_CHARS


# ------------------------------------------------------------------ CLI setup paths (never reaches the network)
def test_cli_check_and_tools_report_missing_setup_without_network(monkeypatch, capsys):
    from scripts import copilot as cli

    for name in ("PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET", "PAYPAL_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert cli.cmd_check(None) == 2
    assert "PAYPAL_CLIENT_ID" in capsys.readouterr().out
    assert cli.cmd_tools(None) == 2


def test_cli_prints_transcript_readably(capsys):
    from scripts import copilot as cli

    s = _session()
    cli._print_transcript(s["turns"])
    out = capsys.readouterr().out
    assert "QUESTION: Why?" in out and "[step 1] CALL   echo  source=payguard-console" in out
    assert "ok=False" in out and "NOT_AUTHORIZED" in out and "ANSWER:\nAnswer." in out


def test_cli_rejects_bad_record_id_before_doing_anything(capsys):
    from scripts import copilot as cli

    ns = type("NS", (), {"record": "Bad Id", "question": "q"})()
    assert cli.cmd_ask(ns) == 2
    assert "slug" in capsys.readouterr().out
