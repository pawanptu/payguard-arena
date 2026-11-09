"""Console tools (local data, tmp files) and the read-only Payouts REST tool (fake client). No network."""
import json

import pytest

from payguard.copilot.console_tools import NOT_BUILT, ConsoleTools
from payguard.copilot.rest_tools import get_payout_batch_tool
from payguard.copilot.tooling import SOURCE_CONSOLE, SOURCE_REST, ToolError
from payguard.paypal_client import PayPalError


def _case(i, verdict, risk, share):
    return {"event_id": i, "t": 1.5, "worker_id": 160, "requester_id": 0, "amount": 99.47, "verdict": verdict,
            "risk": risk, "decided_by": "swarm", "scores": {"graph": 1.0}, "weights": {"graph": 0.25},
            "vote_share": share, "layer": {"verdict": verdict, "fired": []}, "features": {"age_days": 3.2},
            "outcome": {"paid": True, "paid_at": 25.0, "delay_h": 24.0},
            "truth": {"is_attack": False, "family": "honest"}}


QUEUE = {
    "schema_version": 1, "simulated": True, "source": {"split": "replay-split"},
    "thresholds": {"step_up": 0.3, "hold": 0.6},
    "counts": {"events": 10, "RELEASE": 7, "STEP_UP": 1, "HOLD": 2, "cases": 3, "cases_attack": 2, "cases_honest": 1,
               "attack_usd_total": 5.0, "attack_usd_stopped": 3.0, "attack_usd_leaked": 2.0},
    "specialists": [{"key": "graph", "label": "Graph", "what": "..."}],
    "cases": [_case(24, "HOLD", 0.71, {"graph": 1.0, "velocity": 0.0}), _case(7, "STEP_UP", 0.35, {"graph": 0.0}),
              _case(9, "HOLD", 0.65, {"velocity": 0.7, "graph": 0.3})],
}
EXPL = {"explanations": {"24": {"source": "template", "summary": "Held because graph.", "evidence": ["Graph: x"],
                                "check_next": "Check.", "suggested_action": "HOLD"}}}
REPLAY = {"payouts": [
    {"event_id": 1823, "mode": "live", "sender_batch_id": "pg-1", "payout_batch_id": "U2UPGB9BXLFRY",
     "batch_status": "SUCCESS", "item_status": "SUCCESS", "amount_usd": 1.0, "created_at": "2026-11-08T02:58:53.894292Z"},
    {"event_id": 736, "payout_batch_id": "34VMJ8WBCCWEG", "batch_status": "SUCCESS", "amount_usd": 1.0,
     "created_at": "2026-11-08T02:58:54.405995Z"}]}


@pytest.fixture
def tools(tmp_path):
    (tmp_path / "queue.json").write_text(json.dumps(QUEUE))
    (tmp_path / "explanations.json").write_text(json.dumps(EXPL))
    (tmp_path / "replay_live.json").write_text(json.dumps(REPLAY))
    return ConsoleTools(tmp_path / "queue.json", tmp_path / "explanations.json", tmp_path / "replay_live.json")


def test_tool_list_and_sources(tools):
    names = [t.name for t in tools.tools()]
    assert names == ["list_hold_cases", "get_hold_case", "queue_stats", "list_live_payouts"]
    assert {t.source for t in tools.tools()} == {SOURCE_CONSOLE}


def test_list_hold_cases_rank_order_filter_limit_and_no_truth(tools):
    out = tools.list_hold_cases({})
    assert [r["event_id"] for r in out["cases"]] == [24, 7, 9]  # file (rank) order
    assert out["cases"][0]["rank"] == 1 and out["cases"][0]["top_specialist"] == "graph"
    assert out["cases"][1]["top_specialist"] is None
    assert [r["event_id"] for r in tools.list_hold_cases({"verdict": "hold"})["cases"]] == [24, 9]
    assert tools.list_hold_cases({"limit": 1})["returned"] == 1
    assert tools.list_hold_cases({"limit": "junk"})["returned"] == 3
    blob = json.dumps(out)
    assert "truth" not in blob and "outcome" not in blob and "is_attack" not in blob
    with pytest.raises(ToolError, match="HOLD or STEP_UP"):
        tools.list_hold_cases({"verdict": "RELEASE"})


def test_get_hold_case_has_no_truth_or_outcome_and_includes_explanation(tools):
    out = tools.get_hold_case({"event_id": 24})
    assert "truth" not in out and "outcome" not in out
    blob = json.dumps(out)
    assert "is_attack" not in blob and "paid_at" not in blob and "family" not in blob
    assert out["scores"] == {"graph": 1.0} and out["explanation"]["summary"] == "Held because graph."
    assert out["thresholds"] == {"step_up": 0.3, "hold": 0.6}
    assert "explanation" not in tools.get_hold_case({"event_id": "7"})  # string id accepted, no explanation built
    with pytest.raises(ToolError, match="no hold or step-up case"):
        tools.get_hold_case({"event_id": 999})
    with pytest.raises(ToolError, match="integer"):
        tools.get_hold_case({"event_id": "abc"})


def test_queue_stats_has_no_case_level_truth_counts(tools):
    out = tools.queue_stats({})
    assert out["counts"] == {"events": 10, "RELEASE": 7, "STEP_UP": 1, "HOLD": 2, "cases": 3}
    assert out["thresholds"] == {"step_up": 0.3, "hold": 0.6}
    assert out["episode_totals_usd"]["attack_usd_leaked"] == 2.0  # the contract's whole-episode totals
    assert "cases_attack" not in json.dumps(out) and "cases_honest" not in json.dumps(out)


def test_list_live_payouts_rows_and_window(tools):
    out = tools.list_live_payouts({})
    assert out["payouts"][0] == {"event_id": 1823, "payout_batch_id": "U2UPGB9BXLFRY", "batch_status": "SUCCESS",
                                 "amount": 1.0, "currency": "USD", "created_at": "2026-11-08T02:58:53.894292Z"}
    assert out["suggested_transaction_window"] == {"start_date": "2026-11-07T02:58:53Z",
                                                   "end_date": "2026-11-09T02:58:54Z"}
    assert "sender_batch_id" not in json.dumps(out)


def test_missing_files_say_not_built(tmp_path):
    t = ConsoleTools(tmp_path / "queue.json", tmp_path / "explanations.json", tmp_path / "replay_live.json")
    for fn, args in ((t.list_hold_cases, {}), (t.get_hold_case, {"event_id": 1}), (t.queue_stats, {}),
                     (t.list_live_payouts, {})):
        with pytest.raises(ToolError, match=NOT_BUILT):
            fn(args)


def test_unreadable_file_is_a_clean_error(tmp_path):
    (tmp_path / "queue.json").write_text("{not json")
    with pytest.raises(ToolError, match="unreadable"):
        ConsoleTools(tmp_path / "queue.json").queue_stats({})


def test_get_hold_case_works_without_explanations_file(tmp_path):
    (tmp_path / "queue.json").write_text(json.dumps(QUEUE))
    out = ConsoleTools(tmp_path / "queue.json", tmp_path / "nope.json").get_hold_case({"event_id": 24})
    assert "explanation" not in out and out["event_id"] == 24


# ------------------------------------------------------------------ get_payout_batch
class FakeClient:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def get_payout(self, batch_id):
        self.calls.append(batch_id)
        if self.exc:
            raise self.exc
        return self.result


def test_payout_tool_metadata_says_toolkit_has_no_payouts_tools():
    t = get_payout_batch_tool(FakeClient())
    assert t.name == "get_payout_batch" and t.source == SOURCE_REST
    assert "no Payouts tools" in t.description and "read-only" in t.description.lower()


def test_payout_tool_reads_and_validates_id():
    c = FakeClient(result={"batch_header": {"batch_status": "SUCCESS"}})
    t = get_payout_batch_tool(c)
    assert t.fn({"payout_batch_id": " U2UPGB9BXLFRY "}) == {"batch_header": {"batch_status": "SUCCESS"}}
    assert c.calls == ["U2UPGB9BXLFRY"]
    for bad in ({}, {"payout_batch_id": "../../x"}, {"payout_batch_id": "a b c d e f"}, {"payout_batch_id": 5}):
        with pytest.raises(ToolError, match="payout_batch_id"):
            t.fn(bad)
    assert len(c.calls) == 1


def test_payout_tool_maps_errors():
    nf = PayPalError(404, json.dumps({"name": "RESOURCE_NOT_FOUND", "message": "secret detail"}), "dbg")
    with pytest.raises(ToolError) as e:
        get_payout_batch_tool(FakeClient(exc=nf)).fn({"payout_batch_id": "ABCDEFGH"})
    assert str(e.value) == "PayPal sandbox: HTTP 404 RESOURCE_NOT_FOUND"
    with pytest.raises(ToolError, match="NOT_AUTHORIZED"):
        get_payout_batch_tool(FakeClient(exc=PayPalError(403, "{}", None))).fn({"payout_batch_id": "ABCDEFGH"})


def test_payout_tool_refuses_non_sandbox_host_before_any_client_is_built():
    from payguard.copilot.toolkit_tools import ToolkitConfigError

    t = get_payout_batch_tool(env={"PAYPAL_CLIENT_ID": "a", "PAYPAL_CLIENT_SECRET": "b",
                                   "PAYPAL_BASE_URL": "https://api-m.paypal.com"})
    with pytest.raises(ToolkitConfigError):
        t.fn({"payout_batch_id": "ABCDEFGH"})
