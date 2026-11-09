"""AI explanations (payguard/console/explain.py, scripts/explain_holds.py). A fake client only: no network, no key."""
from __future__ import annotations

import copy
import json

import pytest

from payguard.api.console_models import Explanations
from payguard.console import explain as ex
from payguard.llm.cache import LLMCache
from payguard.llm.errors import LLMConfigError, LLMHTTPError, LLMQuotaError, LLMSchemaError

THRESHOLDS = {"step_up": 0.3, "hold": 0.6}


def make_case(event_id=24, verdict="HOLD", risk=0.714, scores=None, share=None, fired=("graph_cluster",),
              layer_verdict="HOLD", requester_workers=6, device_workers=1, window_n=1, amount_ratio=1.0, **extra):
    scores = scores or {"velocity": 0.0, "graph": 0.714, "drift": 0.0, "policy": 0.0}
    weights = {"velocity": 0.25, "graph": 0.25, "drift": 0.25, "policy": 0.25}
    total = sum(weights[k] * scores[k] for k in scores)
    share = share or {k: (round(weights[k] * scores[k] / total, 3) if total else 0.0) for k in scores}
    case = {
        "event_id": event_id, "t": 1.69, "worker_id": 160, "requester_id": 7, "amount": 99.47, "verdict": verdict,
        "risk": risk, "decided_by": "both", "scores": scores, "weights": weights, "vote_share": share,
        "layer": {"verdict": layer_verdict, "fired": list(fired)},
        "features": {"age_days": 3.2, "n_prior": 4, "window_n": window_n, "window_usd": 140.1,
                     "amount_ratio": amount_ratio, "device_workers": device_workers, "destination_workers": 1,
                     "requester_workers": requester_workers, "worker_requesters": 1},
        "outcome": {"paid": True, "paid_at": 25.69, "delay_h": 24.0},
        "truth": {"is_attack": True, "family": "structuring"}, "rank": 1,
    }
    case.update(extra)
    return case


GOOD = {
    "summary": "Held because requester 7 is tied to 6 accounts and the Graph check scored 0.71 on a $99.47 payout.",
    "evidence": ["Graph: score 0.71, 100% of the alarm", "Requester 7 is linked to 6 accounts"],
    "check_next": "Check whether the linked accounts are a genuine business before releasing.",
    "suggested_action": "HOLD",
}


class FakeClient:
    """Stands in for GeminiClient / SarvamClient: records the calls, never touches the network."""
    model = "fake-model"

    def __init__(self, replies=None, retries=3):
        self.replies, self.calls, self.requests_made, self.retries, self.messages = replies, 0, 0, retries, []

    def complete_json(self, messages, schema):
        self.calls += 1
        self.requests_made += 1
        self.messages.append(messages)
        r = self.replies
        if callable(r):
            r = r(messages)
        if isinstance(r, Exception):
            raise r
        return copy.deepcopy(r if r is not None else GOOD)

    def close(self):
        pass


def explainer(client=None, cap=10, cache=None, log=None):
    return ex.Explainer(THRESHOLDS, model="fake-model", cache=cache if cache is not None else LLMCache(None), client=client, cap=cap, log=log)


# ---------------------------------------------------------------- numbers ---------------------------------
def test_numbers_in():
    assert ex.numbers_in("Paid $1,234.50 at 71% risk, 6 accounts, swarm_r2, #24, 0.5") == [1234.5, 71.0, 6.0, 24.0, 0.5]
    assert ex.numbers_in("no digits here") == []


def test_allowed_numbers_cover_roundings_percentages_and_small_ints():
    ok = ex.allowed_numbers(ex.case_evidence(make_case(), THRESHOLDS))
    for n in (99.47, 99.0, 99.5, 0.71, 0.7, 71.0, 71.4, 100.0, 6.0, 7.0, 24.0, 1.0, 2.0, 3.0, 0.3, 0.6, 140.1, 3.2):
        assert round(n, 6) in ok, n
    for n in (8.0, 9.0, 42.0, 99.48, 0.72, 140.2):
        assert round(n, 6) not in ok, n


# ---------------------------------------------------------------- grounding ------------------------------
def evidence():
    return ex.case_evidence(make_case(), THRESHOLDS)


def test_grounding_accepts_a_grounded_reply():
    assert ex.check_grounding(GOOD, evidence()) == []


def test_grounding_accepts_roundings_percentages_and_small_counts():
    reply = {**GOOD, "summary": "Risk is about 0.7, or 71%, and 2 of the checks flagged $99 for review.",
             "evidence": ["Graph carries 100% of the alarm", "Share 66.7% is not here but 1 account is"]}
    problems = ex.check_grounding(reply, evidence())
    assert problems == ["number 66.7 is not in the evidence"]  # only the invented one fails
    reply["evidence"][1] = "Only 1 specialist spoke up"
    assert ex.check_grounding(reply, evidence()) == []


@pytest.mark.parametrize("field,value,fragment", [
    ("summary", "Held because 42 payouts were sent in an hour.", "number 42"),
    ("summary", "Held; total of 140.1 plus 99.47 is 239.57 dollars.", "number 239.57"),
    ("evidence", ["Requester 7 is tied to 9 accounts"], "number 9"),
    ("check_next", "Verify the 55 linked accounts.", "number 55"),
    ("summary", "The vote share of Graph is 0.71.", "jargon"),
    ("summary", "Graph has a high z-score.", "jargon"),
    ("summary", " ".join(["word"] * 61), "words"),
    ("suggested_action", "BLOCK", "suggested_action"),
    ("evidence", [], "evidence"),
    ("evidence", ["a", "b", "c", "d"], "evidence"),
    ("summary", "   ", "summary"),
    ("check_next", "", "check_next"),
])
def test_grounding_rejects(field, value, fragment):
    reply = {**copy.deepcopy(GOOD), field: value}
    problems = ex.check_grounding(reply, evidence())
    assert problems and any(fragment in p for p in problems), problems


def test_grounding_ignores_digits_glued_to_names():
    reply = {**GOOD, "summary": "Compared with swarm_r2 style checks, the Graph check flagged payout 24."}
    assert ex.check_grounding(reply, evidence()) == []


# ---------------------------------------------------------------- template -------------------------------
TEMPLATE_CASES = [
    make_case(),
    make_case(event_id=5, verdict="STEP_UP", risk=0.5, scores={"velocity": 0.5, "graph": 0.0, "drift": 0.4, "policy": 0.0},
              fired=(), layer_verdict="RELEASE", requester_workers=1, window_n=4, amount_ratio=3.7),
    make_case(event_id=6, verdict="STEP_UP", risk=0.0, scores={"velocity": 0.0, "graph": 0.0, "drift": 0.0, "policy": 0.0},
              fired=("window_usd", "rate_z", "sharing", "amount"), layer_verdict="STEP_UP", requester_workers=1,
              device_workers=5),
    make_case(event_id=7, verdict="HOLD", risk=0.9, scores={"velocity": 0.0, "graph": 0.0, "drift": 0.0, "policy": 0.9},
              fired=("amount_over_cap",), requester_workers=1),
    make_case(event_id=8, verdict="HOLD", risk=1.0, scores={"velocity": 1.0, "graph": 1.0, "drift": 1.0, "policy": 1.0},
              requester_workers=1),
]


@pytest.mark.parametrize("case", TEMPLATE_CASES, ids=lambda c: f"case{c['event_id']}")
def test_template_is_grounded_short_and_deterministic(case):
    t = ex.template_explanation(case, THRESHOLDS)
    assert ex.check_grounding(t, ex.case_evidence(case, THRESHOLDS)) == []
    assert len(t["summary"].split()) <= 60 and 1 <= len(t["evidence"]) <= 3
    assert t["suggested_action"] == case["verdict"]
    assert t == ex.template_explanation(copy.deepcopy(case), THRESHOLDS)
    assert t["check_next"].endswith(".")


def test_template_never_reads_truth_or_outcome():
    a = make_case()
    b = make_case(truth={"is_attack": False, "family": "honest"}, outcome={"paid": False, "paid_at": None, "delay_h": None})
    assert ex.template_explanation(a, THRESHOLDS) == ex.template_explanation(b, THRESHOLDS)


def test_template_for_real_committed_queue_if_present():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "eval" / "console" / "queue.json"
    if not path.is_file():
        pytest.skip("queue not built")
    q = json.loads(path.read_text(encoding="utf-8"))
    for c in q["cases"]:
        t = ex.template_explanation(c, q["thresholds"])
        assert ex.check_grounding(t, ex.case_evidence(c, q["thresholds"])) == [], c["event_id"]
        assert len(t["summary"].split()) <= 60


# ---------------------------------------------------------------- the prompt -----------------------------
FORBIDDEN = ("truth", "outcome", "is_attack", "family", "attack", "ground")


def test_prompt_never_contains_truth_outcome_or_family():
    for case in TEMPLATE_CASES:
        for family in ("structuring", "ring", "sleeper", "honest", "h2"):
            c = make_case(**{k: case[k] for k in ("event_id", "verdict", "risk")},
                          truth={"is_attack": True, "family": family},
                          outcome={"paid": True, "paid_at": 9.9, "delay_h": 8.8})
            text = json.dumps(ex.build_messages(ex.case_evidence(c, THRESHOLDS))).lower()
            for word in (*FORBIDDEN, "structuring", "sleeper", "honest"):
                assert word not in text, word
            assert '"ring"' not in text and '"h2"' not in text
    ev = ex.case_evidence(make_case(), THRESHOLDS)
    assert set(ev) == {"payout", "system_verdict", "thresholds", "overall_risk", "set_by", "specialists",
                       "anomaly_checks", "account_and_platform"}


def test_prompt_is_identical_whatever_the_truth_and_outcome_are():
    a = make_case()
    b = make_case(truth={"is_attack": False, "family": "honest"}, outcome={"paid": False, "paid_at": None, "delay_h": None})
    ma, mb = ex.build_messages(ex.case_evidence(a, THRESHOLDS)), ex.build_messages(ex.case_evidence(b, THRESHOLDS))
    assert ma == mb
    cache = LLMCache(None)
    e1, e2 = ex.Explainer(THRESHOLDS, model="m", cache=cache), ex.Explainer(THRESHOLDS, model="m", cache=cache)
    assert e1.key(ex.case_evidence(a, THRESHOLDS)) == e2.key(ex.case_evidence(b, THRESHOLDS))


def test_client_receives_exactly_the_built_messages_and_schema():
    client = FakeClient()
    case = make_case()
    explainer(client).explain(case)
    assert client.messages == [ex.build_messages(ex.case_evidence(case, THRESHOLDS))]
    assert [m["role"] for m in client.messages[0]] == ["system", "user"]
    assert set(ex.SCHEMA["required"]) == {"summary", "evidence", "check_next", "suggested_action"}


# ---------------------------------------------------------------- explainer: llm, template, cache, cap ---
def test_grounded_reply_is_used_and_marked_llm():
    out = explainer(FakeClient()).explain(make_case())
    assert out["source"] == "llm" and out["summary"] == GOOD["summary"] and out["suggested_action"] == "HOLD"


def test_ungrounded_reply_falls_back_to_the_template():
    bad = {**GOOD, "summary": "Held because 42 payouts went out in one hour."}
    e = explainer(FakeClient(bad))
    out = e.explain(make_case())
    assert out["source"] == "template" and e.ungrounded == 1
    assert out == {"source": "template", **ex.template_explanation(make_case(), THRESHOLDS)}


def test_cache_hit_means_no_client_call(tmp_path):
    path = tmp_path / "cache.jsonl"
    c1 = FakeClient()
    first = explainer(c1, cache=LLMCache(path)).explain(make_case())
    assert c1.calls == 1 and first["source"] == "llm"
    c2 = FakeClient(LLMHTTPError(500, "must not be called"))
    second = explainer(c2, cache=LLMCache(path)).explain(make_case())  # a fresh cache object reading the file
    assert c2.calls == 0 and second == first
    # no client at all (offline): still served from the cache
    third = explainer(None, cap=0, cache=LLMCache(path)).explain(make_case())
    assert third == first


def test_cached_ungrounded_reply_is_not_asked_again(tmp_path):
    bad = {**GOOD, "evidence": ["Seven accounts are tied to this requester 98"]}
    cache = LLMCache(tmp_path / "c.jsonl")
    explainer(FakeClient(bad), cache=cache).explain(make_case())
    c2 = FakeClient()
    out = explainer(c2, cache=cache).explain(make_case())
    assert c2.calls == 0 and out["source"] == "template"


def test_cache_key_depends_on_model_and_prompt_version():
    ev = ex.case_evidence(make_case(), THRESHOLDS)
    cache = LLMCache(None)
    a = ex.Explainer(THRESHOLDS, model="m1", cache=cache).key(ev)
    b = ex.Explainer(THRESHOLDS, model="m2", cache=cache).key(ev)
    assert a != b
    assert ex.PROMPT_VERSION == "explain-v1"


def test_cap_is_respected_and_the_rest_is_template():
    cases = [make_case(event_id=i) for i in range(20, 25)]
    client = FakeClient(lambda m: GOOD)  # not case specific: some will be ungrounded for other ids, fine
    e = explainer(client, cap=2)
    out = ex.explain_cases(cases, e)
    assert client.calls == 2 and e.calls == 2 and client.requests_made == 2
    assert len(out) == 5 and sum(1 for v in out.values() if v["source"] == "template") >= 3


def test_cap_zero_makes_no_call():
    client = FakeClient()
    ex.explain_cases([make_case()], explainer(client, cap=0))
    assert client.calls == 0


def test_cap_counts_real_requests_and_clamps_retries():
    """A client that spends several requests per call must never push the total beyond the cap."""
    class Retrying(FakeClient):
        def complete_json(self, messages, schema):
            self.requests_made += 1 + self.retries  # worst case: every retry is used
            self.calls += 1
            return copy.deepcopy(GOOD)

    client = Retrying(retries=3)
    cases = [make_case(event_id=i) for i in range(40, 50)]
    e = explainer(client, cap=6)
    ex.explain_cases(cases, e)
    assert client.requests_made <= 6 and client.calls >= 1


def test_quota_error_stops_the_run_and_keeps_the_rest_as_template():
    client = FakeClient(LLMQuotaError(429, "RESOURCE_EXHAUSTED"))
    logs = []
    e = explainer(client, cap=50, log=logs.append)
    out = ex.explain_cases([make_case(event_id=i) for i in range(30, 36)], e)
    assert client.calls == 1 and e.quota_stop and all(v["source"] == "template" for v in out.values())
    assert any("quota" in line for line in logs)


def test_repeated_errors_stop_the_run():
    client = FakeClient(LLMSchemaError("reply content was not valid JSON"))
    e = explainer(client, cap=50)
    out = ex.explain_cases([make_case(event_id=i) for i in range(30, 50)], e)
    assert client.calls == ex.MAX_CONSECUTIVE_ERRORS and e.errors == ex.MAX_CONSECUTIVE_ERRORS
    assert all(v["source"] == "template" for v in out.values())


def test_a_failed_call_is_not_cached():
    cache = LLMCache(None)
    explainer(FakeClient(LLMHTTPError(500, "boom")), cache=cache).explain(make_case())
    assert len(cache) == 0


def test_default_cache_has_its_own_file(monkeypatch):
    monkeypatch.delenv("PAYGUARD_LLM_CACHE_EXPLAIN", raising=False)
    p = ex.default_cache_path()
    assert p.name == "llm_cache_explain.jsonl" and p.parent.name == "data"
    monkeypatch.setenv("PAYGUARD_LLM_CACHE_EXPLAIN", "somewhere/x.jsonl")
    assert str(ex.default_cache_path()).replace("\\", "/") == "somewhere/x.jsonl"


# ---------------------------------------------------------------- the CLI ---------------------------------
@pytest.fixture
def cli(tmp_path, monkeypatch):
    from scripts import explain_holds as cli_mod

    cases = [make_case(event_id=i, rank=i - 19) for i in range(20, 26)]
    queue = {"schema_version": 1, "simulated": True, "thresholds": THRESHOLDS, "cases": cases}
    q = tmp_path / "queue.json"
    q.write_text(json.dumps(queue), encoding="utf-8")
    monkeypatch.setattr(cli_mod, "provider_model", lambda provider: f"{provider}-model")
    return cli_mod, q, tmp_path


def args_for(q, tmp, *more):
    return [*more, "--queue", str(q), "--out", str(tmp / "explanations.json"), "--cache", str(tmp / "cache.jsonl")]


def test_cli_offline_writes_a_valid_all_template_file(cli, capsys):
    mod, q, tmp = cli
    assert mod.main(args_for(q, tmp, "--offline")) == 0
    doc = json.loads((tmp / "explanations.json").read_text(encoding="utf-8"))
    assert doc["counts"] == {"llm": 0, "template": 6} and set(doc["explanations"]) == {str(i) for i in range(20, 26)}
    assert doc["provider"] == "gemini" and doc["prompt_version"] == "explain-v1" and doc["grounding"] == ex.GROUNDING_TEXT
    Explanations.model_validate(doc)  # the API's model accepts it
    assert not (tmp / "cache.jsonl").exists()  # offline never writes the cache


def test_cli_plan_makes_no_call_and_counts_uncached(cli, capsys, monkeypatch):
    mod, q, tmp = cli
    monkeypatch.setattr(mod, "make_client", lambda p: pytest.fail("--plan must not build a client"))
    assert mod.main(args_for(q, tmp, "--plan", "--provider", "sarvam")) == 0
    out = capsys.readouterr().out
    assert "sarvam-model" in out and "cases: 6; uncached: 6" in out and "estimated paid Sarvam calls" in out and " 6 " in out
    assert not (tmp / "explanations.json").exists()


def test_cli_run_respects_the_cap_then_rerun_is_free(cli, monkeypatch):
    mod, q, tmp = cli
    made = []

    def reply(messages):
        evidence = json.loads(messages[1]["content"].split("\n", 1)[1])
        p = evidence["payout"]
        return {"summary": f"Held payout {p['event_id']} of ${p['amount_usd']:.2f}.", "evidence": ["Graph raised the alarm"],
                "check_next": "Check the linked accounts.", "suggested_action": "HOLD"}

    def make(provider):
        made.append(FakeClient(reply))
        return made[-1]

    monkeypatch.setattr(mod, "make_client", make)
    assert mod.main(args_for(q, tmp, "--run", "--provider", "gemini", "--cap", "4")) == 0
    doc = json.loads((tmp / "explanations.json").read_text(encoding="utf-8"))
    assert made[0].calls == 4 and doc["counts"] == {"llm": 4, "template": 2} and doc["model"] == "fake-model"
    # top of the queue first: the two template ones are the last two
    assert [doc["explanations"][str(i)]["source"] for i in range(20, 26)] == ["llm"] * 4 + ["template"] * 2
    Explanations.model_validate(doc)
    # second run, bigger cap: only the two missing cases are asked
    assert mod.main(args_for(q, tmp, "--run", "--provider", "gemini", "--cap", "50")) == 0
    assert made[1].calls == 2
    # third run: everything cached, no call, same file content apart from the timestamp
    before = json.loads((tmp / "explanations.json").read_text(encoding="utf-8"))
    assert mod.main(args_for(q, tmp, "--run", "--provider", "gemini", "--cap", "50")) == 0
    after = json.loads((tmp / "explanations.json").read_text(encoding="utf-8"))
    assert made[2].calls == 0
    before.pop("generated_at"), after.pop("generated_at")
    assert before == after and after["counts"] == {"llm": 6, "template": 0}


def test_cli_run_needs_provider_and_cap(cli):
    mod, q, tmp = cli
    with pytest.raises(SystemExit):
        mod.main(args_for(q, tmp, "--run"))
    with pytest.raises(SystemExit):
        mod.main(args_for(q, tmp, "--run", "--provider", "gemini", "--cap", "0"))


def test_cli_run_without_a_key_stops_cleanly(cli, monkeypatch, capsys):
    mod, q, tmp = cli

    def no_key(provider):
        raise LLMConfigError("Missing environment variable: GEMINI_API_KEY (see .env.example)")

    monkeypatch.setattr(mod, "make_client", no_key)
    assert mod.main(args_for(q, tmp, "--run", "--provider", "gemini", "--cap", "3")) == 1
    assert not (tmp / "explanations.json").exists() and "GEMINI_API_KEY" in capsys.readouterr().err


def test_cli_will_not_overwrite_llm_explanations_with_fewer(cli, capsys):
    mod, q, tmp = cli
    out = tmp / "explanations.json"
    out.write_text(json.dumps({"counts": {"llm": 5, "template": 1}}), encoding="utf-8")
    assert mod.main(args_for(q, tmp, "--offline")) == 2
    assert json.loads(out.read_text(encoding="utf-8"))["counts"]["llm"] == 5 and "REFUSED" in capsys.readouterr().err
    assert mod.main(args_for(q, tmp, "--offline", "--force")) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["counts"]["llm"] == 0


def test_cli_missing_queue_is_a_clear_error(tmp_path):
    from scripts import explain_holds as mod

    with pytest.raises(SystemExit, match="build_console"):
        mod.load_queue(tmp_path / "nope.json")
