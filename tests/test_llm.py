"""LLM layer: client, payload, cache, judge. No test ever reaches a real API (mock transports / fakes only)."""
import json
import logging
import re

import httpx
import pytest

from payguard.llm.cache import LLMCache, cache_key
from payguard.llm.client import SarvamClient, validate_schema
from payguard.llm.errors import LLMConfigError, LLMError, LLMHTTPError, LLMSchemaError, LLMTransportError
from payguard.llm.judge import (
    PROMPT_VERSION,
    SCHEMA,
    LLMAssessor,
    LLMJudge,
    QualityDefender,
    fallback_risk,
    risk_to_verdict,
)
from payguard.llm.payload import FIELDS, LABELS, NO_HISTORY, judge_payload
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from payguard.sim.features import PlatformLog
from payguard.sim.runner import run_episode
from payguard.sim.types import Event, Verdict

KEY = "sk-test-SECRET-0123456789"
GOOD = {"verdict": "STEP_UP", "risk": 0.4, "rationale": "young account, large amount"}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Never pick up a developer's real .env or env vars: a stray key must not trigger paid calls."""
    monkeypatch.setattr("payguard.config.load_dotenv", lambda *a, **k: False)
    for name in ("SARVAM_API_KEY", "SARVAM_MODEL", "SARVAM_BASE_URL", "PAYGUARD_LLM_OFFLINE",
                 "PAYGUARD_LLM_MAX_CALLS", "PAYGUARD_LLM_CACHE"):
        monkeypatch.delenv(name, raising=False)


def reply(obj, status=200):
    body = {"choices": [{"message": {"content": obj if isinstance(obj, str) else json.dumps(obj)},
                         "finish_reason": "stop"}]}
    return httpx.Response(status, json=body)


def make_client(handler, **kw):
    sleeps = []
    c = SarvamClient(api_key=KEY, transport=httpx.MockTransport(handler), sleep=sleeps.append, **kw)
    return c, sleeps


MSGS = [{"role": "user", "content": "{}"}]


# ---------------------------------------------------------------- client
def test_client_happy_path_sends_documented_request():
    seen = []

    def handler(req):
        seen.append(req)
        return reply(GOOD)

    c, _ = make_client(handler)
    assert c.complete_json(MSGS, SCHEMA) == GOOD
    req = seen[0]
    assert req.url.path == "/v1/chat/completions" and str(req.url).startswith("https://api.sarvam.ai")
    assert req.headers["api-subscription-key"] == KEY and req.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["model"] == "sarvam-105b-conversations" and body["messages"] == MSGS
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "reasoning_effort" not in body and c.requests_made == 1


@pytest.mark.parametrize("status", [429, 503])
def test_client_retries_transient_errors_with_backoff(status):
    codes = iter([status, status, 200])

    def handler(req):
        code = next(codes)
        return reply(GOOD) if code == 200 else httpx.Response(code, text="busy")

    c, sleeps = make_client(handler)
    assert c.complete_json(MSGS, SCHEMA) == GOOD
    assert c.requests_made == 3 and sleeps == [0.5, 1.0]


def test_client_gives_up_after_two_retries():
    c, sleeps = make_client(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(LLMHTTPError) as ei:
        c.complete_json(MSGS, SCHEMA)
    assert ei.value.status == 500 and c.requests_made == 3 and len(sleeps) == 2


def test_client_does_not_retry_4xx():
    c, sleeps = make_client(lambda req: httpx.Response(401, text="bad key"))
    with pytest.raises(LLMHTTPError) as ei:
        c.complete_json(MSGS, SCHEMA)
    assert ei.value.status == 401 and c.requests_made == 1 and sleeps == []


def test_client_transport_errors_are_retried_then_typed():
    def handler(req):
        raise httpx.ConnectError(f"cannot connect (key={KEY})")

    c, _ = make_client(handler)
    with pytest.raises(LLMTransportError) as ei:
        c.complete_json(MSGS, SCHEMA)
    assert c.requests_made == 3 and KEY not in str(ei.value)


@pytest.mark.parametrize("content", [
    {"verdict": "REFUND", "risk": 0.1, "rationale": "x"},   # enum violation
    {"verdict": "HOLD", "risk": 0.1},                        # missing required field
    {"verdict": "HOLD", "risk": "high", "rationale": "x"},   # wrong type
    {"verdict": "HOLD", "risk": True, "rationale": "x"},     # bool is not a number
    "this is not json",
    "[1, 2]",
])
def test_client_schema_violation_raises_and_is_not_retried(content):
    c, sleeps = make_client(lambda req: reply(content))
    with pytest.raises(LLMSchemaError):
        c.complete_json(MSGS, SCHEMA)
    assert c.requests_made == 1 and sleeps == []


def test_client_malformed_envelope_is_a_schema_error():
    c, _ = make_client(lambda req: httpx.Response(200, json={"choices": []}))
    with pytest.raises(LLMSchemaError):
        c.complete_json(MSGS, SCHEMA)


def test_validate_schema_handles_range_and_nested_errors():
    schema = {"type": "object", "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["x"]}
    validate_schema({"x": 0.5, "extra": 1}, schema)  # extras tolerated
    for bad in ({"x": 2}, {"x": -1}, {}):
        with pytest.raises(LLMSchemaError):
            validate_schema(bad, schema)


def test_api_key_never_appears_in_errors_repr_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    c, _ = make_client(lambda req: httpx.Response(400, text=f"invalid key {KEY}"))
    with pytest.raises(LLMHTTPError) as ei:
        c.complete_json(MSGS, SCHEMA)
    c2, _ = make_client(lambda req: httpx.Response(503, text=f"upstream said {KEY}"))
    with pytest.raises(LLMHTTPError) as ei2:
        c2.complete_json(MSGS, SCHEMA)
    for text in (str(ei.value), repr(ei.value), str(ei2.value), repr(c), caplog.text):
        assert KEY not in text


def test_client_without_a_key_names_the_variable_not_a_value():
    with pytest.raises(LLMConfigError, match="SARVAM_API_KEY"):
        SarvamClient()


# ---------------------------------------------------------------- payload
def ev(id, t, worker, requester, device, dest, amount):
    return Event(id=id, t=t, worker_id=worker, requester_id=requester, device_id=device,
                 destination_id=dest, amount=amount)


def small_log(workers=(1, 2), age_days=100):
    return PlatformLog({w: -24.0 * age_days for w in workers})


def test_payload_has_only_bucket_labels_and_no_ids_times_or_truth():
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    log = PlatformLog(ep.accounts)
    seen = set()
    for e in ep.events[:1500]:
        p = judge_payload(e, log)
        log.add(e)
        assert tuple(p) == FIELDS
        assert all(p[f] in LABELS[f] or (f == "amount_vs_own_median" and p[f] == NO_HISTORY) for f in FIELDS)
        seen.add(json.dumps(p, sort_keys=True))
    text = " ".join(seen)
    for forbidden in ("family", "campaign", "attack", "truth", "structuring", "sleeper",
                      "worker_id", "requester_id", "device_id", "destination_id", "_id", "event"):
        assert forbidden not in text
    assert not re.search(r"ring", text)
    assert len(seen) > 5  # not degenerate


def test_equivalent_events_give_identical_payloads():
    a = ev(1, 10.0, 1, 5, 7, 9, 120.0)
    b = ev(99, 500.0, 2, 6, 8, 10, 130.0)  # other ids, other time, same buckets
    assert judge_payload(a, small_log()) == judge_payload(b, small_log())
    c = ev(2, 10.0, 1, 5, 7, 9, 400.0)     # different amount bucket
    assert judge_payload(a, small_log()) != judge_payload(c, small_log())


def test_payload_flags_graph_structure_and_young_accounts():
    log = small_log(workers=range(1, 9))
    for w in range(2, 8):  # six other workers all paid via requesters 5 and 6
        for r in (5, 6):
            log.add(ev(w * 10 + r, 1.0, w, r, 100 + w, 200 + w, 50.0))
    ringlike = judge_payload(ev(1000, 2.0, 1, 5, 101, 201, 50.0), log)
    assert ringlike["workers_with_similar_requesters"] == "4+"
    loner = judge_payload(ev(1001, 2.0, 8, 77, 108, 208, 50.0), log)
    assert loner["workers_with_similar_requesters"] == "1-3" and loner["workers_per_requester"] == "<10"
    young = PlatformLog({1: 0.0})
    p = judge_payload(ev(1, 2.0, 1, 5, 7, 9, 50.0), young)
    assert p["account_age"] == "<7d" and p["amount_vs_own_median"] == NO_HISTORY


def test_payload_does_not_mutate_the_log():
    log = small_log()
    judge_payload(ev(1, 1.0, 1, 5, 7, 9, 50.0), log)
    for index in (log.workers_by_requester, log.requesters_by_worker, log.workers_by_device,
                  log.workers_by_destination):
        assert not any(index.values())  # features() may touch defaultdict keys, but never records anything


# ---------------------------------------------------------------- cache
def test_cache_key_is_stable_and_sensitive():
    p = {"a": "1", "b": "2"}
    k = cache_key("m", "v1", p)
    assert k == cache_key("m", "v1", {"b": "2", "a": "1"}) and len(k) == 64
    assert len({k, cache_key("m2", "v1", p), cache_key("m", "v2", p), cache_key("m", "v1", {"a": "1"})}) == 4


def test_cache_persists_and_survives_a_torn_line(tmp_path):
    path = tmp_path / "sub" / "c.jsonl"
    c = LLMCache(path)
    c.put("k1", GOOD, model="m", prompt_version="v1", payload={"a": "1"})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "torn", "resp')  # crash mid-write, no trailing newline
    c2 = LLMCache(path)
    assert c2.get("k1") == GOOD and c2.get("torn") is None and len(c2) == 1
    c2.put("k2", GOOD, model="m", prompt_version="v1", payload={"a": "2"})
    c3 = LLMCache(path)
    assert c3.get("k2") == GOOD and len(c3) == 2


def test_cache_is_thread_safe(tmp_path):
    import threading

    c = LLMCache(tmp_path / "t.jsonl")
    ts = [threading.Thread(target=lambda i=i: [c.put(f"k{i}-{j}", GOOD, model="m", prompt_version="v", payload={})
                                                 for j in range(25)]) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(c) == 200 and len(LLMCache(tmp_path / "t.jsonl")) == 200


# ---------------------------------------------------------------- judge
class FakeClient:
    model = "fake-model"

    def __init__(self, reply=GOOD, exc=None):
        self.reply, self.exc, self.n = reply, exc, 0

    def complete_json(self, messages, schema):
        self.n += 1
        if self.exc:
            raise self.exc
        return dict(self.reply)


# Five payouts that fall in five different payload buckets (distinct cache keys).
def distinct_events():
    log = small_log(workers=(1,))
    amounts = (10.0, 100.0, 300.0, 450.0, 600.0)
    return log, [ev(i, 5.0, 1, 5, 7, 9, a) for i, a in enumerate(amounts)]


def assessor(client=None, **kw):
    return LLMAssessor(client, LLMCache(None), **{"max_calls": 300, **kw})


def test_cache_hit_never_calls_the_client():
    log, events = distinct_events()
    client = FakeClient()
    a = assessor(client)
    first = a.assess(events[0], log)
    again = a.assess(events[0], log)
    assert (first.source, again.source) == ("api", "cache") and client.n == 1 and a.calls == 1
    assert again.verdict is Verdict.STEP_UP and again.risk == 0.4 and a.cache_hits == 1


def test_prepopulated_cache_means_zero_calls():
    log, events = distinct_events()
    cache = LLMCache(None)
    payload = judge_payload(events[0], log)
    cache.put(cache_key("fake-model", PROMPT_VERSION, payload), {**GOOD, "verdict": "HOLD"},
              model="fake-model", prompt_version=PROMPT_VERSION, payload=payload)
    client = FakeClient()
    a = LLMAssessor(client, cache)
    assert a.assess(events[0], log).verdict is Verdict.HOLD and client.n == 0 and a.calls == 0


def test_corrupt_cache_entry_is_treated_as_a_miss():
    log, events = distinct_events()
    cache = LLMCache(None)
    payload = judge_payload(events[0], log)
    cache.put(cache_key("fake-model", PROMPT_VERSION, payload), {"verdict": "MAYBE"},
              model="fake-model", prompt_version=PROMPT_VERSION, payload=payload)
    client = FakeClient()
    assert LLMAssessor(client, cache, max_calls=1).assess(events[0], log).source == "api" and client.n == 1


def test_spend_cap_then_deterministic_fallback():
    log, events = distinct_events()
    client = FakeClient()
    a = assessor(client, max_calls=2)
    got = [a.assess(e, log) for e in events]
    assert [g.source for g in got] == ["api", "api", "fallback", "fallback", "fallback"]
    assert client.n == 2 and a.calls == 2 and a.fallbacks == {"cap": 3}
    # Fallback is a pure function of the payload: asking again gives the same answer and still no call.
    again = [a.assess(e, log) for e in events[2:]]
    assert [(g.verdict, g.risk) for g in again] == [(g.verdict, g.risk) for g in got[2:]]
    assert client.n == 2 and a.fallback_count == 6


def test_fallback_is_not_release_everything_and_not_hold_everything():
    young = PlatformLog({1: 0.0, 2: -24.0 * 400})
    big_young = judge_payload(ev(1, 2.0, 1, 5, 7, 9, 600.0), young)
    ordinary = judge_payload(ev(2, 2.0, 2, 5, 8, 10, 40.0), young)
    assert risk_to_verdict(fallback_risk(big_young)) is Verdict.HOLD
    assert risk_to_verdict(fallback_risk(ordinary)) is Verdict.RELEASE
    mid = judge_payload(ev(3, 2.0, 2, 5, 8, 10, 600.0), young)
    assert risk_to_verdict(fallback_risk(mid)) is Verdict.STEP_UP


def test_api_error_falls_back_counts_the_call_and_is_not_cached():
    log, events = distinct_events()
    client = FakeClient(exc=LLMHTTPError(500, "down"))
    a = assessor(client)
    r = a.assess(events[4], log)  # big payout: fallback should not release it
    assert r.source == "fallback" and r.called and r.verdict is not Verdict.RELEASE
    assert a.calls == 1 and a.fallbacks == {"error": 1} and len(a.cache) == 0
    a.assess(events[4], log)
    assert client.n == 2  # not cached, so retried


def test_bad_model_reply_falls_back():
    log, events = distinct_events()
    a = assessor(FakeClient(reply={"verdict": "MAYBE", "risk": 0.5, "rationale": "x"}))
    assert a.assess(events[0], log).source == "fallback" and len(a.cache) == 0


def test_risk_is_clamped_to_unit_interval():
    log, events = distinct_events()
    q = QualityDefender(assessor(FakeClient(reply={**GOOD, "risk": 1.7})))
    assert q.score(events[0], log) == 1.0
    q = QualityDefender(assessor(FakeClient(reply={**GOOD, "risk": -0.2})))
    assert q.score(events[0], log) == 0.0


def test_offline_mode_is_cache_only(monkeypatch):
    log, events = distinct_events()
    client = FakeClient()
    cache = LLMCache(None)
    payload = judge_payload(events[0], log)
    cache.put(cache_key("fake-model", PROMPT_VERSION, payload), GOOD, model="fake-model",
              prompt_version=PROMPT_VERSION, payload=payload)
    a = LLMAssessor(client, cache, offline=True)
    assert a.assess(events[0], log).source == "cache"
    assert a.assess(events[1], log).source == "fallback"
    assert client.n == 0 and a.calls == 0 and a.fallbacks == {"offline": 1}
    # Env switch behaves the same, even with a client injected.
    monkeypatch.setenv("PAYGUARD_LLM_OFFLINE", "1")
    b = LLMAssessor(client, LLMCache(None))
    assert b.offline and b.assess(events[1], log).source == "fallback" and client.n == 0


def test_missing_api_key_means_offline_without_building_a_client():
    a = LLMAssessor(cache=LLMCache(None))
    assert a.offline and a.client is None
    log, events = distinct_events()
    assert a.assess(events[0], log).source == "fallback" and a.calls == 0


def test_env_cap_and_default_cap(monkeypatch):
    # No cap given anywhere -> 0: paid calls are opt-in, even with a client (key) present.
    assert LLMAssessor(FakeClient(), LLMCache(None)).max_calls == 0
    monkeypatch.setenv("PAYGUARD_LLM_MAX_CALLS", "7")
    assert LLMAssessor(FakeClient(), LLMCache(None)).max_calls == 7
    assert assessor(FakeClient(), max_calls=0).max_calls == 0


def test_default_makes_no_paid_call():
    a = LLMAssessor(FakeClient(), LLMCache(None))
    log, events = distinct_events()
    got = a.assess(events[0], log)
    assert got.source == "fallback" and a.calls == 0 and a.fallbacks["cap"] == 1


def test_cap_of_zero_never_calls():
    log, events = distinct_events()
    client = FakeClient()
    assessor(client, max_calls=0).assess(events[0], log)
    assert client.n == 0


def test_judge_and_quality_share_cache_and_count_their_own_calls():
    log, events = distinct_events()
    client = FakeClient()
    shared = assessor(client)
    judge, quality = LLMJudge(shared), QualityDefender(shared)
    assert judge.name == "llm_judge" and quality.name == "quality"
    assert judge.decide(events[0], log) is Verdict.STEP_UP
    assert quality.score(events[0], log) == 0.4  # same payload: cache, not a second call
    assert quality.score(events[1], log) == 0.4
    assert (judge.llm_calls, quality.llm_calls, shared.calls, client.n) == (1, 1, 2, 2)
    assert judge.learn(events[0], True) is None


def test_system_prompt_states_costs_and_has_no_attack_family_hints():
    from payguard.llm.judge import SYSTEM_PROMPT

    for needle in ("RELEASE", "STEP_UP", "HOLD", "4 hours", "24 hours"):
        assert needle in SYSTEM_PROMPT
    for family in (r"ring", "structuring", "sleeper", "collu", "smurf", "launder"):
        assert not re.search(family, SYSTEM_PROMPT.lower())


def test_llm_judge_runs_end_to_end_on_a_train_episode_and_replays_from_cache(tmp_path):
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    path = tmp_path / "cache.jsonl"

    class Rule(FakeClient):
        def complete_json(self, messages, schema):
            self.n += 1
            young = '"<7d"' in messages[-1]["content"]
            return {"verdict": "HOLD" if young else "RELEASE", "risk": 0.9 if young else 0.05, "rationale": "r"}

    c1 = Rule()
    j1 = LLMJudge(LLMAssessor(c1, LLMCache(path), max_calls=10_000))
    r1 = run_episode(ep, j1)
    assert set(r1.verdicts) == {e.id for e in ep.events}
    assert r1.llm_calls == j1.llm_calls == c1.n > 0
    assert r1.llm_calls < 0.10 * len(ep.events)  # bucketing: far fewer calls than payouts
    assert Verdict.HOLD in set(r1.verdicts.values()) and Verdict.RELEASE in set(r1.verdicts.values())

    c2 = Rule()  # fresh judge, same cache file: deterministic replay with zero API calls
    j2 = LLMJudge(LLMAssessor(c2, LLMCache(path), max_calls=10_000))
    r2 = run_episode(ep, j2)
    assert r2.verdicts == r1.verdicts and r2.llm_calls == 0 and c2.n == 0


def test_end_to_end_with_cap_never_releases_everything():
    ep = episode_for(TRAIN_SEEDS[1], TRAIN_FAMILIES)
    judge = LLMJudge(assessor(FakeClient(reply={**GOOD, "verdict": "RELEASE", "risk": 0.0}), max_calls=3))
    r = run_episode(ep, judge)
    assert r.llm_calls == 3 and judge.assessor.fallback_count > 0
    assert len(set(r.verdicts.values())) > 1


def test_schema_constant_matches_contract():
    assert SCHEMA["required"] == ["verdict", "risk", "rationale"]
    assert SCHEMA["properties"]["verdict"]["enum"] == ["RELEASE", "HOLD", "STEP_UP"]
    assert issubclass(LLMHTTPError, LLMError) and issubclass(LLMSchemaError, LLMError)
