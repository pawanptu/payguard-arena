"""Gemini judge: client, throttle, retries, quota, factory, llm_warm. No test reaches a real API (mock transports
and fakes only). TRAIN seeds / synthetic payloads only."""
import json
import logging

import httpx
import pytest

from payguard.llm.cache import LLMCache, cache_key, default_cache_path, default_gemini_cache_path
from payguard.llm.errors import (
    LLMConfigError,
    LLMHTTPError,
    LLMQuotaError,
    LLMSchemaError,
    LLMTransportError,
)
from payguard.llm.gemini import DEFAULT_MODEL, GeminiClient
from payguard.llm.judge import (
    PROMPT_VERSION,
    SCHEMA,
    SYSTEM_PROMPT,
    LLMAssessor,
    LLMJudge,
    QualityDefender,
    make_assessor,
)
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from payguard.sim.types import Verdict

KEY = "AIza-test-SECRET-0123456789"
GOOD = {"verdict": "STEP_UP", "risk": 0.4, "rationale": "young account, large amount"}
PAYLOAD = {"account_age": "<7d", "amount_usd": "500+", "payouts_24h": "1-2", "amount_vs_own_median": "n/a",
           "workers_sharing_device_or_destination": "1", "workers_per_requester": "<10",
           "workers_with_similar_requesters": "1-3"}
# Computed from the Sarvam code BEFORE the Gemini change (cache_key("sarvam-105b-conversations", "v1", PAYLOAD)).
SARVAM_KEY_BEFORE = "20679a507bf2db6d7a7db0fdb79d739a92870c44560847f7c130664f39106027"
MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": '{"a": "1"}'}]


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Never pick up a developer's real .env or env vars: a stray key must not trigger calls."""
    monkeypatch.setattr("payguard.config.load_dotenv", lambda *a, **k: False)
    for name in ("SARVAM_API_KEY", "SARVAM_MODEL", "SARVAM_BASE_URL", "PAYGUARD_LLM_OFFLINE",
                 "PAYGUARD_LLM_MAX_CALLS", "PAYGUARD_LLM_CACHE", "PAYGUARD_LLM_CACHE_GEMINI",
                 "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_BASE_URL", "GEMINI_MIN_INTERVAL_S"):
        monkeypatch.delenv(name, raising=False)


def ok(obj=GOOD, **extra):
    text = obj if isinstance(obj, str) else json.dumps(obj)
    cand = {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP", **extra}
    return httpx.Response(200, json={"candidates": [cand]})


class Clock:
    """Fake time: sleeping advances it, so throttle and retry waits are observable without waiting."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def make_client(handler, *, min_interval_s=0.0, **kw):
    clock = Clock()
    c = GeminiClient(api_key=KEY, transport=httpx.MockTransport(handler), sleep=clock.sleep, clock=clock,
                     min_interval_s=min_interval_s, **kw)
    return c, clock


def quota_429(quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier", value="20"):
    return httpx.Response(429, json={"error": {"code": 429, "message": f"quota used up {KEY}",
                                                "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaMetric": "x", "quotaId": quota_id, "quotaValue": value}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]}})


def rate_429(delay="12s"):
    return httpx.Response(429, json={"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED",
                                                "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}})


# ---------------------------------------------------------------- request shape
def test_request_shape():
    seen = []

    def handler(req):
        seen.append(req)
        return ok()

    c, _ = make_client(handler)
    assert c.complete_json(MSGS, SCHEMA) == GOOD
    req = seen[0]
    assert req.method == "POST"
    assert req.url.path == f"/v1beta/models/{DEFAULT_MODEL}:generateContent"
    assert str(req.url).startswith("https://generativelanguage.googleapis.com/")
    assert req.headers["x-goog-api-key"] == KEY
    assert KEY not in str(req.url) and not req.url.query and "authorization" not in req.headers
    body = json.loads(req.content)
    assert KEY not in req.content.decode()
    assert body["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": '{"a": "1"}'}]}]
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json" and cfg["temperature"] == 0.0
    assert 0 < cfg["maxOutputTokens"] <= 4096
    js = cfg["responseJsonSchema"]
    assert js["type"] == "object" and js["required"] == SCHEMA["required"]
    assert js["properties"]["verdict"]["enum"] == SCHEMA["properties"]["verdict"]["enum"]
    assert "additionalProperties" not in json.dumps(js)  # dropped for Gemini; our validator tolerates extras
    assert "model" not in body and c.requests_made == 1


def test_model_and_base_url_from_env_and_models_prefix(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "models/gemini-x")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.test/")
    seen = []
    c = GeminiClient(api_key=KEY, min_interval_s=0, transport=httpx.MockTransport(lambda r: (seen.append(r), ok())[1]))
    assert c.model == "gemini-x" and c.base_url == "https://example.test"
    c.complete_json(MSGS, SCHEMA)
    assert seen[0].url.host == "example.test" and seen[0].url.path == "/v1beta/models/gemini-x:generateContent"


def test_judge_prompt_goes_through_unchanged():
    seen = []
    c, _ = make_client(lambda r: (seen.append(json.loads(r.content)), ok())[1])
    c.complete_json([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "{}"}], SCHEMA)
    assert seen[0]["systemInstruction"]["parts"][0]["text"] == SYSTEM_PROMPT


# ---------------------------------------------------------------- parsing
def test_thought_parts_are_ignored():
    body = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                   {"text": json.dumps(GOOD)}]}}]}
    c, _ = make_client(lambda r: httpx.Response(200, json=body))
    assert c.complete_json(MSGS, SCHEMA) == GOOD


@pytest.mark.parametrize("content", [
    {"verdict": "REFUND", "risk": 0.1, "rationale": "x"},
    {"verdict": "HOLD", "risk": 0.1},
    {"verdict": "HOLD", "risk": "high", "rationale": "x"},
    "this is not json",
    "[1, 2]",
])
def test_invalid_json_or_schema_violation_raises_and_is_not_retried(content):
    c, clock = make_client(lambda r: ok(content))
    with pytest.raises(LLMSchemaError):
        c.complete_json(MSGS, SCHEMA)
    assert c.requests_made == 1 and clock.sleeps == []


@pytest.mark.parametrize("body", [
    {"candidates": []},
    {},
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": [{"finishReason": "SAFETY"}]},
    {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]},
    {"candidates": [{"content": {"parts": [{"text": "  "}]}, "finishReason": "MAX_TOKENS"}]},
    {"candidates": "nope"},
])
def test_safety_block_or_empty_candidates_is_a_schema_error(body):
    c, clock = make_client(lambda r: httpx.Response(200, json=body))
    with pytest.raises(LLMSchemaError, match="no candidate text"):
        c.complete_json(MSGS, SCHEMA)
    assert c.requests_made == 1 and clock.sleeps == []


def test_non_json_body_is_a_schema_error():
    c, _ = make_client(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(LLMSchemaError):
        c.complete_json(MSGS, SCHEMA)


# ---------------------------------------------------------------- retries
def test_retry_after_header_is_honoured_and_capped():
    seq = iter([httpx.Response(503, headers={"Retry-After": "7"}, text="busy"),
                httpx.Response(429, headers={"Retry-After": "600"}, text="slow"), None])

    def handler(req):
        return next(seq) or ok()

    c, clock = make_client(handler)
    assert c.complete_json(MSGS, SCHEMA) == GOOD
    assert clock.sleeps == [7.0, 60.0] and c.requests_made == 3


def test_retry_info_delay_in_error_body_is_honoured_and_capped():
    seq = iter([rate_429("12s"), rate_429("0.5s"), rate_429("900s"), None])
    c, clock = make_client(lambda r: next(seq) or ok(), retries=3)
    assert c.complete_json(MSGS, SCHEMA) == GOOD
    assert clock.sleeps == [12.0, 0.5, 60.0] and c.requests_made == 4


def test_backoff_without_a_hint_and_giving_up():
    c, clock = make_client(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMHTTPError) as ei:
        c.complete_json(MSGS, SCHEMA)
    assert ei.value.status == 500 and not isinstance(ei.value, LLMQuotaError)
    assert c.requests_made == 4 and clock.sleeps == [2.0, 4.0, 8.0]


@pytest.mark.parametrize("status", [502, 504])
def test_other_transient_statuses_are_retried(status):
    seq = iter([httpx.Response(status, text="x"), None])
    c, _ = make_client(lambda r: next(seq) or ok())
    assert c.complete_json(MSGS, SCHEMA) == GOOD and c.requests_made == 2


def test_client_errors_are_not_retried():
    c, clock = make_client(lambda r: httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT"}}))
    with pytest.raises(LLMHTTPError) as ei:
        c.complete_json(MSGS, SCHEMA)
    assert ei.value.status == 400 and "INVALID_ARGUMENT" in str(ei.value)
    assert c.requests_made == 1 and clock.sleeps == []


def test_transport_errors_are_retried_then_typed():
    def handler(req):
        raise httpx.ConnectError(f"cannot connect (key={KEY})")

    c, _ = make_client(handler, retries=2)
    with pytest.raises(LLMTransportError) as ei:
        c.complete_json(MSGS, SCHEMA)
    assert c.requests_made == 3 and KEY not in str(ei.value)


@pytest.mark.parametrize("resp", [
    quota_429(),
    quota_429("GenerateContentFreeTierRequestsPerDay"),
    quota_429("SomeMetric-FreeTier", value="0"),  # no free quota at all for this model
    httpx.Response(429, json={"error": {"code": "quota_exceeded", "message": "daily quota exceeded"}}),
])
def test_daily_quota_429_raises_quota_error_without_retry(resp):
    c, clock = make_client(lambda r: resp)
    with pytest.raises(LLMQuotaError) as ei:
        c.complete_json(MSGS, SCHEMA)
    assert isinstance(ei.value, LLMHTTPError) and ei.value.status == 429
    assert c.requests_made == 1 and clock.sleeps == []


def test_per_minute_429_is_not_a_quota_error():
    c, _ = make_client(lambda r: rate_429("1s"), retries=1)
    with pytest.raises(LLMHTTPError) as ei:
        c.complete_json(MSGS, SCHEMA)
    assert not isinstance(ei.value, LLMQuotaError) and c.requests_made == 2


# ---------------------------------------------------------------- throttle
def test_minimum_interval_between_requests_is_enforced():
    c, clock = make_client(lambda r: ok(), min_interval_s=6.5)
    start = clock()
    for _ in range(3):
        c.complete_json(MSGS, SCHEMA)
    assert clock.sleeps == [6.5, 6.5] and clock() - start == 13.0  # first call is not delayed


def test_throttle_counts_time_already_elapsed():
    c, clock = make_client(lambda r: ok(), min_interval_s=6.5)
    c.complete_json(MSGS, SCHEMA)
    clock.now += 4.0  # the caller was busy for 4 s
    c.complete_json(MSGS, SCHEMA)
    assert clock.sleeps == [2.5]
    clock.now += 10.0
    c.complete_json(MSGS, SCHEMA)
    assert clock.sleeps == [2.5]  # no wait needed


def test_retries_are_throttled_too():
    seq = iter([httpx.Response(503, headers={"Retry-After": "1"}, text="x"), None])
    c, clock = make_client(lambda r: next(seq) or ok(), min_interval_s=6.5)
    c.complete_json(MSGS, SCHEMA)
    assert clock.sleeps == [1.0, 5.5]  # server hint, then top up to the 6.5 s spacing


def test_default_interval_and_env_override(monkeypatch):
    assert GeminiClient(api_key=KEY).min_interval_s == 6.5
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "2.5")
    assert GeminiClient(api_key=KEY).min_interval_s == 2.5
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "soon")
    with pytest.raises(LLMConfigError, match="GEMINI_MIN_INTERVAL_S"):
        GeminiClient(api_key=KEY)


# ---------------------------------------------------------------- no leaks
def test_api_key_never_appears_in_errors_repr_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    errors = []
    for resp in (quota_429(), httpx.Response(400, text=f"invalid key {KEY}"),
                 httpx.Response(503, text=f"upstream said {KEY}"),
                 httpx.Response(200, json={"candidates": [], "echo": KEY})):
        c, _ = make_client(lambda r, resp=resp: resp, retries=1)
        with pytest.raises(Exception) as ei:
            c.complete_json(MSGS, SCHEMA)
        errors.append(ei.value)
    for e in errors:
        assert KEY not in str(e) and KEY not in repr(e)
    assert KEY not in repr(c) and KEY not in caplog.text
    # The response body (here the "message" text) is not echoed either, only the short status enum.
    assert "quota used up" not in str(errors[0]) and "RESOURCE_EXHAUSTED" in str(errors[0])


def test_client_without_a_key_names_the_variable_not_a_value():
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        GeminiClient()


# ---------------------------------------------------------------- factory / judge wiring
def test_factory_gemini_without_key_is_cache_only_keyed_by_gemini_model(tmp_path):
    cache = LLMCache(tmp_path / "g.jsonl")
    a = make_assessor("gemini", cache=cache)
    assert a.offline and a.client is None and a.model == DEFAULT_MODEL and a.max_calls == 0
    cache.put(cache_key(DEFAULT_MODEL, PROMPT_VERSION, PAYLOAD), GOOD, model=DEFAULT_MODEL,
              prompt_version=PROMPT_VERSION, payload=PAYLOAD)
    got = a.assess_payload(PAYLOAD)
    assert got.source == "cache" and got.verdict is Verdict.STEP_UP and a.calls == 0
    other = dict(PAYLOAD, account_age=">60d")
    assert a.assess_payload(other).source == "fallback" and a.fallbacks == {"offline": 1}


def test_factory_gemini_model_from_env_offline(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env")
    assert make_assessor("gemini", cache=LLMCache(None)).model == "gemini-env"


def test_factory_gemini_with_key_builds_client_but_makes_no_call_by_default(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    a = make_assessor("gemini", cache=LLMCache(None))
    assert isinstance(a.client, GeminiClient) and not a.offline and a.max_calls == 0
    assert a.client.requests_made == 0 and a.model == a.client.model == DEFAULT_MODEL
    got = a.assess_payload(PAYLOAD)
    assert got.source == "fallback" and a.fallbacks == {"cap": 1} and a.client.requests_made == 0
    assert KEY not in repr(a.client) and KEY not in json.dumps(a.stats())


def test_factory_gemini_offline_flag_wins_over_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    a = make_assessor("gemini", cache=LLMCache(None), offline=True)
    assert a.offline and a.client is None
    monkeypatch.setenv("PAYGUARD_LLM_OFFLINE", "1")
    assert make_assessor("gemini", cache=LLMCache(None)).client is None


def test_factory_gemini_never_builds_a_sarvam_client(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "sarvam-key-present")  # only Sarvam is configured
    a = make_assessor("gemini", cache=LLMCache(None))
    assert a.client is None and a.offline and a.model == DEFAULT_MODEL


def test_factory_gemini_uses_its_own_cache_file(tmp_path, monkeypatch):
    path = tmp_path / "llm_cache_gemini.jsonl"
    monkeypatch.setenv("PAYGUARD_LLM_CACHE_GEMINI", str(path))
    monkeypatch.setenv("PAYGUARD_LLM_CACHE", str(tmp_path / "sarvam.jsonl"))
    assert default_gemini_cache_path() == path and default_cache_path() == tmp_path / "sarvam.jsonl"

    class Fake:
        model = "fake-gemini"

        def complete_json(self, messages, schema):
            return dict(GOOD)

    a = make_assessor("gemini", client=Fake(), max_calls=5)
    assert a.cache.path == path and a.model == "fake-gemini"
    assert a.assess_payload(PAYLOAD).source == "api"
    assert path.exists() and not (tmp_path / "sarvam.jsonl").exists()


def test_default_gemini_cache_path_is_separate_and_not_created():
    p = default_gemini_cache_path()
    assert p.name == "llm_cache_gemini.jsonl" and p != default_cache_path() and p.parent.name == "data"
    make_assessor("gemini")  # building the assessor reads, never creates, the file
    assert LLMCache(p).path == p


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="provider"):
        make_assessor("openai")


def test_sarvam_cache_key_and_default_assessor_are_unchanged(monkeypatch):
    assert cache_key("sarvam-105b-conversations", PROMPT_VERSION, PAYLOAD) == SARVAM_KEY_BEFORE
    a = LLMAssessor(cache=LLMCache(None))
    assert a.model == "sarvam-105b-conversations" and a.offline and a.max_calls == 0
    b = make_assessor(cache=LLMCache(None))  # default provider is sarvam
    assert b.model == a.model and b.offline and b.cache.path is None
    assert default_cache_path().name == "llm_cache.jsonl"


def test_names_for_a_second_judge():
    shared = LLMAssessor(cache=LLMCache(None))
    assert LLMJudge(shared).name == "llm_judge" and QualityDefender(shared).name == "quality"
    assert LLMJudge(shared, name="llm_judge_gemini").name == "llm_judge_gemini"
    assert QualityDefender(shared, name="quality_gemini").name == "quality_gemini"
    assert LLMJudge.name == "llm_judge"  # class default untouched


def test_quota_error_stops_further_client_calls_and_is_not_cached():
    class Quota:
        model = "fake-gemini"
        n = 0

        def complete_json(self, messages, schema):
            self.n += 1
            raise LLMQuotaError(429, "RESOURCE_EXHAUSTED")

    cl = Quota()
    a = LLMAssessor(cl, LLMCache(None), max_calls=100)
    first = a.assess_payload(PAYLOAD)
    second = a.assess_payload(dict(PAYLOAD, account_age=">60d"))
    assert first.source == second.source == "fallback" and first.called and not second.called
    assert cl.n == 1 and a.calls == 1 and a.quota_exhausted and a.fallbacks == {"error": 1, "quota": 1}
    assert len(a.cache) == 0


def test_end_to_end_gemini_judge_over_http_mock_and_cache_replay(tmp_path):
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)

    def handler(req):
        young = "<7d" in json.loads(req.content)["contents"][0]["parts"][0]["text"]
        return ok({"verdict": "HOLD" if young else "RELEASE", "risk": 0.9 if young else 0.05, "rationale": "r"})

    from payguard.sim.runner import run_episode

    c, _ = make_client(handler)
    a1 = make_assessor("gemini", client=c, cache=LLMCache(tmp_path / "g.jsonl"), max_calls=10_000)
    r1 = run_episode(ep, LLMJudge(a1, name="llm_judge_gemini"))
    assert 0 < r1.llm_calls == a1.calls == c.requests_made < 0.10 * len(ep.events)
    assert Verdict.HOLD in set(r1.verdicts.values()) and Verdict.RELEASE in set(r1.verdicts.values())
    a2 = make_assessor("gemini", cache=LLMCache(tmp_path / "g.jsonl"))  # no key, cache only
    r2 = run_episode(ep, LLMJudge(a2, name="llm_judge_gemini"))
    assert r2.verdicts == r1.verdicts and r2.llm_calls == 0


# ---------------------------------------------------------------- scripts.llm_warm --provider gemini
class QuotaAfter:
    model = "fake-gemini"

    def __init__(self, ok_calls):
        self.ok_calls, self.n = ok_calls, 0

    def complete_json(self, messages, schema):
        self.n += 1
        if self.n > self.ok_calls:
            raise LLMQuotaError(429, "RESOURCE_EXHAUSTED")
        return dict(GOOD)


def test_llm_warm_gemini_stops_cleanly_on_quota_and_keeps_the_cache(tmp_path, monkeypatch, capsys):
    from scripts import llm_warm

    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    monkeypatch.setattr(llm_warm, "episodes", lambda seeds, families: [ep])
    path = tmp_path / "g.jsonl"
    client = QuotaAfter(3)
    assessor = make_assessor("gemini", client=client, cache=LLMCache(path), max_calls=500)
    assert llm_warm.warm(500, "gemini", assessor) == 0
    out = capsys.readouterr().out
    assert "quota exhausted after 4 calls" in out and "cached" in out and "resumes" in out
    assert client.n == 4 and len(LLMCache(path)) == 3  # 3 good replies kept, the quota failure is not cached
    assert "TRAIN obj" not in out  # no results table for the aborted run

    # Re-running later resumes from the cache: the 3 cached payloads cost no calls.
    client2 = QuotaAfter(10_000)
    a2 = make_assessor("gemini", client=client2, cache=LLMCache(path), max_calls=500)
    llm_warm.warm(500, "gemini", a2)
    assert a2.cache_hits > 0 and len(LLMCache(path)) == 3 + a2.calls


def test_llm_warm_gemini_smoke_stops_cleanly_on_quota(tmp_path, monkeypatch, capsys):
    from scripts import llm_warm

    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    monkeypatch.setattr(llm_warm, "episodes", lambda seeds, families: [ep])
    client = QuotaAfter(2)
    assessor = make_assessor("gemini", client=client, cache=LLMCache(tmp_path / "g.jsonl"), max_calls=10)
    assert llm_warm.smoke(10, "gemini", assessor) == 0
    out = capsys.readouterr().out
    assert "quota exhausted after 3 calls" in out and client.n == 3 and len(assessor.cache) == 2


def test_llm_warm_progress_every_25_calls():
    from scripts import llm_warm

    lines, t = [], iter(range(0, 10_000, 60))
    a = LLMAssessor(QuotaAfter(10_000), LLMCache(None), max_calls=1000)
    llm_warm.watch(a, total=60, clock=lambda: next(t), out=lines.append)
    for i in range(60):
        a.assess_payload({**PAYLOAD, "n": str(i)})
    assert len(lines) == 2 and lines[0].startswith("progress: 25/60 calls, 0 cache hits")
    assert "elapsed" in lines[0] and "min left" in lines[0] and lines[1].startswith("progress: 50/60")


def test_llm_warm_provider_flag_defaults_to_sarvam_and_needs_a_key(monkeypatch):
    from scripts import llm_warm

    with pytest.raises(SystemExit) as ei:
        llm_warm.make_assessor(5, "gemini")
    assert "GEMINI_API_KEY" in str(ei.value)
    with pytest.raises(SystemExit) as ei:
        llm_warm.make_assessor(5)
    assert "SARVAM_API_KEY" in str(ei.value)
    monkeypatch.setattr("sys.argv", ["llm_warm", "--warm", "1", "--provider", "bogus"])
    with pytest.raises(SystemExit):
        llm_warm.main()
