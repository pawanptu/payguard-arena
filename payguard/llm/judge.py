"""LLM defender: one prompt, one schema, one cache; two faces.

  LLMJudge         baseline (c): decides every payout alone (Defender protocol, name "llm_judge").
  QualityDefender  swarm specialist: score(event, log) -> risk in [0, 1] (name "quality").

Both sit on an `LLMAssessor`, which resolves a payout to a verdict + risk in this order:
  1. cache hit (payload -> sha256 key): no API call, ever;
  2. offline mode (PAYGUARD_LLM_OFFLINE=1, or no SARVAM_API_KEY and no injected client): fallback;
  3. spend cap reached (calls >= max_calls, env PAYGUARD_LLM_MAX_CALLS, default 0 = no paid calls): fallback;
  4. real API call; on any LLMError: fallback (not cached, so a later run can retry it).
Only validated API replies are cached. `llm_calls` counts real API calls (logical calls: client retries
are visible in `client.requests_made`). The fallback is deterministic, a function of the bucketed payload only.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter
from dataclasses import dataclass

from ..config import optional
from ..sim.features import PlatformLog
from ..sim.types import Event, Verdict
from .cache import LLMCache, cache_key, default_cache_path
from .client import DEFAULT_MODEL, SarvamClient, validate_schema
from .errors import LLMError, LLMSchemaError
from .payload import COHORT_FIELD, judge_payload, level

PROMPT_VERSION = "v1"
# Paid calls are opt-in: with no explicit cap (argument or PAYGUARD_LLM_MAX_CALLS) every cache miss
# falls back, even when SARVAM_API_KEY is set. A forgotten setting must never spend money.
DEFAULT_MAX_CALLS = 0

SYSTEM_PROMPT = """\
You review payout requests on an instant-payout gig-work platform. Workers complete tasks that requesters approve, then ask to be paid out immediately. You receive one payout request as a JSON object of approximate, bucketed features computed from the platform's history up to now. It contains no names or identifiers.

Fields:
- account_age: age of the worker's account.
- amount_usd: size of this payout request, in USD.
- payouts_24h: number of payout requests by this worker in the last 24 hours, including this one.
- amount_vs_own_median: this amount relative to the worker's own median payout ("n/a" if fewer than 3 earlier payouts).
- workers_sharing_device_or_destination: the larger of the number of distinct workers (this one included) using the same device and the number using the same payout destination.
- workers_per_requester: number of distinct workers who have been paid for work approved by this requester.
- workers_with_similar_requesters: number of workers (this one included) whose set of requesters largely overlaps this worker's.

Choose exactly one action:
- RELEASE: pay now. No cost and no delay, but money that has been paid out cannot be recovered if the payout turns out to be fraudulent.
- STEP_UP: ask the worker for an extra automated verification. The payout is delayed about 4 hours; a legitimate worker normally passes. Cheap, small friction.
- HOLD: send the payout to human review. It is delayed about 24 hours, review is costly, reviewers are right about 90% of the time, and an honest worker who is held is paid late and is sometimes wrongly refused.

Most payout requests are legitimate. Pick the action with the lowest expected total cost given the evidence.
Reply with JSON only: "verdict" (RELEASE, STEP_UP or HOLD), "risk" (your probability, from 0 to 1, that this payout is fraudulent), "rationale" (one short sentence).
"""

# Sent to the API as-is; the range of "risk" is enforced in code (clamped), not in the schema, because
# minimum/maximum support in Sarvam's json_schema mode is unverified.
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
        "risk": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "risk", "rationale"],
    "additionalProperties": False,
}

# ---- deterministic fallback (offline cache miss, spend cap reached, API failure) -----------------
# A transparent additive score over the bucketed payload. Deliberately simple and untuned; its only
# jobs are "never RELEASE everything" and "same payload -> same answer". Points per condition:
_POINTS = (
    ("account_age", lambda lv: 0.25 if lv == 0 else 0.0),                     # account younger than 7 days
    ("amount_usd", lambda lv: 0.4 if lv >= 4 else 0.2 if lv >= 2 else 0.0),   # >= $500 / >= $250
    ("payouts_24h", lambda lv: 0.15 if lv >= 2 else 0.0),                     # 5+ payouts in 24 h
    ("amount_vs_own_median", lambda lv: 0.3 if lv >= 1 else 0.0),             # >= 2x own median
    ("workers_sharing_device_or_destination", lambda lv: 0.3 if lv >= 1 else 0.0),
    (COHORT_FIELD, lambda lv: 0.2 if lv >= 1 else 0.0),                       # 4+ workers, similar requesters
)
FALLBACK_STEP_UP = 0.3
FALLBACK_HOLD = 0.6


def fallback_risk(payload: dict) -> float:
    risk = sum(fn(level(payload, field)) for field, fn in _POINTS)
    if level(payload, "account_age") == 0 and level(payload, "amount_usd") >= 2:
        risk += 0.25  # young account asking for >= $250
    return min(risk, 1.0)


def risk_to_verdict(risk: float) -> Verdict:
    if risk >= FALLBACK_HOLD:
        return Verdict.HOLD
    return Verdict.STEP_UP if risk >= FALLBACK_STEP_UP else Verdict.RELEASE


@dataclass(frozen=True)
class Assessment:
    verdict: Verdict
    risk: float
    rationale: str
    source: str           # "cache" | "api" | "fallback"
    called: bool = False  # True when this very assessment made a real API call


def _parse(reply: dict) -> tuple[Verdict, float, str]:
    validate_schema(reply, SCHEMA)
    risk = float(reply["risk"])
    if risk != risk:  # NaN
        raise LLMSchemaError("risk is NaN")
    return Verdict(reply["verdict"]), min(max(risk, 0.0), 1.0), reply["rationale"]


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class LLMAssessor:
    def __init__(
        self,
        client: SarvamClient | None = None,
        cache: LLMCache | None = None,
        *,
        max_calls: int | None = None,
        offline: bool | None = None,
        model: str | None = None,
    ):
        optional("SARVAM_API_KEY", "")  # loads .env (if any) before the checks below
        if offline is None:
            offline = _truthy("PAYGUARD_LLM_OFFLINE")
        if client is None and not offline:
            if os.environ.get("SARVAM_API_KEY", "").strip():
                client = SarvamClient()
            else:
                offline = True  # no key: cache-only
        self.model = model or getattr(client, "model", None) or optional("SARVAM_MODEL", DEFAULT_MODEL)
        self.client = None if offline else client
        self.offline = offline
        self.cache = cache if cache is not None else LLMCache(default_cache_path())
        env_cap = os.environ.get("PAYGUARD_LLM_MAX_CALLS", "").strip()
        self.max_calls = int(max_calls if max_calls is not None else env_cap or DEFAULT_MAX_CALLS)
        self.calls = 0       # real API calls
        self.cache_hits = 0
        self.fallbacks: Counter[str] = Counter()  # reason -> count: "offline", "cap", "error"
        self._lock = threading.Lock()

    @property
    def fallback_count(self) -> int:
        return sum(self.fallbacks.values())

    def stats(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits, "fallbacks": dict(self.fallbacks),
                "max_calls": self.max_calls, "offline": self.offline, "model": self.model,
                "prompt_version": PROMPT_VERSION}

    def _from_cache(self, key: str) -> Assessment | None:
        hit = self.cache.get(key)
        if hit is None:
            return None
        try:
            v, r, why = _parse(hit)
        except (LLMError, ValueError, TypeError, KeyError):
            return None  # corrupt entry: treat as a miss
        self.cache_hits += 1
        return Assessment(v, r, why, "cache")

    def _fallback(self, payload: dict, reason: str, called: bool = False) -> Assessment:
        self.fallbacks[reason] += 1
        risk = fallback_risk(payload)
        return Assessment(risk_to_verdict(risk), risk, f"fallback({reason})", "fallback", called)

    def assess(self, event: Event, log: PlatformLog) -> Assessment:
        payload = judge_payload(event, log)
        key = cache_key(self.model, PROMPT_VERSION, payload)
        got = self._from_cache(key)
        if got is not None:
            return got
        with self._lock:  # serialize misses so concurrent callers cannot double-spend on one key
            got = self._from_cache(key)
            if got is not None:
                return got
            if self.client is None:
                return self._fallback(payload, "offline")
            if self.calls >= self.max_calls:
                return self._fallback(payload, "cap")
            self.calls += 1  # counted before the request: an attempted call may be billed
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, sort_keys=True)}]
            try:
                reply = self.client.complete_json(messages, SCHEMA)
                v, r, why = _parse(reply)
            except LLMError:
                return self._fallback(payload, "error", called=True)
            self.cache.put(key, reply, model=self.model, prompt_version=PROMPT_VERSION, payload=payload)
            return Assessment(v, r, why, "api", called=True)


class _Face:
    """Shared plumbing: each face counts the real API calls it triggered itself."""

    def __init__(self, assessor: LLMAssessor | None = None, **kw):
        self.assessor = assessor if assessor is not None else LLMAssessor(**kw)
        self.llm_calls = 0

    def _assess(self, event: Event, log: PlatformLog) -> Assessment:
        a = self.assessor.assess(event, log)
        if a.called:
            self.llm_calls += 1
        return a

    def learn(self, event: Event, is_attack: bool) -> None:
        pass  # the model is frozen; adaptation is the swarm's job


class LLMJudge(_Face):
    """Baseline (c): a single cached LLM judge that decides every payout."""

    name = "llm_judge"

    def decide(self, event: Event, log: PlatformLog) -> Verdict:
        return self._assess(event, log).verdict


class QualityDefender(_Face):
    """Swarm specialist: the model's fraud probability for a payout, in [0, 1]."""

    name = "quality"

    def score(self, event: Event, log: PlatformLog) -> float:
        return self._assess(event, log).risk
