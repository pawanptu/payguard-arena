"""Day-0 gate: can Sarvam return schema-valid JSON reliably, and at what latency/cost?

Run from the repo root:  python -m scripts.smoke_sarvam

Sends a few borderline payout cases N times with response_format=json_schema and
reports validity rate, latency and token usage (reasoning tokens are billed as
completion tokens, so watch completion_tokens). Never prints the API key.
Kill criterion: if validity is clearly below 100% the LLM defender drops to cached/templated.
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import httpx

from payguard.config import MissingConfig, optional, require

DECISIONS = {"RELEASE", "HOLD", "STEP_UP"}

SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": sorted(DECISIONS)},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "confidence", "reason"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a payout-fraud defender on an instant-pay gig platform. Given one payout's "
    "features, answer RELEASE, HOLD or STEP_UP (ask the worker for extra verification). "
    "Reply with JSON only."
)

CASES = [
    {"worker_age_days": 2, "payout_usd": 480, "payouts_last_24h": 6, "shared_device_with_workers": 3},
    {"worker_age_days": 400, "payout_usd": 55, "payouts_last_24h": 1, "shared_device_with_workers": 0},
    {"worker_age_days": 35, "payout_usd": 190, "payouts_last_24h": 3, "shared_device_with_workers": 1},
]
ROUNDS = 2


def valid(parsed: object) -> bool:
    return (
        isinstance(parsed, dict)
        and parsed.get("decision") in DECISIONS
        and isinstance(parsed.get("confidence"), (int, float))
        and 0 <= float(parsed["confidence"]) <= 1
        and isinstance(parsed.get("reason"), str)
        and bool(parsed["reason"].strip())
    )


def main() -> int:
    try:
        cfg = require("SARVAM_API_KEY")
    except MissingConfig as exc:
        print(f"FAIL config: {exc}")
        return 2
    model = optional("SARVAM_MODEL", "sarvam-105b")
    base = optional("SARVAM_BASE_URL", "https://api.sarvam.ai")
    headers = {
        # Docs list both headers; sending both avoids guessing which is enforced.
        "api-subscription-key": cfg["SARVAM_API_KEY"],
        "Authorization": f"Bearer {cfg['SARVAM_API_KEY']}",
        "Content-Type": "application/json",
    }

    ok = total = 0
    latencies: list[float] = []
    completion_tokens: list[int] = []
    with httpx.Client(base_url=base, timeout=60.0) as http:
        for _ in range(ROUNDS):
            for case in CASES:
                total += 1
                body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(case)},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 600,
                    "reasoning_effort": "low",
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "defender_verdict", "schema": SCHEMA},
                    },
                }
                t0 = time.perf_counter()
                resp = http.post("/v1/chat/completions", json=body, headers=headers)
                dt = time.perf_counter() - t0
                if resp.status_code >= 400:
                    print(f"  call {total}: HTTP {resp.status_code} {resp.text[:300]}")
                    continue
                data = resp.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                latencies.append(dt)
                completion_tokens.append(int(usage.get("completion_tokens", 0)))
                try:
                    parsed = json.loads(choice["message"].get("content") or "")
                except json.JSONDecodeError:
                    parsed = None
                good = valid(parsed)
                ok += good
                print(
                    f"  call {total}: valid={good} finish={choice.get('finish_reason')} "
                    f"latency={dt:.1f}s completion_tokens={usage.get('completion_tokens')} "
                    f"decision={(parsed or {}).get('decision')}"
                )

    print(f"\nschema-valid {ok}/{total} model={model}")
    if latencies:
        print(
            f"latency median={statistics.median(latencies):.1f}s max={max(latencies):.1f}s; "
            f"completion_tokens median={statistics.median(completion_tokens)}"
        )
    if total and ok == total:
        print("PASS: structured JSON reliable on this sample.")
        return 0
    print("FAIL/INCONCLUSIVE: see kill criterion (drop to cached/templated rationale).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
