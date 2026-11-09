"""AI explanations of the queue's cases (docs/CONSOLE_CONTRACT.md, section B).

The LLM is given ONLY decision-time evidence (`case_evidence`: scores, weights, share of the alarm, anomaly-check
statistics, platform features, amount, verdict, thresholds). Ground truth and outcomes never enter the prompt.

Every reply is checked by `check_grounding`, a deterministic validator: each number in the summary, the evidence
lines and the next-check sentence must appear in the case's evidence (rounded to 0, 1 or 2 decimals, also as a
percentage of a 0-1 value); the summary has at most 60 words, there are 1-3 evidence lines, and no jargon such as
"vote share" or "z-score". A reply that fails is replaced by `template_explanation`, built only from the same
evidence, and marked source "template". Raw replies are cached (payguard.llm.cache) so a rerun makes no new call.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

from ..llm.cache import LLMCache, cache_key
from ..llm.errors import LLMError, LLMQuotaError
from ..robust.layer import STAT_NAMES
from .queue import SPECIALISTS

PROMPT_VERSION = "explain-v1"
GROUNDING_TEXT = ("Every number in summary and evidence must appear in the case's decision-time evidence "
                  "(after rounding); otherwise the template text is used and source is 'template'.")
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "llm_cache_explain.jsonl"
ACTIONS = ("HOLD", "STEP_UP", "RELEASE")
MAX_SUMMARY_WORDS = 60
JARGON = ("vote share", "vote-share", "z-score", "zscore", "z score", "sigma", "quantile", "l1 weight")
MAX_CONSECUTIVE_ERRORS = 5

SPECIALIST_LABEL = {s["key"]: s["label"] for s in SPECIALISTS}
SPECIALIST_WHAT = {s["key"]: s["what"] for s in SPECIALISTS}

# Plain-language names of the anomaly layer's nine statistics (payguard.robust.layer.STAT_NAMES).
STAT_PLAIN = {
    "window_usd": "USD paid out in the last 24 h",
    "window_n": "number of payouts in the last 24 h",
    "amount": "size of this single payout",
    "amount_ratio": "amount compared with the account's usual payout",
    "amount_over_cap": "amount against the limit for the account's age",
    "window_over_cap": "24 h volume against the limit for the account's age",
    "rate_z": "payout rate far above the account's own normal rate",
    "graph_cluster": "a tight cluster of linked accounts",
    "sharing": "a device or destination shared by several accounts",
}
assert set(STAT_PLAIN) == set(STAT_NAMES)
# The rolling window in the feature definitions is 24 hours: that number is part of the evidence's meaning.
CONSTANT_NUMBERS = (24.0,)
SMALL_INTS = (1.0, 2.0, 3.0)

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "check_next": {"type": "string", "minLength": 1},
        "suggested_action": {"type": "string", "enum": list(ACTIONS)},
    },
    "required": ["summary", "evidence", "check_next", "suggested_action"],
}

SYSTEM_PROMPT = """You help a payments operations reviewer at a payout platform understand why an automated \
defence flagged one payout request. You receive the evidence the system had at the moment it decided, as JSON.

Write a JSON object with exactly these fields:
- summary: at most 60 words, plain English for a payments operations reviewer. Say what looks unusual and why the \
system held the payout or asked for extra verification.
- evidence: 1 to 3 short strings, each one concrete fact taken from the evidence (for example which check raised \
the alarm and the figures behind it).
- check_next: one sentence saying what a human reviewer should verify before releasing the payout.
- suggested_action: one of HOLD, STEP_UP, RELEASE.

Rules:
- Use only numbers that appear in the evidence. Do not compute new numbers, sums or differences. Copy figures as \
given, or rounded to at most two decimals.
- Plain words only: do not say "vote share", "z-score", "quantile" or similar terms.
- Do not guess whether the payout is fraudulent and do not state anything the evidence does not show.
- The reviewer decides; you only explain and recommend."""


# ---------------------------------------------------------------- evidence (decision-time only) ------------
def case_evidence(case: dict, thresholds: dict) -> dict:
    """What the LLM sees about one case. Built from decision-time fields only: no `truth`, no `outcome`."""
    f = case["features"]
    return {
        "payout": {"event_id": case["event_id"], "amount_usd": case["amount"], "worker_id": case["worker_id"],
                   "requester_id": case["requester_id"]},
        "system_verdict": case["verdict"],
        "thresholds": {"step_up_at_risk": thresholds["step_up"], "hold_at_risk": thresholds["hold"]},
        "overall_risk": case["risk"],
        "set_by": case["decided_by"],
        "specialists": [
            {"name": SPECIALIST_LABEL[k], "looks_at": SPECIALIST_WHAT[k], "score": case["scores"][k],
             "weight": case["weights"][k], "share_of_alarm": case["vote_share"][k]}
            for k in case["scores"]
        ],
        "anomaly_checks": {"verdict": case["layer"]["verdict"],
                           "fired": [STAT_PLAIN.get(n, n.replace("_", " ")) for n in case["layer"]["fired"]]},
        "account_and_platform": {
            "account_age_days": f["age_days"], "earlier_payouts_by_account": f["n_prior"],
            "payouts_in_last_24h": f["window_n"], "usd_in_last_24h": f["window_usd"],
            "amount_vs_account_usual_ratio": f["amount_ratio"],
            "accounts_sharing_this_device": f["device_workers"],
            "accounts_sharing_this_destination": f["destination_workers"],
            "accounts_paid_through_this_requester": f["requester_workers"],
            "requesters_paying_this_account": f["worker_requesters"],
        },
    }


def build_messages(evidence: dict) -> list[dict]:
    user = "Evidence for one payout request:\n" + json.dumps(evidence, indent=1, sort_keys=False)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- grounding validator ---------------------
_NUM = re.compile(r"(?<![A-Za-z_\d.])\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    """Every number in `text` ("$1,234.50" -> 1234.5, "71%" -> 71). Digits glued to letters (swarm_r2) are skipped."""
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _flatten_numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)] if math.isfinite(obj) else []
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _flatten_numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _flatten_numbers(v)]
    return []


def allowed_numbers(evidence: dict) -> set[float]:
    """Numbers a reply may quote: each evidence number as is and rounded to 0, 1, 2 decimals; a 0-1 value also as
    a percentage (rounded to 0 or 1 decimal); the small integers 1-3; the 24 h window."""
    ok = {round(x, 6) for x in (*SMALL_INTS, *CONSTANT_NUMBERS)}
    for v in _flatten_numbers(evidence):
        for d in (0, 1, 2):
            ok.add(round(round(v, d), 6))
        ok.add(round(v, 6))
        if 0.0 <= v <= 1.0:
            for d in (0, 1):
                ok.add(round(round(v * 100.0, d), 6))
    return ok


def check_grounding(expl: dict, evidence: dict) -> list[str]:
    """Reasons `expl` is not acceptable (empty list = grounded and well-formed)."""
    problems: list[str] = []
    summary, ev, nxt, act = expl.get("summary"), expl.get("evidence"), expl.get("check_next"), expl.get("suggested_action")
    if not isinstance(summary, str) or not summary.strip():
        return ["summary missing"]
    if not (isinstance(ev, list) and 1 <= len(ev) <= 3 and all(isinstance(x, str) and x.strip() for x in ev)):
        return ["evidence must be 1-3 non-empty strings"]
    if not isinstance(nxt, str) or not nxt.strip():
        return ["check_next missing"]
    if act not in ACTIONS:
        problems.append(f"suggested_action {act!r} not one of {ACTIONS}")
    if len(summary.split()) > MAX_SUMMARY_WORDS:
        problems.append(f"summary has {len(summary.split())} words (> {MAX_SUMMARY_WORDS})")
    texts = [summary, *ev, nxt]
    low = " ".join(texts).lower()
    problems += [f"jargon: {j!r}" for j in JARGON if j in low]
    ok = allowed_numbers(evidence)
    for text in texts:
        for n in numbers_in(text):
            if round(n, 6) not in ok:
                problems.append(f"number {n:g} is not in the evidence")
    return problems


# ---------------------------------------------------------------- deterministic template ------------------
def _active(case: dict) -> list[str]:
    """Specialists that speak up (score >= 0.05), strongest share of the alarm first."""
    keys = [k for k, s in case["scores"].items() if s >= 0.05]
    return sorted(keys, key=lambda k: (-case["vote_share"][k], k))


def _pct(x: float) -> int:
    return int(round(x * 100))


def template_explanation(case: dict, thresholds: dict) -> dict:
    """A plain explanation built only from decision-time fields of the case. Passes `check_grounding`."""
    v, risk, f = case["verdict"], case["risk"], case["features"]
    line = thresholds["hold"] if v == "HOLD" else thresholds["step_up"]
    head = "Held for human review" if v == "HOLD" else "Extra verification requested"
    active = _active(case)
    fired = [STAT_PLAIN.get(n, n.replace("_", " ")) for n in case["layer"]["fired"]][:3]
    if risk >= line:
        why = f"overall risk {risk:.2f} reaches the {'hold' if v == 'HOLD' else 'step-up'} line of {line:.2f}"
    else:
        why = (f"overall risk {risk:.2f} is below the {'hold' if v == 'HOLD' else 'step-up'} line of {line:.2f}, "
               "but the anomaly checks flagged this payout")
    parts = [f"{head}: {why}."]
    if active:
        names = " and ".join(f"{SPECIALIST_LABEL[k]} (score {case['scores'][k]:.2f})" for k in active[:2])
        parts.append(f"Raised by the {names} check{'s' if len(active[:2]) > 1 else ''}.")
    if fired:
        parts.append("Anomaly checks fired on " + "; ".join(fired) + ".")
    summary = " ".join(parts)

    lines: list[str] = []
    for k in active[:2]:
        lines.append(f"{SPECIALIST_LABEL[k]}: score {case['scores'][k]:.2f}, "
                     f"{_pct(case['vote_share'][k])}% of the alarm")
    if f["requester_workers"] >= 4:
        lines.append(f"Requester {case['requester_id']} is linked to {int(f['requester_workers'])} accounts")
    if f["device_workers"] > 1:
        lines.append(f"The device is shared by {int(f['device_workers'])} accounts")
    if f["destination_workers"] > 1:
        lines.append(f"The payout destination is shared by {int(f['destination_workers'])} accounts")
    if f["window_n"] > 1:
        lines.append(f"{int(f['window_n'])} payouts totalling ${f['window_usd']:.2f} in the last 24 h")
    if f["amount_ratio"] >= 1.5:
        lines.append(f"This payout is {f['amount_ratio']:.1f} times the account's usual payout")
    if not lines:
        lines.append(f"Payout of ${case['amount']:.2f} from an account aged {f['age_days']:.1f} days")
    return {"summary": summary, "evidence": lines[:3], "check_next": _check_next(active, fired),
            "suggested_action": v}


def _check_next(active: list[str], fired: list[str]) -> str:
    top = active[0] if active else None
    if top == "graph":
        return "Check whether the linked accounts and the requester have a genuine business relationship before releasing."
    if top == "velocity":
        return "Check whether the account's recent payout volume matches its normal activity before releasing."
    if top == "drift":
        return "Check why the account is paid so much faster than its own normal rate before releasing."
    if top == "policy":
        return "Check the account's age and the amount against the limit for new accounts before releasing."
    if fired:
        return "Check the account's history against the flagged anomaly checks before releasing."
    return "Check the account's history and the amount before releasing."


# ---------------------------------------------------------------- orchestration --------------------------
def default_cache_path() -> Path:
    env = os.environ.get("PAYGUARD_LLM_CACHE_EXPLAIN", "").strip()
    return Path(env) if env else DEFAULT_CACHE_PATH


class Explainer:
    """Explains cases, one real LLM call at most per uncached case and never more than `cap` real requests."""

    def __init__(self, thresholds: dict, *, model: str, cache: LLMCache, client: Any | None = None, cap: int = 0,
                 log: Callable[[str], None] | None = None):
        self.thresholds, self.model, self.cache, self.client, self.cap = thresholds, model, cache, client, cap
        self.log = log or (lambda s: None)
        self.calls = 0            # complete_json invocations this run
        self.errors = 0
        self.quota_stop = False
        self.ungrounded = 0       # replies (fresh or cached) rejected by the validator
        self._consecutive_errors = 0

    def key(self, evidence: dict) -> str:
        return cache_key(self.model, PROMPT_VERSION, evidence)

    def is_cached(self, case: dict) -> bool:
        return self.cache.get(self.key(case_evidence(case, self.thresholds))) is not None

    def _requests_made(self) -> int:
        return int(getattr(self.client, "requests_made", self.calls))

    def _can_call(self) -> bool:
        if self.client is None or self.quota_stop or self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            return False
        return self.calls < self.cap and self._requests_made() < self.cap

    def _fetch(self, evidence: dict, key: str) -> dict | None:
        """Cached reply, else (if allowed) one real call. None when there is no reply."""
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if not self._can_call():
            return None
        if hasattr(self.client, "retries"):  # keep the retries inside the cap: total requests never exceed it
            self.client.retries = max(0, min(self.client.retries, self.cap - self._requests_made() - 1))
        self.calls += 1
        try:
            reply = self.client.complete_json(build_messages(evidence), SCHEMA)
        except LLMQuotaError:
            self.quota_stop, self.errors = True, self.errors + 1
            self.log("quota exhausted: stopping the run (a later run resumes from the cache)")
            return None
        except LLMError as exc:
            self.errors += 1
            self._consecutive_errors += 1
            self.log(f"call failed ({type(exc).__name__}); the template is used for this case")
            return None
        self._consecutive_errors = 0
        self.cache.put(key, reply, model=self.model, prompt_version=PROMPT_VERSION, payload=evidence)
        return reply

    def explain(self, case: dict) -> dict:
        evidence = case_evidence(case, self.thresholds)
        reply = self._fetch(evidence, self.key(evidence))
        if reply is not None:
            if not check_grounding(reply, evidence):
                return {"source": "llm", "summary": reply["summary"].strip(),
                        "evidence": [x.strip() for x in reply["evidence"]], "check_next": reply["check_next"].strip(),
                        "suggested_action": reply["suggested_action"]}
            self.ungrounded += 1
        return {"source": "template", **template_explanation(case, self.thresholds)}


def explain_cases(cases: list[dict], explainer: Explainer) -> dict[str, dict]:
    """Explanations keyed by str(event_id), in queue order (the top of the queue gets the calls first)."""
    return {str(c["event_id"]): explainer.explain(c) for c in cases}
