"""The three pre-registered rounds for the ops console's Results view (docs/CONSOLE_CONTRACT.md, section D).

`rounds_doc(repo_root)` reads four result files and nothing else (read-only):
  eval/final/results.json, eval/final_r2/results.json, eval/final_r2/gemini_row.json, eval/final_r3/results.json
Verdict labels, tone and caveat sentences are FIXED text written by the director, never generated. Each round also
checks that the verdict recorded in its result file is the one the fixed text was written for, and raises
ValueError when a file disagrees (stale caveats must not be shown). The sealed round-3 split name is read from the
results file, not named here.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1
OBJECTIVE_TEXT = "leakage % + 0.5 x honest-block % + honest-unpaid % (lower is better)"
R1_PATH = "eval/final/results.json"
R2_PATH = "eval/final_r2/results.json"
R2_GEMINI_PATH = "eval/final_r2/gemini_row.json"
R3_PATH = "eval/final_r3/results.json"

CAVEATS = {
    1: ["The evolved swarm with memory scored 80.56 against static rules' 28.54 on the sleeper family it had never "
        "seen, so the pre-registered rule failed.",
        "On the attack families it trained on (TEST_SEEN) the same swarm passed easily: 17.14 against 65.64."],
    2: ["The Graph specialist alone scored 28.87, only 2.57 behind swarm_r2 (26.30).",
        "Every round-1 swarm also cleared the rule on this family, so it was an easier test than round 1.",
        "swarm_r2 still loses on the round-1 sleeper family: 45.10 against static rules' 28.54 (measured after the "
        "fact; that failure was known while swarm_r2 was designed)."],
    3: ["H1 is a low bar on this family: static rules let 97.6% of it through.",
        "The training check (K1) passed narrowly: two of four folds ended with identical defenders, and the whole "
        "margin came from the other two.",
        "Calibration has only 3 promotion readings per run, too few to judge.",
        "Our explanation (dreams learned only from the known families made the defender over-specialise) is a "
        "guess; the sealed family is spent, so it cannot be tested on it."],
}
RULE_R1_R2 = "primary objective >= 3.0 below static_rules and single_policy, honest block <= static_rules'"

ROW_FIELDS = ("objective", "leakage_pct", "honest_block_rate", "honest_unpaid_rate", "immunity_attempts",
              "never_contained_rate")


def _load(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _row(defender: str, role: str, metrics: dict) -> dict:
    return {"defender": defender, "role": role, **{k: metrics[k] for k in ROW_FIELDS}}


def _role_r1_r2(name: str, primary: str) -> str:
    if name == primary:
        return "primary"
    if name in ("static_rules", "single_policy", "always_release", "always_hold") or name.startswith("single_"):
        return "baseline"
    if name.startswith("llm_judge_"):
        return "llm"
    if name.startswith("swarm_"):
        return "reference"
    raise ValueError(f"no role for defender {name!r}")


def _expect(cond: bool, what: str) -> None:
    if not cond:
        raise ValueError(f"result file no longer matches the fixed round text: {what}")


def _round1(root: Path) -> dict:
    d = _load(root, R1_PATH)
    _expect(d["go_no_go"]["verdict"] == "NO-GO" and d["go_no_go"]["primary"] == "swarm_evolved_memory",
            "round 1 verdict is NO-GO for swarm_evolved_memory")
    primary = d["go_no_go"]["primary"]
    rows = [_row(r["defender"], _role_r1_r2(r["defender"], primary), r) for r in d["rows"]
            if r["split"] == "TEST_HELDOUT"]
    return {"round": 1, "title": "Round 1: sleeper family", "split": "TEST_HELDOUT", "family": "sleeper",
            "verdict": "NO-GO", "tone": "fail", "decision_ref": "D-033", "rule": RULE_R1_R2, "primary": primary,
            "rows": rows, "caveats": list(CAVEATS[1]), "source": R1_PATH}


def _round2(root: Path) -> dict:
    d = _load(root, R2_PATH)
    g = _load(root, R2_GEMINI_PATH)
    _expect(d["go_no_go"]["verdict"] == "GO" and d["go_no_go"]["primary"] == "swarm_r2",
            "round 2 verdict is GO for swarm_r2")
    primary = d["go_no_go"]["primary"]
    split = "TEST_HELDOUT2"
    rows = [_row(r["defender"], _role_r1_r2(r["defender"], primary), r) for r in [*d["rows"], *g["rows"]]
            if r["split"] == split]
    return {"round": 2, "title": "Round 2: account takeover", "split": split, "family": "account takeover (h2)",
            "verdict": "GO with caveats", "tone": "pass", "decision_ref": "D-037", "rule": RULE_R1_R2,
            "primary": primary, "rows": rows, "caveats": list(CAVEATS[2]),
            "source": f"{R2_PATH}, {R2_GEMINI_PATH}"}


_R3_ROLES = {"dreamer_full": "primary", "static_rules": "baseline", "single_policy": "baseline",
             "swarm_r2": "reference", "dreamer_nowm": "ablation", "dreamer_noskeptic": "ablation",
             "dreamer_nocalib": "ablation"}


def _round3(root: Path) -> dict:
    d = _load(root, R3_PATH)
    dec = d["decision"]
    _expect(dec["h1"]["pass"] is True and dec["h2"]["pass"] is False, "round 3 is H1 GO, H2 FAIL")
    _expect(set(d["arms"]) == set(_R3_ROLES), "round 3 has the seven expected arms")
    rows = []
    for name, role in _R3_ROLES.items():
        row = _row(name, role, d["arms"][name]["pooled"])
        row["objective_ci95"] = d["arms"][name]["ci95"]["objective"]
        rows.append(row)
    hyps = []
    for hid, key in (("H1", "h1"), ("H2", "h2")):
        h = dec[key]
        numbers = {k: v for k, v in h.items() if k not in ("rule", "pass")}
        hyps.append({"id": hid, "verdict": "GO" if h["pass"] else "FAIL", "text": h["rule"], "numbers": numbers})
    paired = [{"arm": arm, "mean": v["mean"], "ci95": v["ci95"]}
              for arm, v in d["paired_margin_vs_dreamer_full"].items()]
    ro = d["report_only"]
    report_only = {
        "immunity_attempts": {a: v["immunity_attempts"] for a, v in ro["h4_time_to_immunity"].items()},
        "never_contained_rate": {a: v["never_contained_rate"] for a, v in ro["h4_time_to_immunity"].items()},
        "calibration_trend": ro["h3_calibration_gap_trend"],
    }
    spent = {s: {"objective": v["pooled"]["objective"], "clean": v["clean"], "why_not_clean": v["why_not_clean"]}
             for s, v in d["spent_splits"].items()}
    return {"round": 3, "title": "Round 3: synthetic-identity account farm", "split": d["split"]["name"],
            "family": "synthetic-identity account farm (h3)", "verdict": "H1 GO, H2 FAIL", "tone": "mixed",
            "decision_ref": "D-046", "rule": f"H1: {dec['h1']['rule']}. H2: {dec['h2']['rule']}.",
            "primary": "dreamer_full", "rows": rows, "hypotheses": hyps, "paired": paired,
            "report_only": report_only, "spent_splits": spent, "caveats": list(CAVEATS[3]), "source": R3_PATH}


def rounds_doc(repo_root: str | Path) -> dict:
    root = Path(repo_root)
    return {"schema_version": SCHEMA_VERSION, "simulated": True, "objective": OBJECTIVE_TEXT,
            "rounds": [_round1(root), _round2(root), _round3(root)]}
