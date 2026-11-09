"""The three-rounds document (payguard/console/rounds.py): read-only from the four result files, fixed caveats."""
from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import pytest

from payguard.console import rounds as R

ROOT = Path(__file__).resolve().parents[1]
FILES = (R.R1_PATH, R.R2_PATH, R.R2_GEMINI_PATH, R.R3_PATH)


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def doc():
    return R.rounds_doc(ROOT)


# the round-2 split's name is read from its result file, never spelled out in a test
SPLIT2 = next(iter(json.loads((ROOT / R.R2_PATH).read_text(encoding="utf-8"))["splits"]))


def row(rows, name):
    return next(r for r in rows if r["defender"] == name)


def file_row(rel, defender, split):
    return next(r for r in load(rel)["rows"] if r["defender"] == defender and r["split"] == split)


def test_top_level_shape(doc):
    assert doc["schema_version"] == 1 and doc["simulated"] is True and "lower is better" in doc["objective"]
    assert [r["round"] for r in doc["rounds"]] == [1, 2, 3]
    json.dumps(doc, allow_nan=False)  # plain JSON types only


def test_round_headers(doc):
    r1, r2, r3 = doc["rounds"]
    assert (r1["split"], r1["family"], r1["verdict"], r1["tone"], r1["decision_ref"], r1["primary"]) == (
        "TEST_HELDOUT", "sleeper", "NO-GO", "fail", "D-033", "swarm_evolved_memory")
    assert (r2["split"], r2["family"], r2["verdict"], r2["tone"], r2["decision_ref"], r2["primary"]) == (
        SPLIT2, "account takeover (h2)", "GO with caveats", "pass", "D-037", "swarm_r2")
    assert (r3["family"], r3["verdict"], r3["tone"], r3["decision_ref"], r3["primary"]) == (
        "synthetic-identity account farm (h3)", "H1 GO, H2 FAIL", "mixed", "D-046", "dreamer_full")
    assert r3["split"] == load(R.R3_PATH)["split"]["name"]  # read from the file
    assert "rule" in r1 and "rule" in r2 and "H1" in r3["rule"] and "H2" in r3["rule"]
    for r in doc["rounds"]:
        assert r["source"].startswith("eval/final")


def test_round1_rows_are_the_test_heldout_rows_with_roles(doc):
    r1 = doc["rounds"][0]
    want = [r for r in load(R.R1_PATH)["rows"] if r["split"] == "TEST_HELDOUT"]
    assert [r["defender"] for r in r1["rows"]] == [r["defender"] for r in want]
    roles = {r["defender"]: r["role"] for r in r1["rows"]}
    assert roles["swarm_evolved_memory"] == "primary"
    for n in ("static_rules", "single_policy", "always_release", "always_hold", "single_graph", "single_drift",
              "single_velocity"):
        assert roles[n] == "baseline", n
    for n in ("swarm_frozen", "swarm_l1", "swarm_evolved", "swarm_memory"):
        assert roles[n] == "reference", n
    assert roles["llm_judge_sarvam"] == roles["llm_judge_gemini"] == "llm"
    assert list(roles.values()).count("primary") == 1
    for r in r1["rows"]:
        src = next(x for x in want if x["defender"] == r["defender"])
        assert all(r[k] == src[k] for k in ("objective", "leakage_pct", "honest_block_rate"))


def test_round2_rows_are_the_round2_split_plus_the_gemini_row(doc):
    r2 = doc["rounds"][1]
    names = [r["defender"] for r in r2["rows"]]
    assert names[-1] == "llm_judge_gemini" and len(names) == len(set(names))
    want = [r["defender"] for r in load(R.R2_PATH)["rows"] if r["split"] == SPLIT2]
    assert names[:-1] == want
    roles = {r["defender"]: r["role"] for r in r2["rows"]}
    assert roles["swarm_r2"] == "primary" and roles["swarm_evolved_memory"] == "reference"
    assert roles["static_rules"] == roles["single_policy"] == "baseline" and roles["llm_judge_gemini"] == "llm"
    g = load(R.R2_GEMINI_PATH)["rows"][0]
    assert row(r2["rows"], "llm_judge_gemini")["objective"] == g["objective"]


def test_round3_rows_hypotheses_paired_report_only_and_spent_splits(doc):
    r3, d = doc["rounds"][2], load(R.R3_PATH)
    roles = {r["defender"]: r["role"] for r in r3["rows"]}
    assert roles == {"dreamer_full": "primary", "static_rules": "baseline", "single_policy": "baseline",
                     "swarm_r2": "reference", "dreamer_nowm": "ablation", "dreamer_noskeptic": "ablation",
                     "dreamer_nocalib": "ablation"}
    for r in r3["rows"]:
        assert r["objective"] == d["arms"][r["defender"]]["pooled"]["objective"]
    h = {x["id"]: x for x in r3["hypotheses"]}
    assert h["H1"]["verdict"] == "GO" and h["H2"]["verdict"] == "FAIL"
    assert h["H1"]["numbers"]["margin_vs_static"] == d["decision"]["h1"]["margin_vs_static"]
    assert h["H2"]["numbers"]["margin_vs_nowm"] == d["decision"]["h2"]["margin_vs_nowm"]
    paired = {p["arm"]: p for p in r3["paired"]}
    assert set(paired) == set(d["paired_margin_vs_dreamer_full"]) and "dreamer_full" not in paired
    assert paired["swarm_r2"]["mean"] == d["paired_margin_vs_dreamer_full"]["swarm_r2"]["mean"] < 0
    assert len(paired["swarm_r2"]["ci95"]) == 2
    ro = r3["report_only"]
    assert ro["immunity_attempts"]["dreamer_full"] == d["report_only"]["h4_time_to_immunity"]["dreamer_full"][
        "immunity_attempts"]
    assert set(ro["never_contained_rate"]) == set(d["arms"]) and "dreamer_full" in ro["calibration_trend"]
    assert list(r3["spent_splits"]) == ["TRAIN", "TEST_SEEN", "TEST_HELDOUT", SPLIT2]
    for s, v in r3["spent_splits"].items():
        assert v["objective"] == d["spent_splits"][s]["pooled"]["objective"] and v["clean"] is d["spent_splits"][s]["clean"]
        assert v["clean"] is False


# ---------------------------------------------------------------- caveats ---------------------------------
def decimals(text):
    return re.findall(r"\d+\.\d+", text)


def r2f(x):
    return f"{x:.2f}"


def test_caveat_texts_are_the_fixed_strings(doc):
    assert [r["caveats"] for r in doc["rounds"]] == [R.CAVEATS[1], R.CAVEATS[2], R.CAVEATS[3]]
    assert len(R.CAVEATS[1]) == 2 and len(R.CAVEATS[2]) == 3 and len(R.CAVEATS[3]) == 4
    assert R.CAVEATS[1][0].startswith("The evolved swarm with memory scored 80.56 against static rules' 28.54")
    assert R.CAVEATS[3][3].endswith("it cannot be tested on it.")


def test_every_decimal_in_the_round1_caveats_matches_the_source_files():
    r1 = load(R.R1_PATH)
    held = lambda n: file_row(R.R1_PATH, n, "TEST_HELDOUT")["objective"]  # noqa: E731
    seen = lambda n: file_row(R.R1_PATH, n, "TEST_SEEN")["objective"]  # noqa: E731
    want = [[r2f(held("swarm_evolved_memory")), r2f(held("static_rules"))],
            [r2f(seen("swarm_evolved_memory")), r2f(seen("static_rules"))]]
    assert [decimals(c) for c in R.CAVEATS[1]] == want == [["80.56", "28.54"], ["17.14", "65.64"]]
    # the go_no_go block agrees with the rows
    chk = r1["go_no_go"]["checks"]
    assert r2f(chk["TEST_HELDOUT"]["objective"]) == "80.56" and r2f(chk["TEST_SEEN"]["static_rules"]) == "65.64"


def test_every_decimal_in_the_round2_caveats_matches_the_source_files():
    graph = file_row(R.R2_PATH, "single_graph", SPLIT2)["objective"]
    r2 = file_row(R.R2_PATH, "swarm_r2", SPLIT2)["objective"]
    sleeper_r2 = file_row(R.R2_PATH, "swarm_r2", "TEST_HELDOUT")["objective"]
    static_r1 = file_row(R.R1_PATH, "static_rules", "TEST_HELDOUT")["objective"]
    want = [[r2f(graph), r2f(graph - r2), r2f(r2)], [], [r2f(sleeper_r2), r2f(static_r1)]]
    assert [decimals(c) for c in R.CAVEATS[2]] == want == [["28.87", "2.57", "26.30"], [], ["45.10", "28.54"]]
    assert r2f(graph) == "28.87" and r2f(r2) == "26.30"
    assert r2f(28.87 - 26.30) == "2.57"  # the difference of the two rounded numbers agrees too
    # "every round-1 swarm also cleared the rule on this family": the secondary pass flags in the file say so
    secondary = load(R.R2_PATH)["go_no_go"]["secondary_pass_not_decisive"]
    assert all(secondary[s] for s in ("swarm_frozen", "swarm_l1", "swarm_evolved", "swarm_memory",
                                      "swarm_evolved_memory"))


def test_every_decimal_in_the_round3_caveats_matches_the_source_files():
    d = load(R.R3_PATH)
    leak = d["arms"]["static_rules"]["pooled"]["leakage_pct"]
    assert [decimals(c) for c in R.CAVEATS[3]] == [["97.6"], [], [], []]
    assert f"{leak:.1f}" == "97.6" and decimals(R.CAVEATS[3][0])[0] == f"{leak:.1f}"
    assert R.CAVEATS[3][0].split("static rules let ")[1].startswith(f"{leak:.1f}%")


# ---------------------------------------------------------------- guard against stale caveats -------------
@pytest.fixture
def copy_root(tmp_path):
    for rel in FILES:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dest)
    return tmp_path


def test_rounds_doc_reads_only_from_the_given_root(copy_root):
    assert R.rounds_doc(copy_root) == R.rounds_doc(ROOT)


@pytest.mark.parametrize("rel,mutate", [
    (R.R1_PATH, lambda d: d["go_no_go"].update(verdict="GO")),
    (R.R2_PATH, lambda d: d["go_no_go"].update(verdict="NO-GO")),
    (R.R3_PATH, lambda d: d["decision"]["h2"].update({"pass": True})),
])
def test_a_changed_verdict_in_a_result_file_is_refused(copy_root, rel, mutate):
    path = copy_root / rel
    d = json.loads(path.read_text(encoding="utf-8"))
    mutate(d)
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed round text"):
        R.rounds_doc(copy_root)


def test_a_missing_result_file_raises_file_not_found(copy_root):
    (copy_root / R.R2_GEMINI_PATH).unlink()
    with pytest.raises(FileNotFoundError):
        R.rounds_doc(copy_root)


def test_source_files_are_not_modified():
    before = {rel: (ROOT / rel).read_bytes() for rel in FILES}
    R.rounds_doc(ROOT)
    assert before == {rel: (ROOT / rel).read_bytes() for rel in FILES}


def test_rounds_doc_does_not_mutate_the_shared_caveat_lists(doc):
    doc["rounds"][0]["caveats"].append("x")
    assert len(R.CAVEATS[1]) == 2 and copy.deepcopy(R.rounds_doc(ROOT))["rounds"][0]["caveats"] == R.CAVEATS[1]
