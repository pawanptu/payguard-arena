"""Round-2 harness (scripts/final_eval_r2.py). Nothing here builds a sealed episode: the sealed seeds and split
name are only reached through the module's own constants, and a tripwire refuses every sealed seed."""
from __future__ import annotations

import json
import typing

import pytest

import payguard.protocol as round1_protocol
import payguard.sim.engine as engine
from payguard.api.replay import Split
from payguard.sim.metrics import Metrics
from scripts import final_eval_r2 as fe2

SEALED = frozenset(fe2.SEALED_SEEDS)
DECISIVE = fe2.DECISIVE_SPLIT


@pytest.fixture(autouse=True)
def tripwire(monkeypatch):
    real = engine.build_episode

    def guarded(seed, *a, **kw):
        if seed in SEALED:
            raise AssertionError(f"tripwire: a test tried to build sealed seed {seed}")
        return real(seed, *a, **kw)

    monkeypatch.setattr(engine, "build_episode", guarded)
    monkeypatch.setattr(round1_protocol, "build_episode", guarded)


def m(obj: float, block: float = 2.0) -> Metrics:
    """A Metrics whose objective is `obj` (leakage = obj - 0.5*block, nothing unpaid)."""
    leak = obj - 0.5 * block
    return Metrics(n_episodes=1, attack_usd=100.0, leaked_usd=leak, leakage_pct=leak, leakage_pct_by_family={},
                   honest_n=100, honest_block_rate=block, honest_unpaid_rate=0.0, honest_median_ttp_h=0.0,
                   honest_p95_ttp_h=0.0, immunity_attempts=0.0, never_contained_rate=0.0, llm_calls=0)


# ---------------------------------------------------------------- merge_rows ----------------------------
ROUND1 = {("static_rules", "TRAIN"): m(55.4), ("llm_judge_gemini", "TEST_HELDOUT"): m(65.2)}
ROUND2 = {("static_rules", DECISIVE): m(30.0), (fe2.PRIMARY, DECISIVE): m(20.0), (fe2.PRIMARY, "TRAIN"): m(19.0)}
LATE_NAME = f"llm_judge_{fe2.LATE_PROVIDER}"


def test_merge_rows_combines_disjoint_sources():
    table = fe2.merge_rows(("round1", ROUND1), ("round2", ROUND2))
    assert set(table) == set(ROUND1) | set(ROUND2)
    assert table[(fe2.PRIMARY, DECISIVE)] is ROUND2[(fe2.PRIMARY, DECISIVE)]


def test_merge_rows_never_overwrites_an_earlier_row():
    clash = {("static_rules", "TRAIN"): m(1.0)}
    with pytest.raises(ValueError, match="static_rules"):
        fe2.merge_rows(("round1", ROUND1), ("round2", clash))


def test_merge_rows_late_source_adds_only_its_judge_on_the_decisive_split():
    ok = fe2.merge_rows(("round1", ROUND1), ("round2", ROUND2), (fe2.LATE_PROVIDER, {(LATE_NAME, DECISIVE): m(70.0)}))
    assert (LATE_NAME, DECISIVE) in ok
    for bad in ({("static_rules", "TEST_SEEN"): m(1.0)}, {(LATE_NAME, "TRAIN"): m(1.0)}):
        with pytest.raises(ValueError):
            fe2.merge_rows(("round1", ROUND1), ("round2", ROUND2), (fe2.LATE_PROVIDER, bad))


# ---------------------------------------------------------------- freeze guard --------------------------
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "eval" / "r2").mkdir(parents=True)
    robust = tmp_path / "payguard" / "robust"
    robust.mkdir(parents=True)
    (robust / "mech.py").write_bytes(b"X = 1\n")  # bytes: write_text would already emit CRLF on Windows
    champ = tmp_path / "eval" / "r2" / "champion.json"
    champ.write_bytes(b'{"params": {}}\n')
    monkeypatch.setattr(fe2, "ROOT", tmp_path)
    monkeypatch.setattr(fe2, "OUT_DIR", tmp_path / "eval" / "final_r2")
    monkeypatch.setattr(fe2, "FREEZE_PATH", tmp_path / "eval" / "final_r2" / "defender_freeze.json")
    monkeypatch.setattr(fe2, "RESULTS_PATH", tmp_path / "eval" / "final_r2" / "results.json")
    monkeypatch.setattr(fe2, "GEMINI_PATH", tmp_path / "eval" / "final_r2" / "gemini_row.json")
    monkeypatch.setattr(fe2, "R2_CHAMPION_PATH", champ)
    monkeypatch.setattr(fe2, "DEFENDER_DIRS", (robust,))
    return tmp_path


def test_nothing_sealed_runs_before_the_freeze(sandbox):
    with pytest.raises(SystemExit, match="not frozen"):
        fe2.require_frozen()
    with pytest.raises(SystemExit, match="not frozen"):
        fe2.sealed_split_episodes()
    with pytest.raises(SystemExit, match="not frozen"):
        fe2.main(["--plan"])


def test_freeze_is_once_and_detects_any_change(sandbox):
    assert fe2.main(["--freeze-defender"]) == 0
    assert fe2.main(["--freeze-defender"]) == 2  # once
    assert "eval/r2/champion.json" in fe2.require_frozen()
    (sandbox / "payguard" / "robust" / "mech.py").write_text("X = 2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed: payguard/robust/mech.py"):
        fe2.require_frozen()


def test_freeze_detects_added_file_and_ignores_line_endings(sandbox):
    fe2.freeze_defender()
    champ = sandbox / "eval" / "r2" / "champion.json"
    champ.write_bytes(champ.read_bytes().replace(b"\n", b"\r\n"))
    fe2.require_frozen()  # CRLF checkout of the same bytes is not a change
    (sandbox / "payguard" / "robust" / "extra.py").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="not in freeze"):
        fe2.require_frozen()


def test_run_and_gemini_row_refuse_out_of_order(sandbox):
    fe2.RESULTS_PATH.parent.mkdir(parents=True)
    assert fe2.main(["--gemini-row"]) == 2  # no results yet
    fe2.RESULTS_PATH.write_text("{}", encoding="utf-8")
    assert fe2.main(["--run"]) == 2  # one-shot already ran
    fe2.GEMINI_PATH.write_text("{}", encoding="utf-8")
    assert fe2.main(["--gemini-row"]) == 2  # run once


# ---------------------------------------------------------------- verdict and contract ------------------
def _verdict_table(primary_obj: float, primary_block: float) -> dict:
    return {("static_rules", DECISIVE): m(30.0, block=9.0), (fe2.BEST_SINGLE, DECISIVE): m(40.0),
            (fe2.PRIMARY, DECISIVE): m(primary_obj, block=primary_block),
            **{(sw, DECISIVE): m(50.0) for sw in fe2.r1.SWARMS}}


@pytest.mark.parametrize("obj, block, verdict", [(26.9, 5.0, "GO"), (27.1, 5.0, "NO-GO"), (20.0, 9.5, "NO-GO")])
def test_go_no_go_is_the_d022_rule(obj, block, verdict):
    d = fe2.go_no_go(_verdict_table(obj, block))
    assert d["verdict"] == verdict
    assert d["checks"][DECISIVE]["margin_vs_static"] == pytest.approx(30.0 - obj)


def test_dashboard_contract_knows_the_decisive_split():
    assert DECISIVE in typing.get_args(Split)


def test_replay_builds_with_decisive_rows(sandbox, monkeypatch):
    monkeypatch.setattr(fe2.r1, "LINEAGE_PATH", fe2.r1.ROOT / "eval" / "evolution" / "lineage.json")
    table = fe2.merge_rows(("round1", ROUND1), ("round2", ROUND2))
    ep = {"seed": 1, "split": DECISIVE, "defender": fe2.PRIMARY, "events": [
        {"id": 0, "t": 0.0, "worker_id": 1, "requester_id": 2, "amount": 10.0, "verdict": "RELEASE", "paid": True,
         "paid_at": 0.0, "is_attack": False, "family": "honest", "signals": {"velocity": 0.1, "layer": 0.0},
         "weights": {"velocity": 1.0}}]}
    doc = fe2.build_replay(table, "2026-11-08T00:00:00+00:00", ep)
    splits = {r["split"] for r in doc["results"]}
    assert DECISIVE in splits and "TRAIN" in splits
    assert [r["split"] for r in doc["results"]][-1] == DECISIVE  # round-2 rows last
    json.dumps(doc, allow_nan=False)


def test_replay_adds_the_decisive_splits_own_best_single(sandbox, monkeypatch):
    """D-038: the single that is best ON the decisive split is shown there (report-only); other splits are unchanged."""
    monkeypatch.setattr(fe2.r1, "LINEAGE_PATH", fe2.r1.ROOT / "eval" / "evolution" / "lineage.json")
    singles = {("single_graph", DECISIVE): m(28.0), ("single_drift", DECISIVE): m(90.0),
               (fe2.BEST_SINGLE, DECISIVE): m(100.0), ("single_graph", "TRAIN"): m(60.0)}
    table = fe2.merge_rows(("round1", ROUND1), ("round2", {**ROUND2, **singles}))
    assert fe2.split_best_single(table, DECISIVE) == "single_graph"
    assert fe2.split_best_single(ROUND1, DECISIVE) is None
    ep = {"seed": 1, "split": DECISIVE, "defender": fe2.PRIMARY, "events": []}
    shown = {(r["defender"], r["split"]) for r in fe2.build_replay(table, "2026-11-08T00:00:00+00:00", ep)["results"]}
    assert ("single_graph", DECISIVE) in shown and (fe2.BEST_SINGLE, DECISIVE) in shown
    assert ("single_drift", DECISIVE) not in shown and ("single_graph", "TRAIN") not in shown


class _FakeDefender:
    name = "fake"

    def __init__(self, layer_verdict):
        self.layer_verdict, self.learned = layer_verdict, []

    def decide(self, event, log):
        self.last_signals = {"scores": {"velocity": 0.2}, "weights": {"velocity": 1.0}}
        if self.layer_verdict:
            self.last_signals["layer"] = {"verdict": self.layer_verdict, "fired": []}
        return "verdict"

    def learn(self, event, is_attack):
        self.learned.append((event, is_attack))


@pytest.mark.parametrize("layer, expected", [(None, None), ("RELEASE", 0.0), ("STEP_UP", 0.5), ("HOLD", 1.0)])
def test_recorder_maps_the_layer_to_a_number(layer, expected):
    class E:
        id = 7

    rec = fe2._Recorder(_FakeDefender(layer))
    assert rec.decide(E, None) == "verdict"
    sig, w = rec.at[7]
    assert sig.get("layer") == expected and sig["velocity"] == 0.2 and w == {"velocity": 1.0}
    rec.learn(E, True)
    assert rec.defender.learned == [(E, True)]
