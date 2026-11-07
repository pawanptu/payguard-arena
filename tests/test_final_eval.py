"""scripts/final_eval.py without ever touching TEST data: every test that could reach the episode builder
replaces it with a tripwire, and the go/no-go logic is checked on hand-made metrics."""
from __future__ import annotations

import json

import pytest

from payguard.sim.metrics import Metrics
from payguard.swarm.single import SPECIALIST_NAMES
from scripts import final_eval as fe


def metrics(leak: float, block: float = 5.0, unpaid: float = 0.0) -> Metrics:
    return Metrics(n_episodes=20, attack_usd=1000.0, leaked_usd=10 * leak, leakage_pct=leak,
                   leakage_pct_by_family={}, honest_n=100, honest_block_rate=block, honest_unpaid_rate=unpaid,
                   honest_median_ttp_h=0.0, honest_p95_ttp_h=0.0, immunity_attempts=0.0,
                   never_contained_rate=0.0, llm_calls=0)


def table(primary_leak: float, primary_block: float = 5.0, *, static_leak: float = 25.0, single_leak: float = 40.0,
          static_block: float = 9.6) -> dict:
    """Same numbers on both TEST splits; objective = leak + 0.5 * block when unpaid = 0."""
    out = {}
    for split in ("TEST_SEEN", "TEST_HELDOUT"):
        out[("static_rules", split)] = metrics(static_leak, static_block)
        out[("single_policy", split)] = metrics(single_leak)
        for sw in fe.SWARMS:
            out[(sw, split)] = metrics(primary_leak, primary_block)
    return out


@pytest.fixture
def tripwire(monkeypatch):
    def boom(split):
        raise AssertionError(f"a test tried to build {split} episodes")
    monkeypatch.setattr(fe, "split_episodes", boom)


def test_pre_registered_constants_are_unchanged():
    assert fe.PRIMARY == "swarm_evolved_memory" and fe.DECISIVE_SPLIT == "TEST_HELDOUT" and fe.MARGIN == 3.0
    assert fe.BUDGET["sarvam"] == 600
    assert list(fe.SPLITS) == ["TRAIN", "TEST_SEEN", "TEST_HELDOUT"]
    assert fe.SPLITS["TEST_HELDOUT"][1] == ("sleeper",)


def test_go_when_three_points_better_than_both_and_no_more_blocking():
    # static obj = 25 + 4.8 = 29.8; primary obj = 24 + 2.5 = 26.5 -> margin 3.3; best single 42.5
    d = fe.go_no_go(table(24.0), "single_policy")
    c = d["checks"]["TEST_HELDOUT"]
    assert d["verdict"] == "GO" and c["pass"]
    assert c["margin_vs_static"] == pytest.approx(3.3) and c["margin_vs_best_single"] == pytest.approx(16.0)


def test_no_go_when_margin_is_short_by_a_hair():
    d = fe.go_no_go(table(24.31), "single_policy")  # margin 2.99
    assert d["verdict"] == "NO-GO" and not d["checks"]["TEST_HELDOUT"]["pass"]


def test_no_go_when_the_best_single_is_close_even_if_static_is_beaten():
    d = fe.go_no_go(table(10.0, single_leak=12.0), "single_policy")  # vs single: 14.5 - 12.5 = 2.0
    assert d["verdict"] == "NO-GO"


def test_no_go_when_blocking_more_honest_workers_than_static():
    d = fe.go_no_go(table(5.0, primary_block=9.7), "single_policy")
    assert d["verdict"] == "NO-GO" and d["checks"]["TEST_HELDOUT"]["honest_block_rate"] > 9.6


def test_only_heldout_decides_seen_is_reported():
    t = table(24.0)
    t[(fe.PRIMARY, "TEST_SEEN")] = metrics(40.0)  # fails on TEST_SEEN
    d = fe.go_no_go(t, "single_policy")
    assert d["verdict"] == "GO" and not d["checks"]["TEST_SEEN"]["pass"]
    assert fe.PRIMARY not in d["secondary_pass_not_decisive"]
    assert set(d["secondary_pass_not_decisive"]) == set(fe.SWARMS) - {fe.PRIMARY}


def test_run_refuses_when_results_exist_and_builds_nothing(tmp_path, monkeypatch, tripwire, capsys):
    done = tmp_path / "results.json"
    done.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fe, "RESULTS_PATH", done)
    monkeypatch.setattr(fe, "ROOT", tmp_path)
    assert fe.main(["--run"]) == 2
    assert "already ran" in capsys.readouterr().out


def test_warm_with_budget_used_up_calls_nothing(tmp_path, monkeypatch, tripwire):
    spend = tmp_path / "llm_spend.json"
    spend.write_text(json.dumps({"sarvam": {"calls": 600, "budget": 600, "runs": []}}), encoding="utf-8")
    monkeypatch.setattr(fe, "SPEND_PATH", spend)
    monkeypatch.setattr(fe, "make_assessor", lambda *a, **k: pytest.fail("an assessor was built"))
    assert fe.main(["--warm", "sarvam", "--cap", "10"]) == 1


def test_warm_without_a_key_calls_nothing(tmp_path, monkeypatch, tripwire):
    class Offline:
        offline, client = True, None
    monkeypatch.setattr(fe, "SPEND_PATH", tmp_path / "llm_spend.json")
    monkeypatch.setattr(fe, "make_assessor", lambda *a, **k: Offline())
    assert fe.main(["--warm", "gemini", "--cap", "5"]) == 1
    assert not (tmp_path / "llm_spend.json").exists()


def test_warm_needs_a_cap():
    with pytest.raises(SystemExit):
        fe.main(["--warm", "sarvam"])


def test_spend_ledger_accumulates_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "FINAL_DIR", tmp_path)
    monkeypatch.setattr(fe, "SPEND_PATH", tmp_path / "llm_spend.json")
    stale = fe.load_spend()  # a concurrent run's view from before anything was written
    fe.record_spend("sarvam", 250, 3)
    fe.record_spend("gemini", 40, 0)
    fe.record_spend("sarvam", 100, 0)
    s = fe.load_spend()
    assert stale == {} and fe.spend_calls(s, "sarvam") == 350 and fe.spend_calls(s, "gemini") == 40
    assert [r["calls"] for r in s["sarvam"]["runs"]] == [250, 100] and s["sarvam"]["budget"] == 600


def test_dashboard_episode_is_the_pre_chosen_heldout_one():
    assert fe.EPISODE == ("TEST_HELDOUT", 3000)  # D-031, chosen before any TEST number existed


def test_final_replay_is_a_valid_contract_with_memory_signals():
    """TRAIN only (the real run uses fe.EPISODE): rows for one split, the primary defender's episode,
    the dashboard's Pydantic contract."""
    champion, meta = fe.load_champion()
    assert meta["id"] and len(meta["sha256"]) == 64
    names = fe.display_names("single_policy")
    m = {(n, "TRAIN"): metrics(10.0 + i) for i, n in enumerate(names)}
    doc = fe.build_final_replay(m, "single_policy", champion, "2026-11-08T00:00:00+00:00", ("TRAIN", 1000))
    assert [r["defender"] for r in doc["results"]] == names and {r["split"] for r in doc["results"]} == {"TRAIN"}
    ep = doc["episode"]
    assert ep["defender"] == fe.PRIMARY and ep["split"] == "TRAIN" and ep["seed"] == 1000
    assert all(set(e["signals"]) == {*SPECIALIST_NAMES, "memory"} for e in ep["events"])
    assert any(e["signals"]["memory"] > 0 for e in ep["events"])
    assert doc["payouts"] == [] and doc["evolution"]
