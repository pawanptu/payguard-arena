"""Round-3 long-run safety (D-044): dev-check checkpoint/resume, freeze refusal without a passed K1, budget guard off,
run logs. Stubs and dev seeds 9400-9403 only; nothing is evolved."""
import json
from dataclasses import replace

import pytest

from payguard.dream import training
from payguard.sim.metrics import Metrics
from payguard.swarm.params import default_params
from scripts import r3_dev_check as dc
from scripts import r3_freeze_arms as fz

DEV_SEEDS = tuple(range(9400, 9404))


def m(obj: float) -> Metrics:
    return Metrics(n_episodes=1, attack_usd=100.0, leaked_usd=obj, leakage_pct=obj, leakage_pct_by_family={},
                   honest_n=100, honest_block_rate=0.0, honest_unpaid_rate=0.0, honest_median_ttp_h=0.0,
                   honest_p95_ttp_h=0.0, immunity_attempts=0.0, never_contained_rate=0.0, llm_calls=0)


def stub_train(cfg, families, evo_seed):
    return replace(default_params(), vel_usd_lo=1000.0 + 10 * evo_seed + (0 if cfg.use_wm else 5) + 100 * cfg.lambda0)


def score(params, eps):
    return m(params.vel_usd_lo / 100.0)


def test_budget_guard_off_and_d028_settings():
    c = training.EVOLVE_CONFIG
    assert (c.pop, c.gens) == (28, 22) and c.budget_min >= 1e6


def test_dev_check_resumes_from_checkpoint_identically(tmp_path):
    out = tmp_path / "dev_check.json"
    calls = []

    def counting(cfg, fams, evo):
        calls.append(1)
        if len(calls) == 6:                       # crash part-way through (after 5 finished folds)
            raise RuntimeError("simulated crash")
        return stub_train(cfg, fams, evo)

    with pytest.raises(RuntimeError):
        dc.run_dev_check(counting, out, score=score, seeds=DEV_SEEDS)
    ckpt = out.with_suffix(".folds.jsonl")
    assert not out.exists() and len(ckpt.read_text().splitlines()) == 5

    resumed_calls = []
    doc = dc.run_dev_check(lambda c, f, e: resumed_calls.append(1) or stub_train(c, f, e), out, score=score,
                           seeds=DEV_SEEDS)
    assert doc["folds_resumed_from_checkpoint"] == 5 and doc["train_calls"] == len(resumed_calls) == 14 - 5

    ref = dc.run_dev_check(stub_train, tmp_path / "ref" / "dev_check.json", score=score, seeds=DEV_SEEDS)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("seconds", "train_calls", "folds_resumed_from_checkpoint")}
    assert strip(doc) == strip(ref)               # resumed == uninterrupted


def test_freeze_refuses_without_passed_k1(tmp_path, monkeypatch):
    p = tmp_path / "dev_check.json"
    with pytest.raises(SystemExit, match="missing"):
        fz.load_k1(p)
    p.write_text(json.dumps({"k1": {"pass": False}}))
    with pytest.raises(SystemExit, match="K1 did not pass"):
        fz.load_k1(p)
    p.write_text(json.dumps({"k1": {"pass": True, "margin": 1.5}, "winner": "c1"}))
    assert fz.load_k1(p)["winner"] == "c1"


def test_save_run_is_json_and_stable_name(tmp_path):
    run = {"families": ["ring"], "evo_seed": 1, "dream_config": {"lambda0": 0.3}, "champion_params": default_params(),
           "lineage": [{"x": float("nan")}], "champion": {"params": {"a": 1.23456}}}
    p1 = training.save_run(run, tmp_path)
    p2 = training.save_run(run, tmp_path)
    assert p1 == p2 and p1.name.startswith("run_ring_s1_")
    doc = json.loads(p1.read_text())
    assert "champion_params" not in doc and doc["lineage"][0]["x"] is None
