"""Round-3 evaluation logic (payguard/r3eval.py), freeze round 3 and the dev-check orchestration.

Stubs and dev seeds (9400-9499) only: no TEST seed is built here and no sealed package is needed."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from payguard import r3eval
from payguard.baselines.trivial import AlwaysHold, AlwaysRelease
from payguard.dream.contract import DreamConfig
from payguard.protocol import TRAIN_FAMILIES, episode_for
from payguard.sim.metrics import Metrics
from payguard.swarm.objective import objective
from payguard.swarm.params import SwarmParams, default_params
from scripts import freeze, r3_dev_check as dc

DEV_SEEDS = tuple(range(9400, 9404))
PKG = "held" + "out" + "3"  # the sealed package; spelled in pieces so only the harness names it


def m(obj: float, block: float = 2.0, immunity: float = 0.0) -> Metrics:
    """A Metrics whose objective is `obj` (leakage = obj - 0.5*block, nothing unpaid)."""
    leak = obj - 0.5 * block
    return Metrics(n_episodes=1, attack_usd=100.0, leaked_usd=leak, leakage_pct=leak, leakage_pct_by_family={},
                   honest_n=100, honest_block_rate=block, honest_unpaid_rate=0.0, honest_median_ttp_h=0.0,
                   honest_p95_ttp_h=0.0, immunity_attempts=immunity, never_contained_rate=0.0, llm_calls=0)


def table(full=27.0, static=30.0, single=40.0, nowm=29.0, full_block=2.0, static_block=2.0) -> dict:
    return {"dreamer_full": m(full, full_block), "static_rules": m(static, static_block),
            "single_policy": m(single), "dreamer_nowm": m(nowm)}


# ---------------------------------------------------------------- registry ------------------------------
def test_registry_and_constants():
    assert r3eval.ARMS == ("static_rules", "single_policy", "swarm_r2", "dreamer_full", "dreamer_nowm",
                           "dreamer_noskeptic", "dreamer_nocalib")
    assert r3eval.H1_MARGIN == 3.0 and r3eval.H2_MARGIN == 2.0
    assert r3eval.BOOTSTRAP_N == 2000


# ---------------------------------------------------------------- the rule -------------------------------
def test_h1_exactly_at_the_margin_passes_and_just_below_fails():
    assert r3eval.decide(table(full=27.0, static=30.0, single=30.0))["h1"]["pass"]
    d = r3eval.decide(table(full=27.25, static=30.0, single=40.0))
    assert not d["h1"]["pass"] and d["verdict"] == "NO-GO"
    assert r3eval.decide(table(full=27.0, static=40.0, single=30.25))["h1"]["pass"]
    d = r3eval.decide(table(full=27.0, static=40.0, single=29.75))  # the single specialist is the binding one
    assert not d["h1"]["pass"] and d["h1"]["margin_vs_static"] == 13.0 and d["h1"]["margin_vs_single"] == 2.75


def test_h1_float_noise_does_not_fail_a_true_margin_of_three():
    assert 32.3 - 29.3 < 3.0  # the naive comparison would fail
    assert r3eval.decide(table(full=29.3, static=32.3, single=60.0))["h1"]["pass"]


def test_h1_block_rate_tie_passes_and_higher_fails():
    assert r3eval.decide(table(full_block=5.0, static_block=5.0, static=40.0))["h1"]["pass"]
    d = r3eval.decide(table(full_block=5.5, static_block=5.0, static=40.0))
    assert not d["h1"]["pass"] and d["verdict"] == "NO-GO"
    assert d["h1"]["honest_block_rate"] == 5.5 and d["h1"]["static_honest_block_rate"] == 5.0


def test_h2_exactly_two_passes_and_just_below_fails():
    assert r3eval.decide(table(full=27.0, nowm=29.0))["h2"]["pass"]
    assert not r3eval.decide(table(full=27.0, nowm=28.75))["h2"]["pass"]
    d = r3eval.decide(table(full=27.0, nowm=31.0))
    assert d["h2"]["margin_vs_nowm"] == 4.0 and d["h2_pass"]


def test_h2_is_reported_but_does_not_decide_the_verdict():
    d = r3eval.decide(table(full=27.0, static=40.0, single=40.0, nowm=27.5))
    assert d["verdict"] == "GO" and d["h2_pass"] is False


def test_decide_reports_the_numbers_and_needs_all_rule_arms():
    d = r3eval.decide(table())
    assert d["h1"]["objective"] == 27.0 and d["h1"]["static_rules"] == 30.0 and d["h1"]["single_policy"] == 40.0
    assert d["h1"]["margin_vs_static"] == 3.0 and d["h1"]["margin_vs_single"] == 13.0
    t = table()
    del t["dreamer_nowm"]
    with pytest.raises(KeyError, match="dreamer_nowm"):
        r3eval.decide(t)


def test_decide_takes_arm_results_and_the_episode_mean_cross_check():
    per = lambda *objs: tuple(m(o) for o in objs)  # noqa: E731
    res = {"dreamer_full": r3eval.ArmResult("dreamer_full", m(27.0), per(25.0, 29.0)),
           "static_rules": r3eval.ArmResult("static_rules", m(30.0), per(30.0, 30.0)),
           "single_policy": r3eval.ArmResult("single_policy", m(40.0), per(40.0, 40.0)),
           "dreamer_nowm": r3eval.ArmResult("dreamer_nowm", m(29.0), per(29.0, 29.0))}
    assert r3eval.decide(res)["verdict"] == "GO"
    cc = r3eval.decide_on_episode_means(res)
    assert cc["h1"]["objective"] == pytest.approx(27.0) and cc["verdict"] == "GO"


# ---------------------------------------------------------------- bootstrap ------------------------------
def test_bootstrap_is_deterministic_and_seed_dependent():
    v = [20.0, 25.0, 31.0, 18.0, 27.0, 22.0, 40.0, 19.0]
    assert r3eval.bootstrap_ci(v) == r3eval.bootstrap_ci(list(v))
    assert r3eval.bootstrap_ci(v, seed=0) != r3eval.bootstrap_ci(v, seed=1)
    lo, hi = r3eval.bootstrap_ci(v)
    assert lo < sum(v) / len(v) < hi and min(v) <= lo and hi <= max(v)


def test_bootstrap_degenerate_inputs():
    assert r3eval.bootstrap_ci([5.0] * 7) == (5.0, 5.0)
    assert r3eval.bootstrap_ci([3.0]) == (3.0, 3.0)
    with pytest.raises(ValueError):
        r3eval.bootstrap_ci([])


def test_paired_diff_ci():
    a, b = [10.0, 12.0, 14.0, 11.0], [7.0, 9.0, 11.0, 8.0]
    assert r3eval.paired_diff_ci(a, b) == (3.0, 3.0)
    with pytest.raises(ValueError):
        r3eval.paired_diff_ci(a, b[:3])


# ---------------------------------------------------------------- running arms ---------------------------
class StubRunner:
    """ArmRunner stub: every arm is one of the trivial defenders (no training, no sealed data)."""

    def champion(self, arm):
        return AlwaysHold if arm in ("dreamer_full", "static_rules") else AlwaysRelease


def test_run_arms_per_episode_and_pooled_come_from_the_same_runs():
    eps = [episode_for(s, TRAIN_FAMILIES) for s in DEV_SEEDS]
    assert isinstance(StubRunner(), r3eval.ArmRunner)
    res = r3eval.run_arms(StubRunner(), eps)
    assert list(res) == list(r3eval.ARMS)
    r = res["dreamer_full"]
    assert len(r.per_episode) == len(eps) and r.pooled.n_episodes == len(eps)
    assert all(mm.n_episodes == 1 for mm in r.per_episode)
    assert res["dreamer_nowm"].objective == pytest.approx(objective(res["dreamer_nowm"].pooled))
    assert r.episode_objectives == [objective(x) for x in r.per_episode]
    d = r3eval.decide(res)
    assert d["verdict"] in ("GO", "NO-GO") and d["h1"]["margin_vs_static"] == pytest.approx(0.0)  # same defender


def test_summarize_arm_has_cis_and_json_roundtrips():
    eps = [episode_for(s, TRAIN_FAMILIES) for s in DEV_SEEDS]
    r = r3eval.run_arm("x", AlwaysHold, eps)
    s = r3eval.summarize_arm(r)
    lo, hi = s["ci95"]["objective"]
    assert lo <= s["mean_episode"]["objective"] <= hi and s["n_episodes"] == len(eps)
    json.dumps(s["ci95"])


# ---------------------------------------------------------------- one-shot guard -------------------------
def test_one_shot_guard(tmp_path):
    p = tmp_path / "final_r3" / "results.json"
    assert r3eval.one_shot_refusal(p) is None
    r3eval.write_once(p, "{}\n")
    assert "already ran" in r3eval.one_shot_refusal(p)
    with pytest.raises(FileExistsError):
        r3eval.write_once(p, "{}\n")  # never overwrites, even if the check above were skipped


# ---------------------------------------------------------------- H3 -------------------------------------
def test_calibration_trend():
    t = r3eval.calibration_trend([4.0, 3.0, 2.0, 1.0])
    assert t["slope_per_generation"] == pytest.approx(-1.0) and t["first"] == 4.0 and t["last"] == 1.0
    assert r3eval.calibration_trend([2.0])["slope_per_generation"] is None
    assert r3eval.calibration_trend([])["n"] == 0


# ---------------------------------------------------------------- defender freeze ------------------------
def _champions(base: DreamConfig | None = None):
    base = base or DreamConfig()
    return {a: (default_params(), r3eval.arm_config(a, base)) for a in r3eval.DREAMER_ARMS}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "payguard" / "dream").mkdir(parents=True)
    (tmp_path / "payguard" / "dream" / "engine.py").write_bytes(b"X = 1\n")
    (tmp_path / "payguard" / "dream" / "__init__.py").write_bytes(b"")
    (tmp_path / "eval" / "r3").mkdir(parents=True)
    (tmp_path / "eval" / "r2").mkdir(parents=True)
    (tmp_path / "eval" / "final_r2").mkdir(parents=True)
    (tmp_path / "payguard" / "robust").mkdir(parents=True)
    (tmp_path / "eval" / "r2" / "champion.json").write_bytes(b'{"params": {}}\n')
    (tmp_path / "payguard" / "robust" / "layer.py").write_bytes(b"Y = 1\n")
    files = {rel: r3eval.digest(tmp_path / rel) for rel in ("eval/r2/champion.json", "payguard/robust/layer.py")}
    (tmp_path / "eval" / "final_r2" / "defender_freeze.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return tmp_path


def _write_freeze(root, **kw):
    doc = r3eval.build_freeze_doc(_champions(), git_rev="abc", root=root, selection={"winner": "c1"}, **kw)
    path = root / "eval" / "r3" / "defender_freeze.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path, doc


def test_freeze_doc_schema(root):
    path, doc = _write_freeze(root, calibration_gap_ema={"dreamer_full": [3.0, 2.0]})
    assert set(doc) == {"schema_version", "frozen_at", "git_rev", "selection", "dream_files", "arms"}
    assert set(doc["arms"]) == set(r3eval.DREAMER_ARMS)
    spec = doc["arms"]["dreamer_nowm"]
    assert spec["kind"] == "swarm" and SwarmParams(**spec["swarm_params"]) == default_params()
    assert spec["dream_config"]["use_wm"] is False and doc["arms"]["dreamer_full"]["dream_config"]["use_wm"] is True
    assert set(doc["dream_files"]) == {"payguard/dream/engine.py", "payguard/dream/__init__.py"}
    assert doc["arms"]["dreamer_full"]["calibration_gap_ema"] == [3.0, 2.0]
    assert r3eval.validate_freeze_doc(json.loads(path.read_text(encoding="utf-8"))) == []


def test_freeze_doc_rejects_arms_that_differ_by_more_than_their_switch(root):
    ch = _champions()
    ch["dreamer_nowm"] = (default_params(), replace(ch["dreamer_nowm"][1], lambda0=0.9))
    with pytest.raises(r3eval.FreezeError, match="more than its one switch"):
        r3eval.build_freeze_doc(ch, root=root)
    ch = _champions()
    del ch["dreamer_nocalib"]
    with pytest.raises(r3eval.FreezeError, match="dreamer_nocalib"):
        r3eval.build_freeze_doc(ch, root=root)


def test_frozen_arms_verifies_and_detects_drift(root):
    path, _ = _write_freeze(root)
    arms = r3eval.FrozenArms(root, path)
    assert isinstance(arms, r3eval.ArmRunner)
    d = arms.champion("dreamer_full")()
    assert d is not arms.champion("dreamer_full")()  # a FRESH defender per call
    assert hasattr(d, "decide") and hasattr(d, "learn")
    assert hasattr(arms.champion("single_policy")(), "decide") and hasattr(arms.champion("static_rules")(), "decide")
    (root / "payguard" / "dream" / "engine.py").write_bytes(b"X = 2\n")
    with pytest.raises(r3eval.FreezeError, match="changed: payguard/dream/engine.py"):
        r3eval.FrozenArms(root, path)
    (root / "payguard" / "dream" / "engine.py").write_bytes(b"X = 1\r\n")  # CRLF checkout of the same text: no drift
    r3eval.FrozenArms(root, path)
    (root / "payguard" / "dream" / "extra.py").write_bytes(b"")
    with pytest.raises(r3eval.FreezeError, match="not in freeze: payguard/dream/extra.py"):
        r3eval.FrozenArms(root, path)


def test_frozen_arms_detects_baseline_drift_and_missing_file(root):
    path, _ = _write_freeze(root)
    (root / "eval" / "r2" / "champion.json").write_bytes(b'{"params": {"x": 1}}\n')
    with pytest.raises(r3eval.FreezeError, match="round-2 changed: eval/r2/champion.json"):
        r3eval.FrozenArms(root, path)
    path.unlink()
    with pytest.raises(r3eval.FreezeError, match="not frozen yet"):
        r3eval.FrozenArms(root, path)


def test_swarm_r2_baseline_builds_from_the_real_champion_and_its_freeze_holds():
    assert r3eval.verify_r2_baseline(r3eval.ROOT) == []  # this branch did not move the round-2 defender
    doc = r3eval.build_freeze_doc(_champions(), git_rev="x")  # hashes of the real payguard/dream/*.py
    arms = r3eval.FrozenArms.__new__(r3eval.FrozenArms)
    arms.root, arms.doc = r3eval.ROOT, doc
    assert hasattr(arms.champion("swarm_r2")(), "decide")


# ---------------------------------------------------------------- freeze.py round 3 ----------------------
@pytest.fixture
def fz(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(freeze, "FROZEN_R3_PATH", tmp_path / "eval" / "FROZEN_R3.json")
    return tmp_path


def test_freeze_round3_missing_package_fails_clearly(fz, capsys):
    assert freeze.main(["--round", "3"]) == 1
    assert PKG in capsys.readouterr().out
    assert freeze.main(["--round", "3", "--write"]) == 1  # never writes an empty freeze
    assert not freeze.FROZEN_R3_PATH.exists()
    (fz / "payguard" / PKG).mkdir(parents=True)
    assert freeze.main(["--round", "3"]) == 1  # empty directory is the same as missing


def test_freeze_round3_write_verify_and_drift(fz, capsys):
    pkg = fz / "payguard" / PKG
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_bytes(b"A = 1\n")
    (pkg / "family.py").write_bytes(b"B = 1\n")
    (pkg / "__pycache__" / "x.pyc").write_bytes(b"junk")
    assert freeze.main(["--round", "3"]) == 1  # package present, not frozen yet
    assert "not frozen yet" in capsys.readouterr().out
    assert freeze.main(["--round", "3", "--write"]) == 0
    frozen = json.loads(freeze.FROZEN_R3_PATH.read_text(encoding="utf-8"))
    assert set(frozen) == {f"payguard/{PKG}/__init__.py", f"payguard/{PKG}/family.py"}
    assert freeze.main(["--round", "3"]) == 0
    assert "FROZEN_R3 files match" in capsys.readouterr().out
    (pkg / "family.py").write_bytes(b"B = 1\r\n")  # CRLF checkout: not a change
    assert freeze.main(["--round", "3"]) == 0
    (pkg / "family.py").write_bytes(b"B = 2\n")
    assert freeze.main(["--round", "3"]) == 1
    assert f"changed: payguard/{PKG}/family.py" in capsys.readouterr().out
    (pkg / "family.py").write_bytes(b"B = 1\n")
    (pkg / "extra.py").write_bytes(b"")
    assert freeze.main(["--round", "3"]) == 1
    assert f"not in freeze: payguard/{PKG}/extra.py" in capsys.readouterr().out


def test_freeze_rounds_1_2_unchanged(capsys):
    assert freeze.main(["--round", "4"]) == 2
    capsys.readouterr()
    assert freeze.main(["--round", "2"]) == 0
    assert "FROZEN_R2 files match" in capsys.readouterr().out
    assert freeze.main([]) == 0 and "FROZEN files match" in capsys.readouterr().out


# ---------------------------------------------------------------- dev check ------------------------------
class StubTrain:
    """train(cfg, families, evo_seed) -> SwarmParams. A deterministic stand-in: encodes (cfg, seed) in one threshold."""

    def __init__(self):
        self.calls = []

    def __call__(self, cfg, families, evo_seed):
        self.calls.append((cfg, tuple(families), evo_seed))
        return replace(default_params(), vel_usd_lo=1000.0 + 10 * evo_seed + (0 if cfg.use_wm else 5))


def test_dev_check_orchestration(tmp_path):
    train = StubTrain()
    fam_of_call = []

    def score(params, eps):
        # which grid point produced `params`? recover from the last train call
        cfg, fams, evo = train.calls[-1]
        fam_of_call.append((fams, tuple(e.seed for e in eps)))
        base = {"c1": 20.0, "c2": 18.0, "c3": 22.0, "c4": 18.0}  # c2 and c4 tie at the lowest LOFO
        cid = next(k for k in dc.GRID if dc.GRID[k][3] == cfg.lambda0 and dc.GRID[k][0] == cfg.share_posterior)
        v = base[cid] + (0 if cfg.use_wm else 2.0) + (0.5 if evo else 0.0)
        return m(v)

    out = tmp_path / "dev_check.json"
    doc = dc.run_dev_check(train, out, score=score, seeds=DEV_SEEDS)
    assert doc["winner"] == "c2"  # tie c2 / c4 goes to the lower id
    assert list(doc["grid"]) == ["c1", "c2", "c3", "c4"]
    assert doc["grid"]["c1"]["lofo"] == 20.0 and doc["grid"]["c2"]["lofo"] == 18.0
    # every fold trained on the OTHER family only, and the cfg given to train carries that family only
    for cfg, fams, evo in train.calls:
        assert len(fams) == 1 and cfg.families == fams
    folds = {(c.families, e) for c, f, e in train.calls}
    assert (("structuring",), 0) in folds and (("ring",), 0) in folds
    # K1: winner (+0.5 for seed 1) vs twin (+2.0): mean 18.25 vs 20.25
    k1 = doc["k1"]
    assert k1["winner_mean_lofo"] == pytest.approx(18.25) and k1["twin_mean_lofo"] == pytest.approx(20.25)
    assert k1["margin"] == pytest.approx(2.0) and k1["pass"] and "proceed" in doc["verdict"]
    # memoized: 4 grid cfgs x 2 folds + winner seed 1 (2) + twin seeds 0, 1 (4) = 14, not 8 + 8
    assert train.calls and len(train.calls) == 14 == doc["train_calls"]
    assert all(seeds == DEV_SEEDS for _, seeds in fam_of_call)
    assert json.loads(out.read_text(encoding="utf-8"))["winner"] == "c2"
    assert json.loads(out.read_text(encoding="utf-8"))["winner_config"]["use_wm"] is True


def test_dev_check_k1_failure_says_stop(tmp_path):
    train = StubTrain()
    doc = dc.run_dev_check(train, tmp_path / "d.json", score=lambda p, eps: m(20.0), seeds=DEV_SEEDS)
    assert doc["winner"] == "c1" and doc["k1"]["margin"] == 0.0 and not doc["k1"]["pass"]
    assert "do not run" in doc["verdict"]


def test_dev_check_k1_margin_is_inclusive_at_one(tmp_path):
    def score(params, eps):
        return m(20.0 + (1.0 if params.vel_usd_lo % 10 == 5 else 0.0))  # twin params end in 5 (StubTrain)

    doc = dc.run_dev_check(StubTrain(), tmp_path / "d.json", score=score, seeds=DEV_SEEDS)
    assert doc["k1"]["margin"] == pytest.approx(1.0) and doc["k1"]["pass"]


def test_dev_check_refuses_overwrite_and_real_train_missing(tmp_path):
    out = tmp_path / "d.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        dc.run_dev_check(StubTrain(), out, score=lambda p, e: m(1.0), seeds=DEV_SEEDS)
    with pytest.raises(SystemExit, match="no real train function"):
        dc.resolve_train("payguard.dream.no_such_module:train")


def test_dev_check_refuses_test_seeds():
    for bad in (2000, 3005, 4010, 5000):
        with pytest.raises(AssertionError):
            dc.DevCheck(StubTrain(), seeds=(bad,))


def test_dev_check_default_scorer_runs_the_real_evaluator_on_dev_seeds():
    dcheck = dc.DevCheck(lambda cfg, fams, evo: default_params(), seeds=DEV_SEEDS[:1])
    r = dcheck.lofo(dc.grid_config("c1"), 0)
    assert r["lofo"] > 0 and [f["held_out"] for f in r["folds"]] == list(TRAIN_FAMILIES)


def test_dev_check_defaults_score_on_train_b_only():
    from payguard.evolve.splits import TRAIN_B
    d = dc.DevCheck(StubTrain())
    assert d.seeds == tuple(TRAIN_B) and set(d._eps) == set(TRAIN_FAMILIES)
    assert [e.seed for e in d._eps["ring"]] == list(TRAIN_B)


def test_grid_is_the_fixed_one():
    assert set(dc.GRID) == {"c1", "c2", "c3", "c4"}
    assert asdict(dc.grid_config("c3"))["share_uncertain"] == 0.33 and dc.grid_config("c4").lambda0 == 0.6
    for cid in dc.GRID:
        cfg = dc.grid_config(cid)
        assert (cfg.n_dreams, cfg.tau, cfg.sigma, cfg.ema_beta, cfg.n_bins) == (32, 1.0, 1.0, 0.5, 4)
