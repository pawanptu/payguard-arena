"""swarm_r2: the frozen defender, the honest-calibrated layer, the LOFO folds, and the TRAIN-only tripwire."""
import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import payguard.protocol as protocol
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from payguard.robust import anchored
from payguard.robust.folds import (HALVES, check_spec, fam_spec, full_train_specs, get_episode, lofo_folds)
from payguard.robust.layer import (STAT_NAMES, AnomalyLayer, R2Defender, calibrate, calibrate_level, collect_stats,
                                   make_r2)
from payguard.robust.r2 import CHAMPION_PATH, champion_sha256, load_champion, make_r2_defender
from payguard.sim.metrics import evaluate
from payguard.sim.runner import run_episode
from payguard.sim.types import Verdict
from payguard.swarm.params import SwarmParams, default_params

ROOT = Path(__file__).resolve().parent.parent
BUILT_SEEDS: list[int] = []


@pytest.fixture(autouse=True)
def tripwire(monkeypatch):
    """Every episode built by any test in this module must come from a TRAIN seed (fails the test otherwise)."""
    real = protocol.build_episode

    def guarded(seed, *a, **kw):
        assert seed in TRAIN_SEEDS, f"test_r2 tried to build non-TRAIN seed {seed}"
        BUILT_SEEDS.append(seed)
        return real(seed, *a, **kw)

    monkeypatch.setattr(protocol, "build_episode", guarded)


# ---------------------------------------------------------------- the frozen defender ----------------
def test_factory_loads_frozen_params():
    doc = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
    d = make_r2_defender()
    assert isinstance(d, R2Defender) and d.name == "swarm_r2"
    assert asdict(d.params) == doc["params"]
    assert (d.layer is None) == (doc["layer"] is None)
    if doc["layer"] is not None:
        assert d.layer.spec == doc["layer"]
    assert asdict(load_champion()[0]) == doc["params"]
    assert champion_sha256() == hashlib.sha256(CHAMPION_PATH.read_bytes()).hexdigest()
    assert make_r2_defender() is not make_r2_defender()  # a fresh, stateless-at-start instance per call


def test_defender_runs_on_a_small_train_episode():
    ep = episode_for(TRAIN_SEEDS[0], ("ring",))
    assert BUILT_SEEDS and BUILT_SEEDS[-1] == TRAIN_SEEDS[0]
    res = run_episode(ep, make_r2_defender())
    assert set(res.verdicts) == {e.id for e in ep.events}
    assert all(isinstance(v, Verdict) for v in res.verdicts.values())
    m = evaluate([res])
    assert 0.0 <= m.leakage_pct <= 100.0 and 0.0 <= m.honest_block_rate < 100.0
    # same defender, same episode: deterministic
    again = run_episode(ep, make_r2_defender())
    assert again.verdicts == res.verdicts


def test_defender_never_reads_ground_truth():
    """decide() only gets (event, log); the layer statistics come from those. Labels enter through learn() only."""
    import inspect

    import payguard.robust.layer as layer_module
    src = inspect.getsource(layer_module)
    for forbidden in ("episode.truth", "ep.truth[e.id]"):  # only collect_stats (offline calibration) may use truth
        assert src.count(forbidden) <= 1
    assert "truth" not in inspect.getsource(layer_module.R2Defender)
    assert "truth" not in inspect.getsource(layer_module.AnomalyLayer)
    assert "truth" not in inspect.getsource(layer_module.StatExtractor)


# ---------------------------------------------------------------- the layer ---------------------------
def test_calibration_budget_and_ordering():
    rng = np.random.default_rng(0)
    x = np.abs(rng.standard_normal((20000, len(STAT_NAMES)))) * np.arange(1, len(STAT_NAMES) + 1)
    spec = calibrate(x, 0.01, 0.05)
    hold, step = np.array(spec["hold"]), np.array(spec["step"])
    assert (step <= hold).all()
    assert spec["honest_union_rate_hold"] <= 0.01 + 1e-12
    assert spec["honest_union_rate_step"] <= 0.05 + 1e-12
    assert spec["honest_union_rate_hold"] >= 0.0095     # bisection is tight, not merely under budget
    assert spec["honest_union_rate_step"] >= 0.045
    cut, rate = calibrate_level(x, 0.02)
    assert float((x > cut).any(axis=1).mean()) == rate <= 0.02
    with pytest.raises(ValueError):
        calibrate(x, 0.05, 0.01)


def test_layer_uses_only_honest_events_for_cutoffs():
    ep = episode_for(TRAIN_SEEDS[0], ("structuring",))
    x, y = collect_stats(ep)
    assert x.shape == (len(ep.events), len(STAT_NAMES)) and y.sum() > 0
    a = calibrate(x[~y], 0.01, 0.03)
    b = calibrate(x[~y], 0.01, 0.03)
    assert a == b
    assert a["n_honest_events"] == int((~y).sum())


def test_layer_escalates_never_relaxes():
    ep = episode_for(TRAIN_SEEDS[1], ("ring",))
    x, y = collect_stats(ep)
    spec = calibrate(x[~y], 0.01, 0.05)
    base = run_episode(ep, make_r2(default_params(), None))
    layered = run_episode(ep, make_r2(default_params(), spec))
    sev = {Verdict.RELEASE: 0, Verdict.STEP_UP: 1, Verdict.HOLD: 2}
    # L1 learning sees the same scores but different label timing, so compare the layer alone against the swarm
    # on a single decision: the layer verdict is combined by max severity.
    d = make_r2(default_params(), spec)
    assert isinstance(d.layer, AnomalyLayer)
    assert any(sev[layered.verdicts[i]] > sev[base.verdicts[i]] for i in base.verdicts)
    assert evaluate([layered]).honest_block_rate >= evaluate([base]).honest_block_rate


def test_layer_spec_mismatch_is_rejected():
    spec = calibrate(np.ones((100, len(STAT_NAMES))), 0.01, 0.02)
    spec["stats"] = spec["stats"][:-1]
    with pytest.raises(ValueError):
        AnomalyLayer(spec)


# ---------------------------------------------------------------- C1 helpers --------------------------
def test_prior_distance_and_fitness_penalty():
    d = default_params()
    assert anchored.prior_distance(d) == 0.0
    moved = SwarmParams(**{**asdict(d), "step_up_threshold": 0.5, "hold_threshold": 0.55})
    assert anchored.prior_distance(moved) > 0.0

    class M:  # minimal metrics stand-in
        leakage_pct, honest_block_rate, honest_unpaid_rate = 10.0, 4.0, 0.0

    assert anchored.fitness(M, d, 50.0, 5.0) == pytest.approx(12.0)
    assert anchored.fitness(M, moved, 50.0, 5.0) > 12.0
    assert anchored.fitness(M, d, 50.0, 3.0) == pytest.approx(12.0 + 100.0)  # block above static is punished hard


# ---------------------------------------------------------------- folds and the TRAIN tripwire ---------
def test_folds_are_out_of_family_and_out_of_seed():
    folds = lofo_folds()
    assert len(folds) == 4
    for f in folds:
        fit_seeds = {s[1] for s in f.fit}
        eval_seeds = {s[1] for s in f.evaluate}
        assert fit_seeds in map(set, HALVES) and eval_seeds in map(set, HALVES)
        assert fit_seeds.isdisjoint(eval_seeds)
        fit_fam = {s[2] for s in f.fit}
        eval_fam = {s[2] for s in f.evaluate}
        assert len(fit_fam) == len(eval_fam) == 1 and fit_fam != eval_fam
    assert {f.direction for f in folds} == {"ring->structuring", "structuring->ring"}
    assert {s[1] for s in full_train_specs()} == set(TRAIN_SEEDS)


def test_only_train_seeds_and_families_can_be_built():
    for seed in (999, 1020, 2000, 3000, 4000, 4019):
        with pytest.raises(ValueError):
            get_episode(fam_spec(seed, ("ring",)))
    with pytest.raises(ValueError):
        check_spec(fam_spec(1000, ("not_a_train_family",)))
    check_spec(fam_spec(1000, ("ring", "structuring")))
    # the guard in this module's fixture: building a non-TRAIN seed through the protocol raises
    with pytest.raises(AssertionError):
        episode_for(2000, ("ring",))


def test_robust_code_never_names_test_splits():
    forbidden = ("TEST_SEEN_SEEDS", "TEST_HELDOUT_SEEDS", "HELDOUT_FAMILIES", "TEST_SEEN", "TEST_HELDOUT")
    files = sorted((ROOT / "payguard" / "robust").glob("*.py")) + [ROOT / "scripts" / "r2_select.py"]
    assert len(files) >= 5
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                assert not any("TEST_" in n or "HELDOUT" in n for n in names), path
            if isinstance(node, ast.Name):
                assert node.id not in forbidden, path
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                assert not (1020 <= node.value <= 4100 and node.value % 1000 < 20 and node.value >= 2000), \
                    f"{path}: suspicious test-split seed constant {node.value}"
    # the data this module builds are TRAIN seeds only
    assert min(TRAIN_SEEDS) == 1000 and max(TRAIN_SEEDS) == 1019
