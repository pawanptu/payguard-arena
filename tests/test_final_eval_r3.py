"""Round-3 one-shot harness (scripts/final_eval_r3.py). Nothing here builds or imports anything sealed: the loaders
that would are replaced by stubs on dev seeds (9400-9499), and a tripwire refuses every non-TRAIN, non-dev seed.

The sealed package's name is spelled in pieces on purpose: only the harness' own lazy-import path may name it."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

import payguard.protocol as protocol
import payguard.sim.engine as engine
from payguard import r3eval
from payguard.baselines.trivial import AlwaysHold, AlwaysRelease
from payguard.dream.contract import assert_seed_allowed
from payguard.protocol import TRAIN_FAMILIES, episode_for
from scripts import final_eval_r3 as fe3

SEALED_PKG = "held" + "out" + "3"
ROOT = Path(__file__).resolve().parent.parent
DEV_SEEDS = tuple(range(9400, 9404))


@pytest.fixture(autouse=True)
def tripwire(monkeypatch, tmp_path):
    real = engine.build_episode

    def guarded(seed, *a, **kw):
        assert_seed_allowed(seed)  # raises on anything but a TRAIN / dev seed
        return real(seed, *a, **kw)

    monkeypatch.setattr(engine, "build_episode", guarded)
    monkeypatch.setattr(protocol, "build_episode", guarded)
    monkeypatch.setattr(fe3, "OUT_DIR", tmp_path / "final_r3")
    monkeypatch.setattr(fe3, "RESULTS_PATH", tmp_path / "final_r3" / "results.json")
    monkeypatch.setattr(fe3, "FREEZE_PATH", tmp_path / "defender_freeze.json")  # absent unless a test writes it
    # Since P1's merge, collecting the sealed family's own test module imports the package before any test here runs. Forget it
    # for the duration of each test (restored at teardown) so "not in sys.modules" checks only what the harness imports.
    sealed = f"payguard.{SEALED_PKG}"
    for name in [m for m in sys.modules if m == sealed or m.startswith(sealed + ".")]:
        monkeypatch.delitem(sys.modules, name)


def boom(*a, **kw):
    raise AssertionError("a sealed loader was reached")


def dev_eps(seeds=DEV_SEEDS):
    return [episode_for(s, TRAIN_FAMILIES) for s in seeds]


class StubArms:
    """Everything holds (like static_rules) except dreamer_nowm and single_policy, which release everything."""

    def champion(self, arm):
        return AlwaysRelease if arm in ("dreamer_nowm", "single_policy") else AlwaysHold

    def calibration_gaps(self, arm):
        return [4.0, 3.0, 2.0] if arm == "dreamer_full" else []

    def file_sha256(self):
        return "stub"


def stub_everything(monkeypatch, arms=None):
    monkeypatch.setattr(fe3, "freeze_problems", lambda: [])
    monkeypatch.setattr(fe3, "load_arms", lambda: arms or StubArms())
    monkeypatch.setattr(fe3, "load_sealed", lambda: {"name": fe3.SEALED_SPLIT, "seeds": [DEV_SEEDS[0], DEV_SEEDS[-1]],
                                                      "families": list(TRAIN_FAMILIES), "episodes": dev_eps()})
    monkeypatch.setattr(fe3, "load_spent", lambda: {s: dev_eps(DEV_SEEDS[:2]) for s in fe3.SPENT_SPLITS})


# ---------------------------------------------------------------- plan -----------------------------------
def test_plan_builds_and_imports_nothing_sealed(monkeypatch, capsys):
    monkeypatch.setattr(fe3, "load_sealed", boom)
    monkeypatch.setattr(fe3, "load_spent", boom)
    assert fe3.main(["--plan"]) == 0
    out = capsys.readouterr().out
    for arm in r3eval.ARMS:
        assert arm in out
    assert "NOT READY" in out  # no defender freeze in the sandbox
    assert f"payguard.{SEALED_PKG}" not in sys.modules


# ---------------------------------------------------------------- guards ---------------------------------
def test_run_refuses_when_results_exist_before_anything_else(monkeypatch, capsys):
    monkeypatch.setattr(fe3, "freeze_problems", boom)
    monkeypatch.setattr(fe3, "load_sealed", boom)
    fe3.RESULTS_PATH.parent.mkdir(parents=True)
    fe3.RESULTS_PATH.write_text("{}", encoding="utf-8")
    assert fe3.main(["--run"]) == 2
    assert "already ran" in capsys.readouterr().out
    assert fe3.RESULTS_PATH.read_text(encoding="utf-8") == "{}"


def test_run_refuses_without_a_defender_freeze(monkeypatch, capsys):
    from scripts import freeze
    monkeypatch.setattr(freeze, "main", lambda argv: 0)  # the sealed-package freeze would be fine
    monkeypatch.setattr(fe3, "load_sealed", boom)
    monkeypatch.setattr(fe3, "load_spent", boom)
    assert fe3.main(["--run"]) == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "not frozen yet" in out
    assert not fe3.RESULTS_PATH.exists()


def test_run_refuses_when_freeze_round_3_fails(monkeypatch, capsys):
    from scripts import freeze
    monkeypatch.setattr(r3eval.FrozenArms, "problems", lambda self: [])
    monkeypatch.setattr(r3eval, "load_freeze", lambda path: {})
    monkeypatch.setattr(freeze, "main", lambda argv: 1 if argv == ["--round", "3"] else 0)
    monkeypatch.setattr(fe3, "load_sealed", boom)
    assert fe3.freeze_problems() == ["python -m scripts.freeze --round 3 fails (see its output above)"]
    assert fe3.main(["--run"]) == 2
    assert "freeze --round 3 fails" in capsys.readouterr().out
    assert not fe3.RESULTS_PATH.exists()


def test_run_refuses_when_the_defender_freeze_drifted(monkeypatch, capsys):
    from scripts import freeze
    monkeypatch.setattr(freeze, "main", lambda argv: 0)
    monkeypatch.setattr(r3eval, "load_freeze", lambda path: {})
    monkeypatch.setattr(r3eval.FrozenArms, "problems", lambda self: ["dream: changed: payguard/dream/engine.py"])
    monkeypatch.setattr(fe3, "load_sealed", boom)
    assert fe3.main(["--run"]) == 2
    assert "changed: payguard/dream/engine.py" in capsys.readouterr().out


# ---------------------------------------------------------------- the one shot ----------------------------
def test_one_shot_with_stubs_writes_results_once(monkeypatch, capsys):
    stub_everything(monkeypatch)
    assert fe3.main(["--run"]) == 0
    doc = json.loads(fe3.RESULTS_PATH.read_text(encoding="utf-8"))
    assert doc["split"]["name"] == fe3.SEALED_SPLIT and doc["split"]["n_episodes"] == len(DEV_SEEDS)
    assert set(doc["arms"]) == set(r3eval.ARMS)
    a = doc["arms"]["dreamer_full"]
    lo, hi = a["ci95"]["objective"]
    assert lo <= a["mean_episode"]["objective"] <= hi and len(a["per_episode_objective"]) == len(DEV_SEEDS)
    assert doc["bootstrap"]["n"] == r3eval.BOOTSTRAP_N and doc["bootstrap"]["seed"] == r3eval.BOOTSTRAP_SEED
    d = doc["decision"]
    assert d["h1"]["margin"] == 3.0 and d["h2"]["margin"] == 2.0
    assert d["h1"]["margin_vs_static"] == pytest.approx(0.0, abs=1e-6)  # dreamer_full IS static_rules here -> NO-GO
    assert d["verdict"] == "NO-GO"
    assert "cross_check_episode_means" in d and doc["git_rev"] and doc["started_at"] and doc["finished_at"]
    assert doc["report_only"]["h3_calibration_gap_trend"]["dreamer_full"]["slope_per_generation"] == pytest.approx(-1.0)
    assert set(doc["report_only"]["h4_time_to_immunity"]) == set(r3eval.ARMS)
    assert set(doc["spent_splits"]) == set(fe3.SPENT_SPLITS)
    for s in doc["spent_splits"].values():
        assert s["clean"] is False and s["arm"] == "dreamer_full" and "pooled" in s
    assert "static_rules" in doc["paired_margin_vs_dreamer_full"]
    before = fe3.RESULTS_PATH.read_bytes()
    assert fe3.main(["--run"]) == 2  # no re-runs
    assert fe3.RESULTS_PATH.read_bytes() == before
    assert "VERDICT" in capsys.readouterr().out
    assert f"payguard.{SEALED_PKG}" not in sys.modules


def test_friction_can_fail_h1_even_with_a_big_objective_margin(monkeypatch):
    """dreamer_full holds everything; the baselines release everything: a huge objective margin, but its honest block
    rate (100%) is above static_rules' (0%), so H1 fails on the friction clause."""

    class Arms(StubArms):
        def champion(self, arm):
            return AlwaysHold if arm == "dreamer_full" else AlwaysRelease

    stub_everything(monkeypatch, Arms())
    assert fe3.main(["--run"]) == 0
    d = json.loads(fe3.RESULTS_PATH.read_text(encoding="utf-8"))["decision"]
    assert d["h1"]["margin_vs_static"] > 3.0 and d["h2"]["pass"]
    assert d["h1"]["honest_block_rate"] > d["h1"]["static_honest_block_rate"] and d["verdict"] == "NO-GO"


# ---------------------------------------------------------------- source scan ---------------------------
SCANNED = ("payguard/r3eval.py", "scripts/final_eval_r3.py", "scripts/r3_dev_check.py", "scripts/freeze.py")
TEST_SEEDS = frozenset(base + i for base in (2 * 1000, 3 * 1000, 4 * 1000, 5 * 1000) for i in range(20))


def _tree(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    return src, ast.parse(src)


@pytest.mark.parametrize("rel", SCANNED)
def test_no_test_seed_literals_or_ranges(rel):
    src, tree = _tree(rel)
    assert not re.search(r"\b[2-5]0[01]\d\b", src), f"{rel}: a TEST seed literal appears in the text"
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            assert node.value not in TEST_SEEDS, f"{rel}: literal {node.value}"
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "range" and len(node.args) == 2
                and all(isinstance(a, ast.Constant) for a in node.args)):
            lo, hi = (a.value for a in node.args)
            assert not any(lo <= s < hi for s in TEST_SEEDS), f"{rel}: range({lo}, {hi}) overlaps TEST seeds"


def _functions_naming(tree, needle):
    """Names of the functions in which `needle` is imported or referenced; '<module>' for module level."""
    found = set()

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            text = ""
            if isinstance(child, ast.ImportFrom):
                text = (child.module or "") + " " + " ".join(a.name for a in child.names)
            elif isinstance(child, ast.Import):
                text = " ".join(a.name for a in child.names)
            elif isinstance(child, ast.Name):
                text = child.id
            elif isinstance(child, ast.Attribute):
                text = child.attr
            if needle in text.lower():
                found.add(fn)
            visit(child, inner)

    visit(tree, "<module>")
    return found


def test_the_sealed_package_is_only_named_inside_the_lazy_loader():
    for rel in SCANNED:
        _, tree = _tree(rel)
        where = _functions_naming(tree, SEALED_PKG)
        if rel == "scripts/final_eval_r3.py":
            assert where == {"load_sealed"}, where
        elif rel == "scripts/freeze.py":
            assert where <= {"round3_files"}, where  # a directory name only; it is never imported there
        else:
            assert where == set(), (rel, where)
