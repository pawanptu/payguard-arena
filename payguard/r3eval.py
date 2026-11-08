"""Round-3 evaluation logic (pure): arms registry, the pre-registered rule, bootstrap, frozen-arm loading, one-shot guard.

Pre-registration (docs/PREREG_R3.md, summarized):
  arms      static_rules, single_policy, swarm_r2, dreamer_full, dreamer_nowm, dreamer_noskeptic, dreamer_nocalib
  H1        round-1 D-022 rule unchanged: dreamer_full objective >= 3.0 below BOTH static_rules and single_policy,
            AND dreamer_full honest block rate <= static_rules'.      (decides GO / NO-GO)
  H2        dreamer_full objective >= 2.0 below dreamer_nowm.        (reported next to the verdict)
  H3, H4    report-only: calibration-gap trend from the training logs; time-to-immunity (Metrics.immunity_attempts).
  Numbers  means over the sealed episodes, with bootstrap 95% intervals over episodes (the rule uses the means).

Nothing in this module builds an episode on its own, imports the sealed package, or knows a TEST seed: callers
pass episodes in. `FrozenArms` (the production `ArmRunner`) loads the frozen defenders; tests inject stubs.

Schema of eval/r3/defender_freeze.json (written once, before any sealed episode exists; see `build_freeze_doc`):
  {"schema_version": 1, "frozen_at": ISO, "git_rev": str,
   "selection": {...free-form provenance, e.g. winner id and the dev_check.json hash...},
   "dream_files": {"payguard/dream/<file>.py": sha256 (CRLF-normalised), ...},
   "arms": {"dreamer_full": {"kind": "swarm" | "memory_swarm",
                              "swarm_params": SwarmParams as a flat dict,
                              "dream_config": DreamConfig as a dict (families as list),
                              "calibration_gap_ema": [optional list of floats, one per generation, for H3]},
            "dreamer_nowm": {...}, "dreamer_noskeptic": {...}, "dreamer_nocalib": {...}}}
The four dreamer arms must share every DreamConfig field except their one switch (checked on load).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .dream.contract import DreamConfig
from .sim.metrics import Metrics
from .sim.metrics import evaluate as aggregate
from .sim.runner import run_episode
from .sim.types import Episode
from .swarm.objective import objective
from .swarm.params import SwarmParams

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "eval" / "r3" / "defender_freeze.json"
R2_FREEZE_PATH = ROOT / "eval" / "final_r2" / "defender_freeze.json"
R2_CHAMPION_REL = "eval/r2/champion.json"
DREAM_DIR_REL = "payguard/dream"
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------------------------- registry
BASELINE_ARMS = ("static_rules", "single_policy", "swarm_r2")
DREAMER_ARMS = ("dreamer_full", "dreamer_nowm", "dreamer_noskeptic", "dreamer_nocalib")
ARMS = BASELINE_ARMS + DREAMER_ARMS
PRIMARY = "dreamer_full"
SINGLE_SPECIALIST = "policy"        # the best single specialist on TRAIN, fixed in round 1 (D-033): "single_policy"
ARM_KINDS = ("swarm", "memory_swarm")

# Pre-registered (PREREG_R3). Do not edit after the one-shot.
H1_MARGIN = 3.0
H2_MARGIN = 2.0
EPS = 1e-9                           # float slack: a margin of exactly 3.0 in decimals must not fail on 2.9999999999999996
BOOTSTRAP_N = 2 * 10**3             # 2,000 resamples (written so no TEST-seed-range literal appears in the source)
BOOTSTRAP_SEED = 0


@runtime_checkable
class ArmRunner(Protocol):
    def champion(self, arm: str) -> Callable[[], Any]:
        """A factory of FRESH defenders (decide / learn, as payguard.sim.runner.Defender) for `arm`, one per episode."""
        ...


# ---------------------------------------------------------------------------------------------- results
@dataclass(frozen=True)
class ArmResult:
    """One arm over a set of episodes: the pooled headline (evaluate over all runs) and the per-episode Metrics."""

    arm: str
    pooled: Metrics
    per_episode: tuple[Metrics, ...] = ()

    @property
    def objective(self) -> float:
        return objective(self.pooled)

    @property
    def episode_objectives(self) -> list[float]:
        return [objective(m) for m in self.per_episode]

    @property
    def mean_episode_objective(self) -> float:
        v = self.episode_objectives
        return float(np.mean(v)) if v else float("nan")


def run_arm(arm: str, factory: Callable[[], Any], episodes: Sequence[Episode]) -> ArmResult:
    """A fresh defender per episode. Per-episode Metrics and the pooled headline come from the SAME runs."""
    runs = [run_episode(ep, factory()) for ep in episodes]
    return ArmResult(arm, aggregate(runs), tuple(aggregate([r]) for r in runs))


def run_arms(runner: ArmRunner, episodes: Sequence[Episode], arms: Sequence[str] = ARMS,
             log: Callable[[str], None] | None = None) -> dict[str, ArmResult]:
    out = {}
    for arm in arms:
        out[arm] = run_arm(arm, runner.champion(arm), episodes)
        if log:
            log(f"  {arm:18s} obj={out[arm].objective:6.2f}  leak={out[arm].pooled.leakage_pct:5.1f}%  "
                f"block={out[arm].pooled.honest_block_rate:4.1f}%")
    return out


def _pooled(x: ArmResult | Metrics) -> Metrics:
    return x.pooled if isinstance(x, ArmResult) else x


# ---------------------------------------------------------------------------------------------- bootstrap
def bootstrap_ci(values: Sequence[float], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval of the MEAN, resampling episodes with replacement. Deterministic in `seed`."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_diff_ci(a: Sequence[float], b: Sequence[float], n: int = BOOTSTRAP_N,
                   seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """CI of mean(a - b) over episodes, pairing by episode (both arms ran the same episodes)."""
    if len(a) != len(b):
        raise ValueError("paired_diff_ci needs equally long sequences")
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n, seed)


# ---------------------------------------------------------------------------------------------- the rule
def _need(results: Mapping[str, Any], names: Sequence[str]) -> None:
    missing = [n for n in names if n not in results]
    if missing:
        raise KeyError(f"decide() needs arms {missing}")


def _apply_rule(of: float, os_: float, o1: float, on: float, block_full: float, block_static: float, basis: str) -> dict:
    m_static, m_single, m_nowm = os_ - of, o1 - of, on - of
    h1 = {"rule": f"dreamer_full objective >= {H1_MARGIN} below BOTH static_rules and single_policy, and honest block "
                  f"rate <= static_rules'",
          "margin": H1_MARGIN, "objective": of, "static_rules": os_, "single_policy": o1,
          "margin_vs_static": m_static, "margin_vs_single": m_single,
          "honest_block_rate": block_full, "static_honest_block_rate": block_static}
    h1["pass"] = bool(m_static >= H1_MARGIN - EPS and m_single >= H1_MARGIN - EPS and block_full <= block_static + EPS)
    h2 = {"rule": f"dreamer_full objective >= {H2_MARGIN} below dreamer_nowm", "margin": H2_MARGIN,
          "objective": of, "dreamer_nowm": on, "margin_vs_nowm": m_nowm, "pass": bool(m_nowm >= H2_MARGIN - EPS)}
    return {"basis": basis, "h1": h1, "h2": h2, "verdict": "GO" if h1["pass"] else "NO-GO", "h2_pass": h2["pass"]}


_RULE_ARMS = ("dreamer_full", "static_rules", "single_policy", "dreamer_nowm")


def decide(results: Mapping[str, ArmResult | Metrics]) -> dict:
    """Apply H1 and H2 exactly as pre-registered, on the pooled headline objective (round-1 D-022 convention).

    H1 passes iff  static - full >= 3.0  AND  single_policy - full >= 3.0  AND  full block <= static block.
    H2 passes iff  nowm - full >= 2.0.       `verdict` is H1's (GO / NO-GO); H2 is reported next to it.
    """
    _need(results, _RULE_ARMS)
    full, static, single, nowm = (_pooled(results[a]) for a in _RULE_ARMS)
    return _apply_rule(objective(full), objective(static), objective(single), objective(nowm),
                       full.honest_block_rate, static.honest_block_rate,
                       "pooled objective over all episodes (payguard.sim.metrics.evaluate), as in round 1")


def decide_on_episode_means(results: Mapping[str, ArmResult]) -> dict:
    """The same rule on plain means of per-episode objectives / block rates. A cross-check only: the two bases differ
    slightly because the pooled objective weighs episodes by their USD and payout counts."""
    _need(results, _RULE_ARMS)
    mo = {a: results[a].mean_episode_objective for a in _RULE_ARMS}
    mb = {a: float(np.mean([m.honest_block_rate for m in results[a].per_episode])) for a in ("dreamer_full", "static_rules")}
    return _apply_rule(mo["dreamer_full"], mo["static_rules"], mo["single_policy"], mo["dreamer_nowm"],
                       mb["dreamer_full"], mb["static_rules"], "plain mean of per-episode objectives (cross-check)")


# ---------------------------------------------------------------------------------------------- report-only
def calibration_trend(gap_ema: Sequence[float]) -> dict:
    """H3 (report-only): does the dream-vs-reality calibration gap shrink over generations? Least-squares slope."""
    g = [float(x) for x in gap_ema if x is not None and math.isfinite(x)]
    if len(g) < 2:
        return {"n": len(g), "first": g[0] if g else None, "last": g[-1] if g else None, "slope_per_generation": None}
    slope = float(np.polyfit(np.arange(len(g)), np.asarray(g), 1)[0])
    return {"n": len(g), "first": g[0], "last": g[-1], "slope_per_generation": slope}


def summarize_arm(r: ArmResult, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """Pooled metrics, per-episode objectives and bootstrap CIs (objective, leakage, honest block, immunity)."""
    out: dict[str, Any] = {"pooled": {**asdict(r.pooled), "objective": r.objective}}
    if r.per_episode:
        series = {"objective": r.episode_objectives,
                  "leakage_pct": [m.leakage_pct for m in r.per_episode],
                  "honest_block_rate": [m.honest_block_rate for m in r.per_episode],
                  "immunity_attempts": [m.immunity_attempts for m in r.per_episode]}
        out["n_episodes"] = len(r.per_episode)
        out["per_episode_objective"] = series["objective"]
        out["mean_episode"] = {k: float(np.mean(v)) for k, v in series.items()}
        out["ci95"] = {k: list(bootstrap_ci(v, n, seed)) for k, v in series.items()}
    return out


def paired_margins(results: Mapping[str, ArmResult], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """Mean per-episode objective gap (other - dreamer_full) with its paired bootstrap 95% interval."""
    out = {}
    full = results.get(PRIMARY)
    if full is None or not full.per_episode:
        return out
    for other, r in results.items():
        if other == PRIMARY or len(r.per_episode) != len(full.per_episode):
            continue
        a, b = r.episode_objectives, full.episode_objectives
        out[other] = {"mean": float(np.mean(np.asarray(a) - np.asarray(b))), "ci95": list(paired_diff_ci(a, b, n, seed))}
    return out


# ---------------------------------------------------------------------------------------------- one-shot guard
def one_shot_refusal(results_path: Path) -> str | None:
    """None when the one shot may run; otherwise the reason. A rerun has to be a visible, deliberate act."""
    if Path(results_path).exists():
        return (f"{Path(results_path).name} exists: the round-3 one-shot already ran. A rerun must be deliberate: "
                "delete the file and record why in docs/DECISIONS.md first.")
    return None


def write_once(path: Path, text: str) -> None:
    """Create `path` exclusively (never overwrites): the filesystem-level half of the one-shot guard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------------------------- hashing / freeze
class FreezeError(Exception):
    """The defender freeze is missing, malformed or no longer matches the code on disk."""


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def dream_hashes(root: Path = ROOT) -> dict[str, str]:
    base = Path(root) / DREAM_DIR_REL
    files = sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts) if base.is_dir() else []
    return {p.relative_to(root).as_posix(): digest(p) for p in files}


def _diff(frozen: Mapping[str, str], now: Mapping[str, str]) -> list[str]:
    return ([f"changed: {k}" for k in now if k in frozen and now[k] != frozen[k]]
            + [f"not in freeze: {k}" for k in now if k not in frozen]
            + [f"missing: {k}" for k in frozen if k not in now])


def arm_config(arm: str, base: DreamConfig) -> DreamConfig:
    return DreamConfig.for_arm(arm, **{f.name: getattr(base, f.name) for f in fields(base)})


def dream_config_from_dict(d: Mapping[str, Any]) -> DreamConfig:
    d = dict(d)
    if "families" in d:
        d["families"] = tuple(d["families"])
    return DreamConfig(**d)


def dream_config_to_dict(cfg: DreamConfig) -> dict:
    d = asdict(cfg)
    d["families"] = list(d["families"])
    return d


def build_freeze_doc(champions: Mapping[str, tuple[SwarmParams, DreamConfig]], *, selection: Mapping[str, Any] | None = None,
                     git_rev: str = "unknown", kind: str = "swarm", root: Path = ROOT,
                     calibration_gap_ema: Mapping[str, Sequence[float]] | None = None) -> dict:
    """The document for eval/r3/defender_freeze.json. `champions` maps each dreamer arm to (its evolved SwarmParams,
    its DreamConfig). Refuses when the four arms are not identical but for their one switch."""
    missing = [a for a in DREAMER_ARMS if a not in champions]
    if missing:
        raise FreezeError(f"champions missing for {missing}")
    if kind not in ARM_KINDS:
        raise FreezeError(f"kind must be one of {ARM_KINDS}")
    arms = {}
    for a in DREAMER_ARMS:
        params, cfg = champions[a]
        arms[a] = {"kind": kind, "swarm_params": asdict(params), "dream_config": dream_config_to_dict(cfg)}
        if calibration_gap_ema and a in calibration_gap_ema:
            arms[a]["calibration_gap_ema"] = [float(x) for x in calibration_gap_ema[a]]
    doc = {"schema_version": SCHEMA_VERSION, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_rev": git_rev, "selection": dict(selection or {}), "dream_files": dream_hashes(root), "arms": arms}
    problems = validate_freeze_doc(doc)
    if problems:
        raise FreezeError("; ".join(problems))
    return doc


def validate_freeze_doc(doc: Mapping[str, Any]) -> list[str]:
    """Structural checks only (no disk): schema version, all four arms, constructible params / configs, and the
    ablation arms differ from dreamer_full in exactly their one DreamConfig switch."""
    problems = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    arms = doc.get("arms", {})
    problems += [f"arm missing: {a}" for a in DREAMER_ARMS if a not in arms]
    if not doc.get("dream_files"):
        problems.append("dream_files is empty")
    cfgs = {}
    for a in (x for x in DREAMER_ARMS if x in arms):
        spec = arms[a]
        try:
            SwarmParams(**spec["swarm_params"])
            cfgs[a] = dream_config_from_dict(spec["dream_config"])
        except Exception as exc:  # noqa: BLE001 - any construction error is a freeze problem
            problems.append(f"{a}: {type(exc).__name__}: {exc}")
        if spec.get("kind") not in ARM_KINDS:
            problems.append(f"{a}: kind must be one of {ARM_KINDS}")
    if "dreamer_full" in cfgs:
        for a, cfg in cfgs.items():
            if cfg != arm_config(a, cfgs["dreamer_full"]):
                problems.append(f"{a}: DreamConfig differs from dreamer_full by more than its one switch")
        if not cfgs["dreamer_full"].use_wm or not cfgs["dreamer_full"].use_skeptic \
                or not cfgs["dreamer_full"].use_calibration:
            problems.append("dreamer_full must have every switch on")
    return problems


def verify_r2_baseline(root: Path = ROOT, freeze_path: Path | None = None) -> list[str]:
    """swarm_r2 must still be what was frozen in round 2: every file listed in eval/final_r2/defender_freeze.json
    (champion + payguard/robust + payguard/swarm), plus no new file in those directories."""
    freeze_path = Path(freeze_path) if freeze_path else Path(root) / "eval" / "final_r2" / "defender_freeze.json"
    if not freeze_path.is_file():
        return [f"{freeze_path.name} (round 2) is missing"]
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))["files"]
    now = {}
    dirs = {Path(k).parent.as_posix() for k in frozen if k.endswith(".py")}
    for rel in frozen:
        p = Path(root) / rel
        if p.is_file():
            now[rel] = digest(p)
    for d in dirs:
        for p in sorted((Path(root) / d).glob("*.py")):
            now.setdefault(p.relative_to(root).as_posix(), digest(p))
    return ["round-2 " + s for s in _diff(frozen, now)]


# ---------------------------------------------------------------------------------------------- FrozenArms
class FrozenArms:
    """The production `ArmRunner`: baselines from the existing round-1/2 artifacts, dreamers from the freeze file.

      static_rules   payguard.baselines.static_rules.StaticRules                      (frozen with the benchmark)
      single_policy  make_single("policy", default_params())                         (round 1, D-033)
      swarm_r2       make_r2_defender(eval/r2/champion.json), verified against eval/final_r2/defender_freeze.json
      dreamer_*      make_swarm / make_memory_swarm(SwarmParams from defender_freeze.json), L1 weights learning

    Construction verifies the freeze (payguard/dream/*.py hashes, swarm_r2 files) and raises `FreezeError` on drift.
    """

    def __init__(self, root: Path = ROOT, freeze_path: Path | None = None, *, verify: bool = True):
        self.root = Path(root)
        self.freeze_path = Path(freeze_path) if freeze_path else self.root / "eval" / "r3" / "defender_freeze.json"
        self.doc = load_freeze(self.freeze_path)
        if verify:
            problems = self.problems()
            if problems:
                raise FreezeError("defender freeze does not hold:\n  " + "\n  ".join(problems))

    def problems(self) -> list[str]:
        out = validate_freeze_doc(self.doc)
        out += [f"dream: {s}" for s in _diff(self.doc.get("dream_files", {}), dream_hashes(self.root))]
        out += verify_r2_baseline(self.root)
        return out

    def file_sha256(self) -> str:
        return digest(self.freeze_path)

    def champion(self, arm: str) -> Callable[[], Any]:
        if arm == "static_rules":
            from .baselines.static_rules import StaticRules
            return StaticRules
        if arm == "single_policy":
            from .swarm.params import default_params
            from .swarm.single import make_single
            d = default_params()
            return lambda: make_single(SINGLE_SPECIALIST, d)
        if arm == "swarm_r2":
            from .robust.r2 import make_r2_defender
            path = self.root / R2_CHAMPION_REL
            return lambda: make_r2_defender(path)
        if arm in DREAMER_ARMS:
            spec = self.doc["arms"][arm]
            params = SwarmParams(**spec["swarm_params"])
            if spec["kind"] == "memory_swarm":
                from .swarm.memory import make_memory_swarm
                return lambda: make_memory_swarm(params, None, name=arm)
            from .swarm.coordinator import make_swarm
            return lambda: make_swarm(params, learn_weights=True)
        raise KeyError(arm)

    def calibration_gaps(self, arm: str = PRIMARY) -> list[float]:
        return list(self.doc["arms"][arm].get("calibration_gap_ema", []))


def load_freeze(path: Path = FREEZE_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FreezeError(f"{path.name} is missing: the round-3 defenders are not frozen yet "
                          "(eval/r3/SELECTION_RULE.md section 4).")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FreezeError(f"{path.name} is not valid JSON: {exc}") from exc
