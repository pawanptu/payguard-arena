"""THE round-3 one-shot evaluation (pre-registered in docs/PREREG_R3.md; selection rule eval/r3/SELECTION_RULE.md).

This is the only script that builds TEST_HELDOUT3 episodes (the sealed package payguard/heldout3/). The package is
imported lazily, inside `load_sealed`, which only the --run path reaches after every guard has passed. Run from the
repo root:

  python -m scripts.final_eval_r3 --plan   what would run: arms, rule, freeze state. Builds and imports nothing sealed.
  python -m scripts.final_eval_r3 --run    THE one-shot. Refuses when eval/final_r3/results.json exists, when the
                                           defender freeze (eval/r3/defender_freeze.json: payguard/dream/*.py hashes,
                                           swarm_r2 files) does not hold, or when `python -m scripts.freeze --round 3`
                                           fails. Then every arm on TEST_HELDOUT3, H1 / H2 decided, bootstrap 95%
                                           intervals over episodes, and dreamer_full alone on the spent splits
                                           (TRAIN, TEST_SEEN, TEST_HELDOUT, TEST_HELDOUT2) marked "clean": false.
                                           Writes eval/final_r3/results.json (exclusive create). No re-runs.

Rule (see payguard/r3eval.py): H1 = dreamer_full >= 3.0 below BOTH static_rules and single_policy and honest block rate
<= static_rules'; H2 = dreamer_full >= 2.0 below dreamer_nowm; H3 / H4 report-only. All numbers are SIMULATED replays
on synthetic data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from payguard import r3eval
from payguard.evolve import replay_export as rx
from payguard.protocol import (HELDOUT_FAMILIES, TEST_HELDOUT_SEEDS, TEST_SEEN_SEEDS, TRAIN_FAMILIES, TRAIN_SEEDS,
                               episodes)
from payguard.r3eval import ARMS, DREAMER_ARMS, PRIMARY, FreezeError

ROOT = r3eval.ROOT
OUT_DIR = ROOT / "eval" / "final_r3"
RESULTS_PATH = OUT_DIR / "results.json"
FREEZE_PATH = ROOT / "eval" / "r3" / "defender_freeze.json"

SEALED_SPLIT = "TEST_HELDOUT3"
# Spent splits where dreamer_full is also reported. None is clean: TRAIN trained it, the others were already seen.
SPENT_SPLITS = ("TRAIN", "TEST_SEEN", "TEST_HELDOUT", "TEST_HELDOUT2")
SPENT_NOTE = {"TRAIN": "trained on", "TEST_SEEN": "spent in rounds 1-2", "TEST_HELDOUT": "spent in round 1",
              "TEST_HELDOUT2": "spent in round 2"}


def rel(path: Path) -> str:
    try:
        return Path(path).relative_to(ROOT).as_posix()
    except ValueError:  # a path outside the repo (tests redirect the outputs)
        return str(path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


# ---------------------------------------------------------------- guards ------------------------------
def freeze_problems() -> list[str]:
    """Everything that must hold before a sealed episode is built. Empty list = go."""
    out: list[str] = []
    try:
        arms = r3eval.FrozenArms(ROOT, FREEZE_PATH, verify=False)
        out += arms.problems()
    except FreezeError as exc:
        out.append(str(exc))
    from scripts import freeze  # the benchmark / sealed-package freeze (round 3 covers payguard/heldout3/*)
    if freeze.main(["--round", "3"]) != 0:
        out.append("python -m scripts.freeze --round 3 fails (see its output above)")
    return out


def load_arms() -> r3eval.ArmRunner:
    return r3eval.FrozenArms(ROOT, FREEZE_PATH)


# ---------------------------------------------------------------- splits ------------------------------
def load_sealed() -> dict:
    """Build TEST_HELDOUT3 with allow_test semantics. The ONLY place payguard.heldout3 is imported."""
    from payguard.dream.contract import assert_seed_allowed
    from payguard.heldout3 import protocol as p3  # lazy: sealed until the freeze has been verified

    for s in p3.TEST_HELDOUT3_SEEDS:
        assert_seed_allowed(s, allow_test=True)
    eps = [p3.episode_for(s, p3.HELDOUT3_FAMILIES) for s in p3.TEST_HELDOUT3_SEEDS]
    return {"name": SEALED_SPLIT, "seeds": [p3.TEST_HELDOUT3_SEEDS[0], p3.TEST_HELDOUT3_SEEDS[-1]],
            "families": list(p3.HELDOUT3_FAMILIES), "episodes": eps}


def load_spent() -> dict[str, list]:
    """Episodes of the already-spent splits. Building TEST seeds is allowed here and only here (the --run path)."""
    import payguard.heldout2  # noqa: F401  registers the round-2 family before its episodes are built
    from payguard.heldout2.protocol import HELDOUT2_FAMILIES, TEST_HELDOUT2_SEEDS
    from payguard.heldout2.protocol import episodes as h2_episodes

    return {"TRAIN": episodes(TRAIN_SEEDS, TRAIN_FAMILIES), "TEST_SEEN": episodes(TEST_SEEN_SEEDS, TRAIN_FAMILIES),
            "TEST_HELDOUT": episodes(TEST_HELDOUT_SEEDS, HELDOUT_FAMILIES),
            "TEST_HELDOUT2": h2_episodes(TEST_HELDOUT2_SEEDS, HELDOUT2_FAMILIES)}


# ---------------------------------------------------------------- plan ---------------------------------
def plan() -> int:
    print("Round-3 one-shot (no sealed package imported, no episode built, no defender run):")
    print(f"  arms ({len(ARMS)}): {', '.join(ARMS)}")
    print(f"  decisive split: {SEALED_SPLIT} (episodes come from payguard.heldout3 at --run time)")
    print(f"  H1 (decides GO / NO-GO): {PRIMARY} objective >= {r3eval.H1_MARGIN} below BOTH static_rules and "
          f"single_policy, and honest block rate <= static_rules'")
    print(f"  H2: {PRIMARY} objective >= {r3eval.H2_MARGIN} below dreamer_nowm")
    print("  H3 (calibration-gap trend from training logs) and H4 (time-to-immunity) are report-only")
    print(f"  intervals: percentile bootstrap, n={r3eval.BOOTSTRAP_N}, seed={r3eval.BOOTSTRAP_SEED}, over episodes")
    print(f"  also reported (not clean): {PRIMARY} on {', '.join(SPENT_SPLITS)}")
    print(f"  results file: {rel(RESULTS_PATH)} "
          f"({'EXISTS: --run would refuse' if RESULTS_PATH.exists() else 'absent: --run may proceed once guards pass'})")
    problems = freeze_problems()
    if problems:
        print("  freeze state: NOT READY")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  freeze state: holds")
    return 0


# ---------------------------------------------------------------- the one-shot -------------------------
def run_once() -> int:
    reason = r3eval.one_shot_refusal(RESULTS_PATH)
    if reason:
        print(reason)
        return 2
    problems = freeze_problems()
    if problems:
        print("REFUSED: the freeze does not hold, no sealed episode was built:")
        for p in problems:
            print(f"  - {p}")
        return 2
    arms = load_arms()
    started = now_iso()
    t0 = time.time()
    sealed = load_sealed()
    eps = sealed["episodes"]
    print(f"== {SEALED_SPLIT}: {len(eps)} episodes, {sum(len(e.events) for e in eps):,} payouts", flush=True)
    results = r3eval.run_arms(arms, eps, ARMS, log=lambda s: print(s, flush=True))
    decision = r3eval.decide(results)
    decision["cross_check_episode_means"] = r3eval.decide_on_episode_means(results)

    spent = {}
    for split, spent_eps in load_spent().items():
        r = r3eval.run_arm(PRIMARY, arms.champion(PRIMARY), spent_eps)
        spent[split] = {"clean": False, "why_not_clean": SPENT_NOTE[split], "arm": PRIMARY,
                        **r3eval.summarize_arm(r)}
        print(f"  [not clean] {split:13s} {PRIMARY} obj={r.objective:6.2f}", flush=True)

    gaps = {a: r3eval.calibration_trend(arms.calibration_gaps(a)) for a in DREAMER_ARMS
            if hasattr(arms, "calibration_gaps")}
    out = {
        "generated_at": now_iso(), "started_at": started, "finished_at": now_iso(), "git_rev": git_rev(),
        "simulated": True, "pre_registration": "docs/PREREG_R3.md; eval/r3/SELECTION_RULE.md",
        "objective": rx.OBJECTIVE_TEXT,
        "defender_freeze": {"path": rel(FREEZE_PATH),
                            "sha256": getattr(arms, "file_sha256", lambda: None)()},
        "split": {"name": sealed["name"], "seeds": sealed["seeds"], "families": sealed["families"],
                  "n_episodes": len(eps)},
        "bootstrap": {"method": "percentile 95% of the mean over episodes, resampling episodes with replacement",
                      "n": r3eval.BOOTSTRAP_N, "seed": r3eval.BOOTSTRAP_SEED},
        "decision": decision,
        "arms": {a: r3eval.summarize_arm(r) for a, r in results.items()},
        "paired_margin_vs_dreamer_full": r3eval.paired_margins(results),
        "report_only": {
            "h3_calibration_gap_trend": gaps,
            "h4_time_to_immunity": {a: {"immunity_attempts": r.pooled.immunity_attempts,
                                        "never_contained_rate": r.pooled.never_contained_rate}
                                    for a, r in results.items()}},
        "spent_splits": spent,
    }
    r3eval.write_once(RESULTS_PATH, json.dumps(rx.clean(out), indent=2, allow_nan=False) + "\n")
    h1, h2 = decision["h1"], decision["h2"]
    print(f"{SEALED_SPLIT} {PRIMARY} obj={h1['objective']:.2f}  vs static {h1['margin_vs_static']:+.2f}  "
          f"vs single_policy {h1['margin_vs_single']:+.2f}  block {h1['honest_block_rate']:.1f}% "
          f"(static {h1['static_honest_block_rate']:.1f}%)  H1 pass={h1['pass']}")
    print(f"H2 vs dreamer_nowm {h2['margin_vs_nowm']:+.2f}  pass={h2['pass']}")
    print(f"VERDICT (H1 decides): {decision['verdict']}")
    print(f"wrote {rel(RESULTS_PATH)} in {time.time() - t0:.0f}s")
    return 0


# ---------------------------------------------------------------- CLI ----------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="print what would run; builds nothing sealed")
    g.add_argument("--run", action="store_true", help="THE round-3 one-shot (refuses if results exist)")
    args = ap.parse_args(argv)
    return plan() if args.plan else run_once()


if __name__ == "__main__":
    sys.exit(main())
