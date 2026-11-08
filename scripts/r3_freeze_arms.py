"""Round-3 freeze (eval/r3/SELECTION_RULE.md section 4), TRAIN only. Never touches a TEST seed.

  python -m scripts.r3_freeze_arms --plan     print what would be trained; trains nothing
  python -m scripts.r3_freeze_arms --run      train the four dreamer arms on full TRAIN, write eval/r3/defender_freeze.json

Refuses unless eval/r3/dev_check.json exists and its K1 passed (a failed K1 means: stop, do not freeze, do not spend
TEST_HELDOUT3). dreamer_full = the dev-check winner's DreamConfig; the ablations flip exactly one switch
(`r3eval.arm_config`). Every arm is evolved on BOTH TRAIN families with evolution seed 0 and the same EVOLVE_CONFIG
(D-028 settings, budget guard off, D-044). Each finished arm is checkpointed with its exact parameters in
`eval/r3/freeze_arms.ckpt.jsonl` so a crash loses at most the arm in progress; full run logs go to `eval/r3/runs/`.
The freeze file is written once (`r3eval.write_once`); it hashes payguard/dream/*.py, so nothing in that package may
change afterwards. The working tree must be clean so the recorded git rev is the code that trained the arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from payguard import r3eval
from payguard.dream import training
from payguard.protocol import TRAIN_FAMILIES
from payguard.swarm.params import SwarmParams

ROOT = r3eval.ROOT
DEV_CHECK = ROOT / "eval" / "r3" / "dev_check.json"
CKPT = ROOT / "eval" / "r3" / "freeze_arms.ckpt.jsonl"
RUN_DIR = ROOT / "eval" / "r3" / "runs"
EVO_SEED = 0


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load_k1(path: Path = DEV_CHECK) -> dict:
    if not path.exists():
        raise SystemExit(f"{path.name} is missing: run `python -m scripts.r3_dev_check --run` first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not doc.get("k1", {}).get("pass"):
        raise SystemExit("K1 did not pass: SELECTION_RULE section 3 says stop. Nothing is frozen and TEST_HELDOUT3 "
                         "stays sealed. Publish the negative finding instead.")
    return doc


def load_ckpt(path: Path = CKPT) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["arm"]] = rec
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)

    if r3eval.FREEZE_PATH.exists():
        raise SystemExit(f"{r3eval.FREEZE_PATH.name} exists: the round-3 defenders are already frozen")
    dev = load_k1()
    base = r3eval.dream_config_from_dict(dev["winner_config"])
    cfgs = {arm: r3eval.arm_config(arm, base) for arm in r3eval.DREAMER_ARMS}
    print(f"winner {dev['winner']}  K1 margin {dev['k1']['margin']:+.2f}  -> freeze {list(cfgs)} on {TRAIN_FAMILIES}, "
          f"evo seed {EVO_SEED}, {training.EVOLVE_CONFIG}", flush=True)
    if args.plan:
        return 0

    if git("status", "--porcelain", "--", "payguard", "scripts"):
        raise SystemExit("payguard/ or scripts/ has uncommitted changes: commit first so the freeze records the real rev")
    rev = git("rev-parse", "HEAD")
    done = load_ckpt()
    for arm, cfg in cfgs.items():
        if arm in done:
            if done[arm]["dream_config"] != r3eval.dream_config_to_dict(cfg) or done[arm]["git_rev"] != rev:
                raise SystemExit(f"checkpoint for {arm} was made with another config or rev; move it aside deliberately")
            print(f"  {arm}: resumed from checkpoint", flush=True)
            continue
        print(f"  {arm}: training ...", flush=True)
        run = training.train_run(cfg, TRAIN_FAMILIES, EVO_SEED, log=lambda s: print(s, flush=True))
        log_path = training.save_run(run, RUN_DIR)
        rec = {"arm": arm, "git_rev": rev, "dream_config": r3eval.dream_config_to_dict(cfg),
               "swarm_params": dict(run["champion"]["params"]), "calibration_gap_ema": run["calibration_gap_ema"],
               "generations_run": run["generations_run"], "seconds": run["seconds"],
               "run_log": log_path.relative_to(ROOT).as_posix()}
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        with open(CKPT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[arm] = rec
        print(f"  {arm}: done in {run['seconds'] / 60:.1f} min, {run['generations_run']} generations", flush=True)

    champions = {arm: (SwarmParams(**done[arm]["swarm_params"]), cfgs[arm]) for arm in cfgs}
    selection = {"rule": "eval/r3/SELECTION_RULE.md section 4", "winner": dev["winner"],
                 "k1_margin": dev["k1"]["margin"], "evo_seed": EVO_SEED, "train_families": list(TRAIN_FAMILIES),
                 "dev_check_sha256": hashlib.sha256(DEV_CHECK.read_bytes()).hexdigest(),
                 "evolve_config": {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in training.EVOLVE_CONFIG.__dict__.items()},
                 "run_logs": {arm: done[arm]["run_log"] for arm in cfgs}}
    doc = r3eval.build_freeze_doc(champions, selection=selection, git_rev=rev, kind="swarm",
                                  calibration_gap_ema={a: done[a]["calibration_gap_ema"] for a in cfgs})
    r3eval.write_once(r3eval.FREEZE_PATH, json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(f"wrote {r3eval.FREEZE_PATH.relative_to(ROOT)} (git rev {rev[:7]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
