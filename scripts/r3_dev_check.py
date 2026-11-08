"""Round-3 dev check: eval/r3/SELECTION_RULE.md sections 2-3, on TRAIN only. Never touches a TEST seed.

  python -m scripts.r3_dev_check --plan                          print the grid and protocol; trains nothing
  python -m scripts.r3_dev_check --run --train MODULE:FUNCTION   grid c1-c4 -> LOFO -> winner -> K1 (evo seeds 0, 1)

`train(cfg: DreamConfig, families: tuple[str, ...], evo_seed: int) -> SwarmParams` is injected: it evolves a
defender with dreams on, training on `families` ONLY (fitness TRAIN_A, gate TRAIN_B), and returns the champion's
parameters. The real one comes with the loop integration (default: payguard.dream.training:train); until it exists
`--run` stops with a clear message. Tests inject a stub.

Section 2: LOFO(c) = mean over f in TRAIN_FAMILIES of the objective (D-019) of train(c, other family) scored on the
TRAIN_B seeds built with family f only (`payguard.protocol.episode_for(seed, (f,))`). Winner = lowest LOFO, ties to the
lower id. Section 3 (K1): winner vs its dreamer_nowm twin (use_wm=False), evolution seeds 0 and 1; passes iff the
twin's mean LOFO is >= 1.0 higher. The DreamConfig handed to `train` has `families` = the training families, so dreams
can never be drawn from the held-out family. Results are memoized per (cfg, families, evo_seed): train is
deterministic by contract, so the winner's seed-0 run is not repeated.

Checkpoint (D-044): every finished fold is appended to `<out>.folds.jsonl` (next to the output) and reloaded on a
restart, so a crash after hours of training loses at most the fold in progress. Because `train` is deterministic a
resumed fold equals a re-trained one; the checkpoint is deleted only by hand, never by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from payguard import r3eval
from payguard.dream.contract import DreamConfig, assert_seed_allowed
from payguard.evolve import replay_export as rx
from payguard.evolve.splits import TRAIN_B
from payguard.protocol import TRAIN_FAMILIES, episode_for
from payguard.sim.metrics import Metrics
from payguard.swarm.objective import objective
from payguard.swarm.params import SwarmParams

ROOT = r3eval.ROOT
OUT_PATH = ROOT / "eval" / "r3" / "dev_check.json"
DEFAULT_TRAIN = "payguard.dream.training:train"

# SELECTION_RULE section 1 (fixed): id -> (share_posterior, share_uncertain, share_recombined, lambda0)
GRID: dict[str, tuple[float, float, float, float]] = {
    "c1": (0.50, 0.25, 0.25, 0.3),
    "c2": (0.50, 0.25, 0.25, 0.6),
    "c3": (0.34, 0.33, 0.33, 0.3),
    "c4": (0.34, 0.33, 0.33, 0.6),
}
EVO_SEED_GRID = 0
EVO_SEEDS_K1 = (0, 1)
K1_MARGIN = 1.0

TrainFn = Callable[[DreamConfig, "tuple[str, ...]", int], SwarmParams]
ScoreFn = Callable[[SwarmParams, list], Metrics]


def grid_config(cid: str) -> DreamConfig:
    sp, su, sr, lam = GRID[cid]
    return DreamConfig(share_posterior=sp, share_uncertain=su, share_recombined=sr, lambda0=lam)


def default_score(params: SwarmParams, eps: list) -> Metrics:
    from payguard.evolve.evaluation import evaluate
    return evaluate(params, eps, True)


class DevCheck:
    def __init__(self, train: TrainFn, *, score: ScoreFn = default_score, seeds: Sequence[int] = TRAIN_B,
                 families: tuple[str, ...] = TRAIN_FAMILIES, log: Callable[[str], None] = lambda s: None,
                 checkpoint: Path | None = None):
        for s in seeds:
            assert_seed_allowed(s)  # TRAIN or dev seeds only; a TEST seed raises here
        self.train, self.score, self.seeds, self.families, self.log = train, score, tuple(seeds), tuple(families), log
        self._eps = {f: [episode_for(s, (f,)) for s in self.seeds] for f in self.families}
        self._memo: dict[tuple, dict] = {}
        self.train_calls = 0
        self.resumed = 0
        self.checkpoint = checkpoint
        if checkpoint is not None and checkpoint.exists():
            for line in checkpoint.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._memo[self._key_from_record(rec)] = rec["fold"]
            self.resumed = len(self._memo)
            self.log(f"  resumed {len(self._memo)} finished fold(s) from {checkpoint.name}")

    @staticmethod
    def _ckpt_key(cfg: DreamConfig, held_out: str, evo_seed: int) -> str:
        return json.dumps({"config": r3eval.dream_config_to_dict(cfg), "held_out": held_out, "evo_seed": int(evo_seed)},
                          sort_keys=True)

    @staticmethod
    def _key_from_record(rec: dict) -> tuple:
        k = json.loads(rec["key"])
        return (r3eval.dream_config_from_dict(k["config"]), k["held_out"], k["evo_seed"])

    def fold(self, cfg: DreamConfig, held_out: str, evo_seed: int) -> dict:
        """Train on every family but `held_out`, score on the TRAIN_B episodes of `held_out` only."""
        train_fams = tuple(f for f in self.families if f != held_out)
        key = (cfg, held_out, evo_seed)
        if key not in self._memo:
            self.train_calls += 1
            params = self.train(replace(cfg, families=train_fams), train_fams, evo_seed)
            m = self.score(params, self._eps[held_out])
            self._memo[key] = {"held_out": held_out, "train_families": list(train_fams), "evo_seed": evo_seed,
                               "objective": objective(m), "leakage_pct": m.leakage_pct,
                               "honest_block_rate": m.honest_block_rate, "honest_unpaid_rate": m.honest_unpaid_rate,
                               "params_sha256": hashlib.sha256(
                                   json.dumps(asdict(params), sort_keys=True).encode()).hexdigest()}
            self.log(f"    held out {held_out:11s} evo_seed={evo_seed} obj={self._memo[key]['objective']:6.2f}")
            if self.checkpoint is not None:
                self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                with open(self.checkpoint, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": self._ckpt_key(cfg, held_out, evo_seed), "fold": self._memo[key]}) + "\n")
        return self._memo[key]

    def lofo(self, cfg: DreamConfig, evo_seed: int) -> dict:
        folds = [self.fold(cfg, f, evo_seed) for f in self.families]
        return {"lofo": sum(f["objective"] for f in folds) / len(folds), "folds": folds}

    def run(self) -> dict:
        grid = {}
        for cid in GRID:
            self.log(f"  {cid}: shares {GRID[cid][:3]} lambda0 {GRID[cid][3]}")
            grid[cid] = {"config": r3eval.dream_config_to_dict(grid_config(cid)), **self.lofo(grid_config(cid), EVO_SEED_GRID)}
        winner = min(GRID, key=lambda c: (grid[c]["lofo"], c))  # ties go to the lower id
        wcfg = grid_config(winner)
        twin = replace(wcfg, use_wm=False)
        self.log(f"  winner {winner} (LOFO {grid[winner]['lofo']:.2f}); K1 with evo seeds {EVO_SEEDS_K1}")
        k1 = {"winner": {s: self.lofo(wcfg, s) for s in EVO_SEEDS_K1},
              "twin_nowm": {s: self.lofo(twin, s) for s in EVO_SEEDS_K1}}
        mean = lambda arm: sum(k1[arm][s]["lofo"] for s in EVO_SEEDS_K1) / len(EVO_SEEDS_K1)  # noqa: E731
        w, t = mean("winner"), mean("twin_nowm")
        k1_out = {"evo_seeds": list(EVO_SEEDS_K1), "winner_mean_lofo": w, "twin_mean_lofo": t, "margin": t - w,
                  "required_margin": K1_MARGIN, "pass": bool(t - w >= K1_MARGIN - r3eval.EPS),
                  "winner": {str(s): v for s, v in k1["winner"].items()},
                  "twin_nowm": {str(s): v for s, v in k1["twin_nowm"].items()}}
        return {"rule": "eval/r3/SELECTION_RULE.md sections 2-3", "split": "TRAIN_B (leave-one-family-out)",
                "seeds": list(self.seeds), "families": list(self.families), "grid": grid, "winner": winner,
                "winner_config": r3eval.dream_config_to_dict(wcfg), "k1": k1_out, "train_calls": self.train_calls,
                "folds_resumed_from_checkpoint": self.resumed,
                "verdict": "K1 passed: proceed to the freeze" if k1_out["pass"]
                           else "K1 FAILED: stop, publish the negative finding, do not run TEST_HELDOUT3"}


def resolve_train(spec: str = DEFAULT_TRAIN) -> TrainFn:
    mod, _, fn = spec.partition(":")
    try:
        return getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"no real train function yet ({spec}: {exc}). The loop integration provides "
                         "train(cfg: DreamConfig, families: tuple[str, ...], evo_seed: int) -> SwarmParams; "
                         "pass --train MODULE:FUNCTION once it exists.") from exc


def run_dev_check(train: TrainFn | None = None, out_path: Path = OUT_PATH, **kw) -> dict:
    if out_path.exists():
        raise SystemExit(f"{out_path.name} exists: SELECTION_RULE.md may not change after the first scoring run. "
                         "Delete it deliberately (and log why) to repeat.")
    train = train if train is not None else resolve_train()
    t0 = time.time()
    kw.setdefault("checkpoint", out_path.with_suffix(".folds.jsonl"))
    doc = DevCheck(train, log=lambda s: print(s, flush=True), **kw).run()
    doc["seconds"] = round(time.time() - t0, 1)
    r3eval.write_once(out_path, json.dumps(rx.clean(doc), indent=2, allow_nan=False) + "\n")
    print(f"winner {doc['winner']}  K1 margin {doc['k1']['margin']:+.2f} (need >= {K1_MARGIN})  "
          f"-> {doc['verdict']}\nwrote {out_path}")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--train", default=DEFAULT_TRAIN, help="MODULE:FUNCTION of the real train (default: %(default)s)")
    args = ap.parse_args(argv)
    if args.plan:
        print(f"grid: {GRID}\nLOFO folds: {TRAIN_FAMILIES}, scored on TRAIN_B {TRAIN_B[0]}..{TRAIN_B[-1]}\n"
              f"K1: evo seeds {EVO_SEEDS_K1}, margin >= {K1_MARGIN}; output {OUT_PATH.relative_to(ROOT)}")
        return 0
    run_dev_check(resolve_train(args.train))
    return 0


if __name__ == "__main__":
    sys.exit(main())
