"""swarm_r2 selection: leave-one-family-out (LOFO) on TRAIN, under the rule in eval/r2/SELECTION_RULE.md.

    python -m scripts.r2_select                 full run (about 35-45 min with 12 workers), writes
                                                eval/r2/selection.json and eval/r2/champion.json
    python -m scripts.r2_select --fresh         ignore the GA cache in eval/r2/cache/ (same result, just slower)
    python -m scripts.r2_select --quick         tiny smoke run (pop 4, 2 generations, 1 grid point): output goes to
                                                eval/r2/quick/ and never touches the real files

TRAIN seeds 1000-1019 and families ring + structuring only (payguard.robust.folds refuses anything else).
All numbers are SIMULATED replays on synthetic data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from payguard.robust import anchored
from payguard.robust.folds import DIRECTIONS, Runner, full_train_specs, lofo_folds
from payguard.robust.layer import calibrate
from payguard.sim.metrics import Metrics
from payguard.swarm.objective import objective
from payguard.swarm.params import SwarmParams, default_params

ROOT = Path(__file__).resolve().parent.parent
R2_DIR = ROOT / "eval" / "r2"
CHAMPION_EVOLVED = ROOT / "eval" / "evolution" / "champion.json"

LAM_GRID = (0.0, 20.0, 80.0)
BETA_GRID = ((0.005, 0.02), (0.01, 0.03), (0.01, 0.05), (0.02, 0.05), (0.02, 0.08))
GA_SEED = 0
MARGIN = 3.0
TRAIN_SLACK = 3.0
BLOCK_MARGIN = 0.75     # points below static_rules' honest block rate (Amendment 1)
BAND = 1.5              # noise band of the LOFO score for the acceptable set (Amendment 1)
BLOCK_TIE = 0.25
SIMPLICITY = {"C2": 0, "C1": 1, "C3": 2}


def r(x: float) -> float:
    return round(float(x), 4)


def summary(m: Metrics) -> dict:
    return {"objective": r(objective(m)), "leakage_pct": r(m.leakage_pct), "honest_block_rate": r(m.honest_block_rate),
            "honest_unpaid_rate": r(m.honest_unpaid_rate),
            "leakage_pct_by_family": {k: r(v) for k, v in m.leakage_pct_by_family.items()}}


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


class Selector:
    def __init__(self, runner: Runner, out_dir: Path, cache_dir: Path, *, fresh: bool, pop: int, gens: int,
                 lam_grid, beta_grid):
        self.runner, self.out_dir, self.cache_dir = runner, out_dir, cache_dir
        self.fresh, self.pop, self.gens = fresh, pop, gens
        self.lam_grid, self.beta_grid = tuple(lam_grid), tuple(beta_grid)
        self.folds = lofo_folds()
        self.full = full_train_specs()
        cache_dir.mkdir(parents=True, exist_ok=True)

    def log(self, s: str) -> None:
        print(s, flush=True)

    # ------------------------------------------------------------ helpers
    def fold_metrics(self, dspecs: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
        """dspecs[name] = one defender spec per fold (same order as self.folds) -> {name: {fold tag: summary}}.
        All jobs of the batch are submitted together so the pool stays busy."""
        jobs, keys = [], []
        for name, per_fold in dspecs.items():
            for fold, ds in zip(self.folds, per_fold):
                jobs.append(("metrics", ds, fold.evaluate))
                keys.append((name, fold.tag))
        res = self.runner.map(jobs)
        out: dict[str, dict[str, dict]] = {n: {} for n in dspecs}
        for (name, tag), m in zip(keys, res):
            out[name][tag] = summary(m)
        return out

    def full_metrics(self, dspecs: dict[str, dict]) -> dict[str, dict]:
        res = self.runner.map([("metrics", ds, self.full) for ds in dspecs.values()])
        return {n: summary(m) for n, m in zip(dspecs, res)}

    def aggregate(self, folds: dict[str, dict]) -> dict:
        """Direction objective/block = mean of the two folds of the direction; LOFO score = mean of directions."""
        dirs = {}
        for a, b in DIRECTIONS:
            d = f"{a}->{b}"
            rows = [folds[f"{d}@h{h}"] for h in (0, 1)]
            dirs[d] = {"objective": r(np.mean([x["objective"] for x in rows])),
                       "leakage_pct": r(np.mean([x["leakage_pct"] for x in rows])),
                       "honest_block_rate": r(np.mean([x["honest_block_rate"] for x in rows])),
                       "honest_unpaid_rate": r(np.mean([x["honest_unpaid_rate"] for x in rows]))}
        score = r(np.mean([v["objective"] for v in dirs.values()]))
        return {"directions": dirs, "lofo_score": score,
                "worst_direction": r(max(v["objective"] for v in dirs.values())),
                "lofo_leakage_pct": r(np.mean([v["leakage_pct"] for v in dirs.values()])),
                "lofo_honest_block_rate": r(np.mean([v["honest_block_rate"] for v in dirs.values()]))}

    # ------------------------------------------------------------ references
    def references(self) -> dict:
        champ = json.loads(CHAMPION_EVOLVED.read_text(encoding="utf-8"))["params"]
        dflt = asdict(default_params())
        defs = {
            "static_rules": {"kind": "static"},
            "single_policy": {"kind": "single", "name": "policy"},
            "swarm_l1": {"kind": "swarm", "params": dflt, "learn": True},
            "swarm_evolved": {"kind": "swarm", "params": champ, "learn": True},
            "swarm_evolved_memory": {"kind": "evo_mem", "params": champ},
        }
        fm = self.fold_metrics({n: [d] * len(self.folds) for n, d in defs.items()})
        full = self.full_metrics(defs)
        refs = {}
        for n in defs:
            refs[n] = {"folds": fm[n], **self.aggregate(fm[n]), "full_train": full[n]}
            self.log(f"  ref {n:22s} LOFO {refs[n]['lofo_score']:7.2f}  block {refs[n]['lofo_honest_block_rate']:5.2f}"
                     f"  full-TRAIN {full[n]['objective']:6.2f}")
        refs["_note"] = ("swarm_evolved / swarm_evolved_memory (champion g17-7) were evolved on ALL TRAIN seeds and both "
                         "families, so their LOFO rows are in-sample for them and optimistic.")
        return refs

    # ------------------------------------------------------------ C1
    def ga(self, lam: float, scope: str, fit_specs, tag: int) -> dict:
        path = self.cache_dir / f"ga_lam{lam:g}_{scope.replace('>', '').replace('@', '_')}.json"
        if path.is_file() and not self.fresh:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("pop") == self.pop and doc.get("gens") == self.gens and doc.get("seed") == GA_SEED:
                return doc["result"]
        t = time.time()
        res = anchored.evolve(self.runner, tuple(fit_specs), lam, GA_SEED, tag, self.pop, self.gens)
        out = {**res, "params": asdict(res["params"])}
        path.write_text(json.dumps({"pop": self.pop, "gens": self.gens, "seed": GA_SEED, "lam": lam, "scope": scope,
                                    "result": out}, indent=1) + "\n", encoding="utf-8")
        self.log(f"  GA lam={lam:g} {scope}: obj on fit set {out['obj_fit_set']:.2f} d={out['d']:.3f} "
                 f"({time.time() - t:.0f}s, {out['evaluated']} evaluated)")
        return out

    def run_c1(self) -> dict[float, dict]:
        """GA per (lam, fold) and per lam on full TRAIN, all concurrently (threads submit to the shared pool)."""
        tasks = []
        for lam in self.lam_grid:
            for i, fold in enumerate(self.folds):
                tasks.append((lam, fold.tag, fold.fit, 100 + i))
            tasks.append((lam, "full", self.full, 200))
        with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as tp:
            results = list(tp.map(lambda t: self.ga(t[0], t[1], t[2], t[3]), tasks))
        by = {}
        for (lam, scope, _, _), res in zip(tasks, results):
            by.setdefault(lam, {})[scope] = res
        return by

    def score_c1(self, ga_by_lam: dict[float, dict]) -> list[dict]:
        cands = []
        names = {lam: f"C1-lam{lam:g}" for lam in self.lam_grid}
        fm = self.fold_metrics({names[lam]: [{"kind": "swarm", "params": ga_by_lam[lam][f.tag]["params"], "learn": True}
                                             for f in self.folds] for lam in self.lam_grid})
        full = self.full_metrics({names[lam]: {"kind": "swarm", "params": ga_by_lam[lam]["full"]["params"], "learn": True}
                                  for lam in self.lam_grid})
        for lam in self.lam_grid:
            n = names[lam]
            cands.append({"id": n, "mechanism": "C1", "hparams": {"lam": lam}, "folds": fm[n], **self.aggregate(fm[n]),
                          "full_train": full[n], "full_train_params": ga_by_lam[lam]["full"]["params"],
                          "full_train_layer": None, "d_full_train": r(ga_by_lam[lam]["full"]["d"]),
                          "fold_d": {f.tag: r(ga_by_lam[lam][f.tag]["d"]) for f in self.folds}})
        return cands

    # ------------------------------------------------------------ C2 / C3
    def honest_stats(self, specs) -> np.ndarray:
        x, y = self.runner.map([("stats", tuple(specs))])[0]
        return x[~y]

    def layers(self) -> dict[tuple, dict]:
        """layer spec per (beta pair, scope) with scope = fold tag or "full"; honest FIT events only."""
        hs = {f.tag: self.honest_stats(f.fit) for f in self.folds}
        hs["full"] = self.honest_stats(self.full)
        return {(bp, scope): calibrate(h, *bp) for bp in self.beta_grid for scope, h in hs.items()}

    def score_layered(self, mech: str, layers: dict, base_params_by_scope: dict[str, dict], tag_fn) -> list[dict]:
        dspecs, full_specs = {}, {}
        for bp in self.beta_grid:
            n = tag_fn(bp)
            dspecs[n] = [{"kind": "r2", "params": base_params_by_scope[f.tag], "layer": layers[(bp, f.tag)]}
                         for f in self.folds]
            full_specs[n] = {"kind": "r2", "params": base_params_by_scope["full"], "layer": layers[(bp, "full")]}
        fm = self.fold_metrics(dspecs)
        full = self.full_metrics(full_specs)
        out = []
        for bp in self.beta_grid:
            n = tag_fn(bp)
            out.append({"id": n, "mechanism": mech, "hparams": {"beta_h": bp[0], "beta_s": bp[1]}, "folds": fm[n],
                        **self.aggregate(fm[n]), "full_train": full[n],
                        "full_train_params": base_params_by_scope["full"], "full_train_layer": layers[(bp, "full")],
                        "fold_layers": {f.tag: {k: layers[(bp, f.tag)][k] for k in
                                                ("honest_union_rate_hold", "honest_union_rate_step", "n_honest_events")}
                                        for f in self.folds}})
        return out

    # ------------------------------------------------------------ rule
    @staticmethod
    def apply_rule(cands: list[dict], refs: dict) -> None:
        static, single, l1 = refs["static_rules"], refs["single_policy"], refs["swarm_l1"]
        floor = min(static["lofo_score"], single["lofo_score"]) - MARGIN
        for c in cands:
            block_ok = all(c["directions"][d]["honest_block_rate"]
                           <= static["directions"][d]["honest_block_rate"] - BLOCK_MARGIN for d in static["directions"])
            c["eligibility"] = {
                "1_block_le_static_minus_0.75_both_directions": bool(block_ok),
                "2_lofo_score_le_best_reference_minus_3": bool(c["lofo_score"] <= floor),
                "3_full_train_le_default_swarm_plus_3": bool(c["full_train"]["objective"]
                                                             <= l1["full_train"]["objective"] + TRAIN_SLACK),
            }
            c["eligible"] = all(c["eligibility"].values())
            c["meets_block_only"] = bool(block_ok)

    @staticmethod
    def pick(cands: list[dict]) -> dict | None:
        """Rule section 4: acceptable set within BAND of the best eligible LOFO score, then the most sensitive one."""
        pool = [c for c in cands if c["eligible"]]
        flag = "eligible"
        if not pool:
            pool = [c for c in cands if c["meets_block_only"]]
            flag = "NONE ELIGIBLE: lowest LOFO score among candidates meeting the block condition"
        if not pool:
            return None
        best = min(c["lofo_score"] for c in pool)
        strict = min(pool, key=lambda c: (c["lofo_score"], c["worst_direction"], SIMPLICITY[c["mechanism"]], c["id"]))
        if flag != "eligible":
            return {"candidate": strict, "basis": flag, "acceptable": [strict["id"]], "strict_min": strict["id"]}
        acceptable = [c for c in pool if c["lofo_score"] <= best + BAND]
        top_block = max(c["lofo_honest_block_rate"] for c in acceptable)
        tied = [c for c in acceptable if top_block - c["lofo_honest_block_rate"] < BLOCK_TIE]
        win = min(tied, key=lambda c: (c["lofo_score"], c["worst_direction"], SIMPLICITY[c["mechanism"]], c["id"]))
        return {"candidate": win, "basis": flag, "acceptable": sorted(c["id"] for c in acceptable),
                "strict_min": strict["id"]}

    # ------------------------------------------------------------ main
    def run(self) -> dict:
        t0 = time.time()
        self.log("== references (same LOFO episodes)")
        refs = self.references()

        self.log("== C1 prior-anchored evolution")
        ga_by_lam = self.run_c1()
        c1 = self.score_c1(ga_by_lam)
        self.apply_rule(c1, refs)
        for c in c1:
            self.log(f"  {c['id']:12s} LOFO {c['lofo_score']:7.2f} worst {c['worst_direction']:7.2f} "
                     f"block {c['lofo_honest_block_rate']:5.2f} full-TRAIN {c['full_train']['objective']:6.2f} "
                     f"eligible={c['eligible']}")

        self.log("== C2 honest-calibrated anomaly union layer on the default swarm")
        layers = self.layers()
        dflt = asdict(default_params())
        c2 = self.score_layered("C2", layers, {**{f.tag: dflt for f in self.folds}, "full": dflt},
                                lambda bp: f"C2-bh{bp[0] * 100:g}-bs{bp[1] * 100:g}")
        self.apply_rule(c2, refs)
        for c in c2:
            self.log(f"  {c['id']:16s} LOFO {c['lofo_score']:7.2f} worst {c['worst_direction']:7.2f} "
                     f"block {c['lofo_honest_block_rate']:5.2f} full-TRAIN {c['full_train']['objective']:6.2f} "
                     f"eligible={c['eligible']}")

        c1_pick, c2_pick = self.pick(c1), self.pick(c2)
        c3: list[dict] = []
        c3_skip = None
        if c1_pick and c1_pick["basis"] == "eligible" and c2_pick and c2_pick["basis"] == "eligible":
            lam = c1_pick["candidate"]["hparams"]["lam"]
            self.log(f"== C3 stack: C1 lam={lam:g} + C2 layer")
            base = {f.tag: ga_by_lam[lam][f.tag]["params"] for f in self.folds}
            base["full"] = ga_by_lam[lam]["full"]["params"]
            c3 = self.score_layered("C3", layers, base,
                                    lambda bp: f"C3-lam{lam:g}-bh{bp[0] * 100:g}-bs{bp[1] * 100:g}")
            self.apply_rule(c3, refs)
            for c in c3:
                self.log(f"  {c['id']:22s} LOFO {c['lofo_score']:7.2f} worst {c['worst_direction']:7.2f} "
                         f"block {c['lofo_honest_block_rate']:5.2f} full-TRAIN {c['full_train']['objective']:6.2f} "
                         f"eligible={c['eligible']}")
        else:
            c3_skip = "C3 not run: C1 and C2 did not each produce an eligible candidate (rule section 3)"
            self.log("== " + c3_skip)

        cands = c1 + c2 + c3
        champ = self.pick(cands)
        self.log(f"== champion: {champ['candidate']['id'] if champ else None}  [{champ['basis'] if champ else ''}]")

        doc = {
            "rule": "eval/r2/SELECTION_RULE.md",
            "generated_by": "python -m scripts.r2_select",
            "git_rev": git_rev(),
            "settings": {"ga_pop": self.pop, "ga_gens": self.gens, "ga_seed": GA_SEED, "lam_grid": list(self.lam_grid),
                         "beta_grid": [list(b) for b in self.beta_grid], "margin": MARGIN, "train_slack": TRAIN_SLACK,
                         "block_margin": BLOCK_MARGIN, "band": BAND, "block_tie": BLOCK_TIE},
            "protocol": {"halves": "H0 = seeds 1000-1009, H1 = seeds 1010-1019",
                         "fold": "fit on family A / one half, evaluate on family B / the other half",
                         "folds": [{"tag": f.tag, "fit_seeds": [s[1] for s in f.fit], "fit_family": f.fit[0][2][0],
                                    "eval_seeds": [s[1] for s in f.evaluate], "eval_family": f.evaluate[0][2][0]}
                                   for f in self.folds]},
            "references": refs,
            "candidates": cands,
            "c3_skipped": c3_skip,
            "champion": None if champ is None else {"id": champ["candidate"]["id"], "basis": champ["basis"],
                                                    "acceptable_set": champ["acceptable"],
                                                    "strict_min_pick": champ["strict_min"],
                                                    "lofo_score": champ["candidate"]["lofo_score"],
                                                    "full_train": champ["candidate"]["full_train"]},
            "seconds": round(time.time() - t0, 1),
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "selection.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        if champ is not None:
            win = champ["candidate"]
            cj = {"id": "swarm_r2", "selected_candidate": win["id"], "mechanism": win["mechanism"],
                  "hparams": win["hparams"], "params": win["full_train_params"], "layer": win["full_train_layer"],
                  "fit": "all TRAIN (seeds 1000-1019, ring + structuring), GA seed 0, same procedure as the LOFO folds"}
            raw = json.dumps(cj, indent=1, sort_keys=True) + "\n"
            (self.out_dir / "champion.json").write_bytes(raw.encode("utf-8"))  # LF on every OS: the printed sha256 is the file's
            self.log(f"champion.json sha256 {hashlib.sha256(raw.encode('utf-8')).hexdigest()}")
        self.log(f"done in {(time.time() - t0) / 60:.1f} min")
        return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 4) - 4)))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        out, cache = R2_DIR / "quick", R2_DIR / "quick" / "cache"
        sel_args = dict(pop=4, gens=2, lam_grid=(20.0,), beta_grid=BETA_GRID[:1])
    else:
        out, cache = R2_DIR, R2_DIR / "cache"
        sel_args = dict(pop=anchored.POP, gens=anchored.GENS, lam_grid=LAM_GRID, beta_grid=BETA_GRID)
    with Runner(args.workers) as runner:
        Selector(runner, out, cache, fresh=args.fresh, **sel_args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
