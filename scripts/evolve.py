"""Generation-level evolution (L2) of the defender swarm, with attacker co-evolution. TRAIN split only.

    python -m scripts.evolve                 full run (seed 0): plan is printed first, target <= 20 min, hard stop 28 min
    python -m scripts.evolve --quick         tiny run (population 4, 2 generations, 2 episodes): < 60 s
    python -m scripts.evolve --plan-only     time one evaluation, print the plan, exit

Outputs (default): eval/evolution/{champion,lineage,candidates,analysis,run_meta}.json and eval/replay.json.
All numbers are SIMULATED replays on synthetic data.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from payguard.evolve import replay_export as rx
from payguard.evolve.analysis import final_analysis
from payguard.evolve.evaluation import Evaluator, proto_spec, swarm_job
from payguard.evolve.loop import EvolveConfig, make_plan, run_evolution
from payguard.swarm.params import SwarmParams, default_params

ROOT = Path(__file__).resolve().parent.parent
MAX_WORKERS = 10


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def measure_episode_seconds(ev: Evaluator, cfg: EvolveConfig, workers: int) -> float:
    """Seconds per episode when all workers run at once (warm-up batch first so process start-up is excluded)."""
    sp = [proto_spec(s) for s in cfg.fitness_seeds[:2]]
    d = default_params()
    ev.run([swarm_job(d, sp[:1]) for _ in range(workers)])
    t = time.time()
    ev.run([swarm_job(d, sp) for _ in range(workers)])
    return (time.time() - t) / len(sp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help=f"processes (default min(cpu-6, {MAX_WORKERS}), at most {MAX_WORKERS})")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None, help="default: sized from a timed evaluation to fit --target-min")
    ap.add_argument("--target-min", type=float, default=20.0, help="planning target in minutes (hard stop: 28)")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "evolution")
    ap.add_argument("--replay", type=Path, default=ROOT / "eval" / "replay.json")
    args = ap.parse_args(argv)

    cfg = EvolveConfig.quick(args.seed) if args.quick else EvolveConfig(seed=args.seed)
    cpu = os.cpu_count() or 4
    workers = max(1, min(args.workers or (min(cpu, 6) if args.quick else min(cpu - 6, MAX_WORKERS)), MAX_WORKERS))
    if args.pop:
        cfg = replace(cfg, pop=args.pop)
    elif not args.quick:
        cfg = replace(cfg, pop=3 * workers - 2)  # population + champion's gate job fill whole rounds of the pool
    preload = tuple(cfg.fitness_seeds) + tuple(cfg.gate_seeds)

    t_start = time.time()
    with Evaluator(workers, preload) as ev:
        t_ep = measure_episode_seconds(ev, cfg, workers)
        plan = make_plan(replace(cfg, gens=args.gens if args.gens is not None else (cfg.gens if args.quick else 0)),
                         workers, t_ep, budget_min=args.target_min)
        cfg = replace(cfg, gens=plan["gens"])
        print(f"PLAN [{cfg.name}] seed={cfg.seed}: population={plan['pop']} generations={plan['gens']} "
              f"workers={plan['workers']} episode={plan['t_episode_s']}s/episode "
              f"-> ~{plan['est_seconds_per_generation']}s/generation + ~{plan['est_post_seconds']}s analysis "
              f"= estimated {plan['estimated_minutes']} min (target {args.target_min:.0f}, hard stop {cfg.budget_min:.0f})",
              flush=True)
        if args.plan_only:
            return 0

        result = run_evolution(cfg, ev, log=lambda s: print(s, flush=True))
        lin = result["lineage"]
        print(f"evolution: {result['generations_run']} generations in {result['seconds'] / 60:.1f} min; "
              f"promotions at {[e['generation'] for e in lin if e['promoted']]}, "
              f"rollbacks at {[e['generation'] for e in lin if e['rolled_back']]}", flush=True)

        # --- evidence + replay (TRAIN only) ---
        champ = SwarmParams(**result["champion"]["params"])
        all_seeds = tuple(cfg.fitness_seeds) + tuple(cfg.gate_seeds)
        metrics_all = rx.train_metrics(ev, champ, all_seeds)
        analysis = final_analysis(cfg, ev, result, metrics_all)
        rows = rx.results_rows(metrics_all)
        replay = rx.build_replay(rows, lin, champ)
        rx.validate_replay(replay)
        size = rx.write_replay(args.replay, replay)

    tab = analysis["tables"]
    champ_prov = {k: result["champion"][k] for k in ("generation", "objective_a", "objective_b")}
    dump(args.out / "champion.json", {
        "id": result["champion"]["id"], "params": result["champion"]["params"],
        "provenance": {**champ_prov, "protocol_train_a_objective": tab["TRAIN_A"]["swarm_evolved"]["objective"],
                       "protocol_train_b_objective": tab["TRAIN_B"]["swarm_evolved"]["objective"],
                       "git_rev": git_rev(), "seed": cfg.seed, "mode": cfg.name,
                       "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}})
    dump(args.out / "lineage.json", lin)
    dump(args.out / "candidates.json", result["candidates"])
    dump(args.out / "analysis.json", analysis)
    dump(args.out / "run_meta.json", {
        "plan": plan, "workers": workers, "seconds_total": round(time.time() - t_start, 1),
        "seconds_evolution": round(result["seconds"], 1), "generations_run": result["generations_run"],
        "stopped_early": result["stopped_early"], "gen_seconds": [round(x, 1) for x in result["gen_seconds"]],
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}})

    print("\n== TRAIN protocol objectives (obj = leak + 0.5*block + 1.0*unpaid, lower is better)")
    for split in ("TRAIN_A", "TRAIN_B"):
        print(f" {split}:")
        for name, s in tab[split].items():
            print(f"   {name:26s} obj={s['objective']:6.2f}  leak {s['leakage_pct']:5.1f}  block {s['honest_block_rate']:4.1f}"
                  f"  unpaid {s['honest_unpaid_rate']:4.1f}")
    ae = analysis["attacker_evolution"]
    print(f" attacker evolution, leakage gain initial->final pool (pct points): vs default swarm_l1 "
          f"{ae['leakage_gain_vs_default_swarm_l1_pct_points']}, vs champion {ae['leakage_gain_vs_champion_pct_points']}")
    print(f" robustness (final pool, TRAIN_B): {analysis['robustness_final_pool_train_b']}")
    print(f" champion {result['champion']['id']} uses l1_abstain={analysis['champion']['uses_l1_abstain']}; "
          f"changed genes: {sorted(analysis['champion']['changed_genes'])}")
    print(f" replay: {args.replay} ({size / 1024:.0f} KiB, {len(replay['episode']['events'])} events, "
          f"honest_sample_every={replay['episode'].get('honest_sample_every', 1)})")
    print(f" total {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
