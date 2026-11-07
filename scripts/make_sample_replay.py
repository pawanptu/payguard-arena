"""Build the SAMPLE replay fixture used by the dashboard and its tests.

Run from the repo root:  python -m scripts.make_sample_replay [--out PATH]

Honest parts: `results` and `episode` come from the REAL simulator on ONE TRAIN seed
(protocol.TRAIN_SEEDS[0], TRAIN_FAMILIES) with the trivial bounds and StaticRules. Only the TRAIN
split is run here; TEST_SEEN / TEST_HELDOUT are never touched.
Fake parts: `evolution` and `payouts` are PLACEHOLDERS (names and ids say so). The file is marked
"synthetic_fixture": true so the UI shows a "sample data" badge. It is NOT an evolution result.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from payguard.api.replay import OBJECTIVE_FORMULA, SCHEMA_VERSION, parse_replay
from payguard.baselines.static_rules import StaticRules
from payguard.baselines.trivial import AlwaysHold, AlwaysRelease
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from payguard.sim.metrics import evaluate
from payguard.sim.runner import run_episode

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "payguard" / "api" / "fixtures" / "sample_replay.json"
SPLIT = "TRAIN"
SEED = TRAIN_SEEDS[0]


def objective(leak: float, block: float, unpaid: float) -> float:
    return leak + 0.5 * block + 1.0 * unpaid


def build(generated_at: str) -> dict:
    ep = episode_for(SEED, TRAIN_FAMILIES)
    results, shown = [], None
    for make in (AlwaysRelease, AlwaysHold, StaticRules):
        run = run_episode(ep, make())
        m = evaluate([run])
        results.append({
            "defender": make.name, "split": SPLIT,
            "leakage_pct": m.leakage_pct, "honest_block_rate": m.honest_block_rate,
            "honest_unpaid_rate": m.honest_unpaid_rate, "honest_median_ttp_h": m.honest_median_ttp_h,
            "immunity_attempts": m.immunity_attempts, "never_contained_rate": m.never_contained_rate,
            "objective": objective(m.leakage_pct, m.honest_block_rate, m.honest_unpaid_rate),
        })
        if make is StaticRules:
            shown = run
    events = []
    for e in ep.events:
        oc, tr = shown.outcomes[e.id], ep.truth[e.id]
        events.append({
            "id": e.id, "t": round(e.t, 3), "worker_id": e.worker_id, "requester_id": e.requester_id,
            "amount": round(e.amount, 2), "verdict": shown.verdicts[e.id].value, "paid": oc.paid,
            "paid_at": None if oc.paid_at is None else round(oc.paid_at, 3),
            "is_attack": tr.is_attack, "family": tr.family, "signals": None, "weights": None,
        })

    # --- PLACEHOLDERS below: not produced by any evolution run ---
    start = next(r["objective"] for r in results if r["defender"] == "static_rules")
    curve = [start, start * 0.93, start * 0.93, start * 0.82, start * 0.74, start * 0.74, start * 0.68, start * 0.61]
    evolution, champ = [], curve[0]
    for g, challenger in enumerate(curve):
        promoted = g > 0 and challenger < champ
        champ = challenger if promoted else champ
        evolution.append({
            "generation": g, "champion_objective": champ, "challenger_objective": challenger + (0.0 if promoted else 1.5),
            "promoted": promoted, "attacker_family": "PLACEHOLDER-family" if g else None,
            "attacker_leakage_pct": None if g == 0 else round(40.0 - 3.0 * g, 2),
            "weights": {"placeholder_agent_a": 1.0 + 0.1 * g, "placeholder_agent_b": 1.0 - 0.05 * g,
                        "placeholder_agent_c": 1.0},
        })
    paid_release = [x for x in events if x["verdict"] == "RELEASE" and x["paid"]][:3]
    payouts = []
    for i, x in enumerate(paid_release):
        live = i == 2
        payouts.append({
            "event_id": x["id"], "mode": "live" if live else "sim",
            "sender_batch_id": f"PLACEHOLDER-{'live' if live else 'sim'}-{i:04d}",
            "payout_batch_id": f"PLACEHOLDER-batch-{i:04d}",
            "batch_status": "SUCCESS" if live else "SIMULATED", "item_status": "SUCCESS" if live else "SIMULATED",
            "amount_usd": x["amount"], "created_at": generated_at,
        })
    return {
        "schema_version": SCHEMA_VERSION, "simulated": True, "synthetic_fixture": True,
        "generated_at": generated_at, "objective": OBJECTIVE_FORMULA, "results": results,
        "evolution": evolution,
        "episode": {"seed": SEED, "split": SPLIT, "defender": "static_rules", "events": events},
        "payouts": payouts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    data = build(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    parse_replay(data, "generated fixture")  # never write a file that breaks the contract
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes, {len(data['episode']['events']):,} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
