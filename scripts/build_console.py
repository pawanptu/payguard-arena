"""Build the ops console review queue: eval/console/queue.json (docs/CONSOLE_CONTRACT.md, section A).

  python -m scripts.build_console            re-play the published replay episode (TEST_HELDOUT2, seed 4000) with
                                             the FROZEN swarm_r2, check every verdict against eval/replay_live.json
                                             and write the queue. Offline and deterministic (apart from generated_at).
  python -m scripts.build_console --check    same, but write nothing (prints the counts).

On any verdict mismatch nothing is written. No sealed number is recomputed or tuned: this only re-plays the one
episode that is already published in the replay, with the defender that was frozen before round 2 (D-034).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from payguard.console.queue import ParityError, assemble_queue, record_episode
from scripts import final_eval_r2 as fe2

ROOT = fe2.ROOT
REPLAY_PATH = ROOT / "eval" / "replay_live.json"
OUT_PATH = ROOT / "eval" / "console" / "queue.json"


def build(replay_path: Path = REPLAY_PATH) -> dict:
    fe2.require_frozen()  # raises SystemExit when swarm_r2 moved after its freeze
    split, seed = fe2.EPISODE
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    ep_meta = replay["episode"]
    if (ep_meta["split"], ep_meta["seed"], ep_meta["defender"]) != (split, seed, fe2.PRIMARY):
        raise SystemExit(f"{replay_path} holds {ep_meta['split']} seed {ep_meta['seed']} ({ep_meta['defender']}), "
                         f"expected {split} seed {seed} ({fe2.PRIMARY}); nothing written")
    defender = fe2.make_r2()
    p = defender.params
    episode = fe2.sealed_episode_for(seed)
    result, at = record_episode(episode, defender)
    source = {"split": split, "seed": seed, "defender": fe2.PRIMARY,
              "replay": replay_path.relative_to(ROOT).as_posix()}
    return assemble_queue(episode, result, at, replay,
                          thresholds={"step_up": p.step_up_threshold, "hold": p.hold_threshold}, source=source)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="build and verify, write nothing")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help=f"output file (default {OUT_PATH.relative_to(ROOT)})")
    args = ap.parse_args(argv)
    t0 = time.time()
    try:
        doc = build()
    except ParityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    c = doc["counts"]
    print(f"parity ok on {doc['source']['parity_checked_events']} events; cases={c['cases']} "
          f"(HOLD {c['HOLD']}, STEP_UP {c['STEP_UP']}; attack {c['cases_attack']}, honest {c['cases_honest']}); "
          f"attack USD total {c['attack_usd_total']:,.2f} stopped {c['attack_usd_stopped']:,.2f} "
          f"leaked {c['attack_usd_leaked']:,.2f}")
    if args.check:
        print(f"--check: nothing written ({time.time() - t0:.1f}s)")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, allow_nan=False) + "\n"
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:  # LF on every OS: same bytes everywhere
        fh.write(text)
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
