"""Freeze (or verify) the evaluation benchmark.

    python -m scripts.freeze --write   record hashes in eval/FROZEN.json (do this ONCE, after the human
                                       finalizes the static baseline)
    python -m scripts.freeze           verify the files still match; exit 1 and list what changed if not
    python -m scripts.freeze --round 2 [--write]
                                       same for the round-2 held-out package (payguard/heldout2/*) and
                                       eval/FROZEN_R2.json. Without --round the behaviour is the round-1 one.

The frozen set is everything that defines the benchmark: the simulator, the metrics, the protocol and the
pre-committed baselines. Changing any of it after the freeze means the swarm was tuned against a moving
target, so it must be a logged decision followed by a deliberate re-freeze, never a quiet edit.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN_PATH = ROOT / "eval" / "FROZEN.json"
FROZEN_R2_PATH = ROOT / "eval" / "FROZEN_R2.json"
FROZEN_FILES = [
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "payguard" / "sim").glob("*.py")),
    "payguard/protocol.py",
    "payguard/baselines/static_rules.py",
    "payguard/baselines/trivial.py",
]


def digest(rel: str) -> str:
    data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")  # same hash on Windows and Linux
    return hashlib.sha256(data).hexdigest()


def round2_files() -> list[str]:
    """Every file in payguard/heldout2/ (caches excluded)."""
    base = ROOT / "payguard" / "heldout2"
    return sorted(p.relative_to(ROOT).as_posix() for p in base.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def current(rnd: int = 1) -> dict[str, str]:
    return {rel: digest(rel) for rel in (FROZEN_FILES if rnd == 1 else round2_files())}


def drift(frozen: dict[str, str], rnd: int = 1) -> list[str]:
    now = current(rnd)
    changed = [f"changed: {rel}" for rel in now if rel in frozen and now[rel] != frozen[rel]]
    added = [f"not in freeze: {rel}" for rel in now if rel not in frozen]
    removed = [f"missing: {rel}" for rel in frozen if rel not in now]
    return changed + added + removed


def main(argv: list[str]) -> int:
    rnd = 1
    if "--round" in argv:
        rnd = int(argv[argv.index("--round") + 1])
        if rnd not in (1, 2):
            print("--round must be 1 or 2")
            return 2
    path = FROZEN_PATH if rnd == 1 else FROZEN_R2_PATH
    if "--write" in argv:
        path.parent.mkdir(exist_ok=True)
        now = current(rnd)
        path.write_text(json.dumps(now, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"froze {len(now)} files -> {path.relative_to(ROOT).as_posix()}")
        return 0
    if not path.exists():
        print("not frozen yet (run with --write once the static baseline is final)" if rnd == 1
              else "round 2 not frozen yet (python -m scripts.freeze --round 2 --write)")
        return 0 if rnd == 1 else 1
    problems = drift(json.loads(path.read_text(encoding="utf-8")), rnd)
    for line in problems:
        print(line)
    ok = "FROZEN files match" if rnd == 1 else "FROZEN_R2 files match"
    print(ok if not problems else f"{len(problems)} problem(s): the benchmark moved")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
