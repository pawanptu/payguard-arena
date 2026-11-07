"""Freeze (or verify) the evaluation benchmark.

    python -m scripts.freeze --write   record hashes in eval/FROZEN.json (do this ONCE, after the human
                                       finalizes the static baseline)
    python -m scripts.freeze           verify the files still match; exit 1 and list what changed if not

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
FROZEN_FILES = [
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "payguard" / "sim").glob("*.py")),
    "payguard/protocol.py",
    "payguard/baselines/static_rules.py",
    "payguard/baselines/trivial.py",
]


def digest(rel: str) -> str:
    data = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")  # same hash on Windows and Linux
    return hashlib.sha256(data).hexdigest()


def current() -> dict[str, str]:
    return {rel: digest(rel) for rel in FROZEN_FILES}


def drift(frozen: dict[str, str]) -> list[str]:
    now = current()
    changed = [f"changed: {rel}" for rel in now if rel in frozen and now[rel] != frozen[rel]]
    added = [f"not in freeze: {rel}" for rel in now if rel not in frozen]
    removed = [f"missing: {rel}" for rel in frozen if rel not in now]
    return changed + added + removed


def main(argv: list[str]) -> int:
    if "--write" in argv:
        FROZEN_PATH.parent.mkdir(exist_ok=True)
        FROZEN_PATH.write_text(json.dumps(current(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"froze {len(FROZEN_FILES)} files -> {FROZEN_PATH.relative_to(ROOT).as_posix()}")
        return 0
    if not FROZEN_PATH.exists():
        print("not frozen yet (run with --write once the static baseline is final)")
        return 0
    problems = drift(json.loads(FROZEN_PATH.read_text(encoding="utf-8")))
    for line in problems:
        print(line)
    print("FROZEN files match" if not problems else f"{len(problems)} problem(s): the benchmark moved")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
