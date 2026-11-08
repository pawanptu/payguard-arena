"""Unattended driver for round-3 M3 + M4 (TRAIN only): the dev check (SELECTION_RULE sections 2-3), then, only if K1
passed, the freeze of the four dreamer arms (section 4). Both steps checkpoint, so re-running this after a crash
resumes where it stopped.

  python -m scripts.r3_run_m3m4

While it runs it asks Windows not to idle-sleep (SetThreadExecutionState, released automatically when the process
exits; no setting is changed). Full run logs go to eval/r3/runs/ (PAYGUARD_R3_RUNLOG).
"""
from __future__ import annotations

import json
import os
import sys
import time

from payguard import r3eval

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def stay_awake() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def main() -> int:
    from scripts import r3_dev_check, r3_freeze_arms
    os.environ.setdefault("PAYGUARD_R3_RUNLOG", str(r3eval.ROOT / "eval" / "r3" / "runs"))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] M3+M4 start; keep-awake {'on' if stay_awake() else 'unavailable'}",
          flush=True)
    if not r3_dev_check.OUT_PATH.exists():
        r3_dev_check.main(["--run"])
    dev = json.loads(r3_dev_check.OUT_PATH.read_text(encoding="utf-8"))
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] dev check: {dev['verdict']}", flush=True)
    if not dev["k1"]["pass"]:
        return 0
    rc = r3_freeze_arms.main(["--run"])
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] M3+M4 done (freeze rc={rc})", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
