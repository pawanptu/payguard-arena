"""Once eval/FROZEN.json exists, any edit to the benchmark files fails CI until it is re-frozen on purpose."""
import json

import pytest

from scripts.freeze import FROZEN_PATH, drift


def test_benchmark_files_match_the_freeze():
    if not FROZEN_PATH.exists():
        pytest.skip("benchmark not frozen yet (python -m scripts.freeze --write)")
    assert drift(json.loads(FROZEN_PATH.read_text(encoding="utf-8"))) == []
