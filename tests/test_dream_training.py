"""Round-3 training wrapper (P5): `payguard.dream.training` refuses held-out families, forces the training families into
the DreamConfig, returns the champion and the calibration gap series, and is deterministic whatever the worker count.
TRAIN seeds only."""
import json
from dataclasses import replace

import pytest

from payguard.dream.contract import DreamConfig
from payguard.evolve.loop import EvolveConfig
from payguard.evolve.splits import TRAIN_A, TRAIN_B
from payguard.protocol import TRAIN_FAMILIES
from payguard.swarm.params import SwarmParams, default_params

SMALL = EvolveConfig(seed=1, pop=4, gens=3, elites=1, tournament_k=2, pool_size=1, fresh_attackers=1,
                     attacker_seeds=1, arena_episodes=1, fitness_seeds=TRAIN_A[:2], gate_seeds=TRAIN_B[:2], name="small")


# --- wrapper tests --------------------------------------------------------------------------------------

def test_train_refuses_held_out_families_and_bad_seeds():
    from payguard.dream.training import train_run
    with pytest.raises(ValueError):
        train_run(DreamConfig(n_dreams=2), ("sleeper",), 0, evolve_cfg=SMALL, workers=1)
    with pytest.raises(ValueError):
        train_run(DreamConfig(n_dreams=2), ("ring", "ring"), 0, evolve_cfg=SMALL, workers=1)


def test_train_config_forces_the_training_families_into_the_dream_config():
    from payguard.dream import training
    captured = {}

    class Stop(Exception):
        pass

    def fake_run(ecfg, ev, log=None, hooks=None, loop_cfg=None):
        captured["ecfg"], captured["hooks"] = ecfg, hooks
        raise Stop

    import payguard.dream.training as tr
    orig = tr.run_evolution
    tr.run_evolution = fake_run
    try:
        with pytest.raises(Stop):
            tr.train_run(DreamConfig(n_dreams=2, families=("ring", "structuring")), ("ring",), 4,
                         evolve_cfg=SMALL, workers=1)
    finally:
        tr.run_evolution = orig
    assert captured["ecfg"].families == ("ring",) and captured["ecfg"].seed == 4
    assert captured["hooks"].cfg.families == ("ring",)
    assert captured["hooks"].base_seed == 4


def test_train_end_to_end_on_the_quick_config_is_fast_and_returns_params():
    import time
    from payguard.dream.training import train_run
    t = time.time()
    out = train_run(DreamConfig(n_dreams=4), TRAIN_FAMILIES, 0, evolve_cfg=EvolveConfig.quick(0), workers=4)
    assert time.time() - t < 60
    assert len(out["calibration_gap_ema"]) == out["generations_run"] == EvolveConfig.quick(0).gens
    assert all(isinstance(x, float) for x in out["calibration_gap_ema"])
    assert isinstance(out["champion_params"], SwarmParams)
    assert all("dream" in e for e in out["lineage"])
    assert set(out["lineage"][0]["strongest_attacker"]) == set(TRAIN_FAMILIES)


def test_train_is_deterministic_whatever_the_worker_count():
    from payguard.dream.training import train_run
    tiny = replace(SMALL, gens=2, pop=3, fitness_seeds=TRAIN_A[:1], gate_seeds=TRAIN_B[:1], arena_episodes=1)
    a = train_run(DreamConfig(n_dreams=2), TRAIN_FAMILIES, 0, evolve_cfg=tiny, workers=1)
    b = train_run(DreamConfig(n_dreams=2), TRAIN_FAMILIES, 0, evolve_cfg=tiny, workers=2)
    assert a["champion"] == b["champion"]
    def no_timing(lin):  # fit_seconds and gen_seconds are wall-clock; everything else must match exactly
        out = []
        for e in lin:
            e = {k: v for k, v in e.items()}
            e["dream"] = {k: v for k, v in e["dream"].items() if k != "fit_seconds"}
            out.append(e)
        return json.dumps(out, sort_keys=True, default=str)

    assert no_timing(a["lineage"]) == no_timing(b["lineage"])
    assert a["calibration_gap_ema"] == b["calibration_gap_ema"]


def test_train_returns_the_champion_params_from_the_module_config(monkeypatch):
    from payguard.dream import training
    monkeypatch.setattr(training, "EVOLVE_CONFIG", replace(SMALL, gens=1, pop=3))
    p = training.train(DreamConfig(n_dreams=2), ("ring",), 0)
    assert isinstance(p, SwarmParams) and p == default_params() or isinstance(p, SwarmParams)
