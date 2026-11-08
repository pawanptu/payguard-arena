"""Round-3 sealed family: registration, well-formedness, bounds and freezes.

Tripwire: NOTHING here may build a TEST_HELDOUT3 episode (seeds 5000-5019). An autouse fixture makes every episode
builder raise on those seeds, and a source scan keeps the seeds out of the other test files.
"""
import ast
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import payguard.heldout3 as h3
import payguard.protocol as round1_protocol
import payguard.sim.engine as engine
from payguard.heldout3.protocol import HELDOUT3_FAMILIES, TEST_HELDOUT3_SEEDS, episode_for
from payguard.sim import attacks, metrics
from payguard.sim.attacks import BOUNDS, GENERATORS, INT_FIELDS, PARAMS, REVIEW_THRESHOLD_USD, mutate, sample_params
from payguard.sim.engine import build_episode
from payguard.sim.types import Verdict

DEV_SEEDS = tuple(range(9100, 9105))
SEALED = frozenset(range(5000, 5020))
FAM = HELDOUT3_FAMILIES[0]
TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def tripwire(monkeypatch):
    real = engine.build_episode

    def guarded(seed, *a, **kw):
        if seed in SEALED:
            raise AssertionError(f"tripwire: a test tried to build sealed TEST_HELDOUT3 seed {seed}")
        return real(seed, *a, **kw)

    monkeypatch.setattr(engine, "build_episode", guarded)
    monkeypatch.setattr(round1_protocol, "build_episode", guarded)


def test_tripwire_fires_on_sealed_seeds():
    for s in (min(SEALED), max(SEALED)):
        with pytest.raises(AssertionError, match="tripwire"):
            episode_for(s)
        with pytest.raises(AssertionError, match="tripwire"):
            round1_protocol.episode_for(s, HELDOUT3_FAMILIES)


def test_no_other_test_file_touches_sealed_seeds():
    for path in TESTS_DIR.glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        src = path.read_text(encoding="utf-8")
        assert "TEST_HELDOUT3" not in src and "heldout3" not in src.lower(), path.name
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                assert node.value not in SEALED, f"{path.name}: literal {node.value}"
            if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "range" and len(node.args) == 2
                    and all(isinstance(a, ast.Constant) for a in node.args)):
                lo, hi = (a.value for a in node.args)
                assert not (lo < 5020 and hi > 5000), f"{path.name}: range({lo}, {hi}) overlaps sealed seeds"


def test_protocol_constants():
    assert TEST_HELDOUT3_SEEDS == tuple(range(5000, 5020))
    assert set(TEST_HELDOUT3_SEEDS) == SEALED
    assert len(HELDOUT3_FAMILIES) == 1 and HELDOUT3_FAMILIES[0] == h3.NAME
    others = set(round1_protocol.TRAIN_SEEDS + round1_protocol.TEST_SEEN_SEEDS + round1_protocol.TEST_HELDOUT_SEEDS)
    assert not SEALED & others


def test_family_registers():
    assert GENERATORS[FAM] is h3.gen_h3 and PARAMS[FAM] is h3.H3Params and BOUNDS[FAM] is h3.BOUNDS_H3
    assert attacks.GENERATORS is engine.GENERATORS  # same dict object, so build_episode sees the registration
    assert {f.name for f in fields(h3.H3Params)} == set(h3.BOUNDS_H3)
    assert h3.INT_FIELDS_H3 <= INT_FIELDS and h3.INT_FIELDS_H3 <= set(h3.BOUNDS_H3)
    assert engine.STEPUP_PASS[FAM] == h3.STEPUP_PASS_PROB
    h3.register()  # idempotent
    assert list(GENERATORS).count(FAM) == 1 and list(metrics.FAMILIES).count(FAM) == 1
    for f in ("ring", "structuring", "sleeper"):
        assert f in GENERATORS and f in PARAMS and f in BOUNDS


def test_default_params_inside_bounds():
    p = h3.H3Params()
    for name, (lo, hi) in h3.BOUNDS_H3.items():
        assert lo <= getattr(p, name) <= hi, name


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_generated_episodes_are_well_formed(seed):
    ep = episode_for(seed)
    assert ep.seed == seed and ep.events
    ids = [e.id for e in ep.events]
    assert ids == list(range(len(ids)))
    ts = [e.t for e in ep.events]
    assert ts == sorted(ts) and all(0.0 <= t < ep.hours for t in ts)
    assert all(e.amount > 0 for e in ep.events)
    assert set(ep.truth) == set(ids)
    assert any(ep.truth[i].is_attack for i in ids), "episode has no attack payouts"
    for e in ep.events:
        tr = ep.truth[e.id]
        if tr.is_attack:
            assert tr.family == FAM and tr.campaign_id >= 0
            assert 5.0 <= e.amount < REVIEW_THRESHOLD_USD  # attackers know the review threshold
        else:
            assert tr.family == "honest" and tr.campaign_id == -1
        assert ep.accounts[e.worker_id] <= e.t  # account exists before its payouts
    assert {ep.truth[i].family for i in ids} == {"honest", FAM}


@pytest.mark.parametrize("seed", DEV_SEEDS)
def test_episode_is_deterministic(seed):
    a, b = episode_for(seed), episode_for(seed)
    assert a.events == b.events and a.truth == b.truth and a.accounts == b.accounts


def test_labels_only_on_losses_and_every_campaign_has_attack_payouts():
    for seed in DEV_SEEDS:
        ep = episode_for(seed)
        per_campaign: dict[int, int] = {}
        for e in ep.events:
            tr = ep.truth[e.id]
            if tr.family == FAM:
                assert tr.is_attack
                per_campaign[tr.campaign_id] = per_campaign.get(tr.campaign_id, 0) + 1
            else:
                assert not tr.is_attack
        assert per_campaign and all(n > 0 for n in per_campaign.values())


def test_sample_params_and_mutate_stay_in_bounds():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        p = sample_params(FAM, rng)
        assert isinstance(p, h3.H3Params)
        for _ in range(3):
            for q in (p, mutate(p, FAM, rng), mutate(p, FAM, rng, scale=0.6)):
                for name, (lo, hi) in h3.BOUNDS_H3.items():
                    v = getattr(q, name)
                    assert lo <= v <= hi, name
                    assert isinstance(v, int) == (name in INT_FIELDS), name
            p = mutate(p, FAM, rng)


def test_extreme_params_still_generate_valid_events():
    for pick in (0, 1):
        p = h3.H3Params(**{n: (int(b[pick]) if n in INT_FIELDS else float(b[pick]))
                           for n, b in h3.BOUNDS_H3.items()})
        ep = build_episode(9150 + pick, [(FAM, p)])
        assert all(0.0 <= e.t < ep.hours and e.amount > 0 for e in ep.events)
        assert all(e.amount < REVIEW_THRESHOLD_USD for e in ep.events if ep.truth[e.id].is_attack)


def test_round1_episodes_unchanged_by_registration():
    ep = round1_protocol.episode_for(9100, round1_protocol.TRAIN_FAMILIES)
    assert {t.family for t in ep.truth.values()} <= {"honest", "ring", "structuring"}


def test_every_verdict_resolves_for_the_new_family():
    """Engine resolution (not a defender): a STEP_UP / HOLD on this family's payouts must not KeyError."""
    ep = episode_for(9100)
    attack = next(e for e in ep.events if ep.truth[e.id].is_attack)
    for v in Verdict:
        engine.resolve(ep.seed, attack, ep.truth[attack.id], v)


def test_honest_traffic_unchanged_by_campaign():
    p = sample_params(FAM, np.random.default_rng(7))
    with_c = build_episode(9101, [(FAM, p)])
    without = build_episode(9101, [])
    key = lambda e: (e.t, e.worker_id, e.requester_id, e.device_id, e.destination_id, e.amount)  # noqa: E731
    honest = [key(e) for e in with_c.events if not with_c.truth[e.id].is_attack]
    assert honest == [key(e) for e in without.events]
    assert all(with_c.accounts[w] == t for w, t in without.accounts.items())


def test_step_up_resolution_is_seeded():
    ep = episode_for(9102)
    atk = [e for e in ep.events if ep.truth[e.id].is_attack]
    ok = [engine.resolve(ep.seed, e, ep.truth[e.id], Verdict.STEP_UP).paid for e in atk]
    again = [engine.resolve(ep.seed, e, ep.truth[e.id], Verdict.STEP_UP).paid for e in atk]
    assert ok == again and any(ok) and not all(ok)
