"""Shared memory (baseline f): taint after fraud labels, decay, honest relief, abstention, parent L1 unchanged,
determinism, and no use of held-out data."""
import ast
import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pytest

import payguard.swarm.memory as memory_module
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for
from payguard.sim.features import PlatformLog
from payguard.sim.runner import run_episode
from payguard.sim.types import Event, Verdict
from payguard.swarm.coordinator import NAMES, Swarm, make_swarm
from payguard.swarm.memory import (
    BOUNDS,
    INT_FIELDS,
    EntityMemory,
    MemoryParams,
    MemorySwarm,
    default_memory_params,
    make_memory_swarm,
    mutate_memory_params,
    sample_memory_params,
)

ROOT = Path(__file__).resolve().parent.parent


def ev(i, t, worker=1, requester=500, device=None, destination=None, amount=50.0):
    return Event(i, t, worker, requester, 10_000 + worker if device is None else device,
                 20_000 + worker if destination is None else destination, amount)


# --- params ---

def test_bounds_cover_every_field_and_defaults_sit_inside():
    assert set(BOUNDS) == {f.name for f in dataclasses.fields(MemoryParams)}
    assert INT_FIELDS <= set(BOUNDS)
    p = default_memory_params()
    for name, (lo, hi) in BOUNDS.items():
        assert lo <= getattr(p, name) <= hi, name


def test_sample_and_mutate_stay_in_bounds_and_are_deterministic():
    a = sample_memory_params(np.random.default_rng(3))
    assert a == sample_memory_params(np.random.default_rng(3))
    m = a
    rng = np.random.default_rng(4)
    for _ in range(30):
        m = mutate_memory_params(m, rng, 0.5)
        for name, (lo, hi) in BOUNDS.items():
            assert lo <= getattr(m, name) <= hi


# --- memory bookkeeping ---

def test_attack_label_taints_all_four_entities_but_not_others():
    mem = EntityMemory(MemoryParams())
    e = ev(1, 10.0, worker=1, requester=500)
    assert mem.score(e) == (0.0, None)
    mem.observe(e, True)
    sibling_same_requester = ev(2, 12.0, worker=2, requester=500)  # other worker, own device and destination
    stranger = ev(3, 12.0, worker=3, requester=501)
    assert mem.score(sibling_same_requester)[1] == "requester"
    assert mem.score(sibling_same_requester)[0] > 0.0
    assert mem.score(ev(4, 12.0, worker=1, requester=777))[1] in {"worker", "device", "destination"}
    assert mem.score(stranger) == (0.0, None)
    for kind, key in [("worker", ev(5, 12.0, worker=1)), ("device", ev(6, 12.0, worker=9, device=10_001)),
                      ("destination", ev(7, 12.0, worker=9, destination=20_001)),
                      ("requester", ev(8, 12.0, worker=9, requester=500))]:
        assert mem.taint((kind, {"worker": key.worker_id, "device": key.device_id,
                                 "destination": key.destination_id, "requester": key.requester_id}[kind]), 12.0) > 0


def test_kinds_do_not_collide_when_ids_coincide_numerically():
    mem = EntityMemory(MemoryParams())
    mem.observe(Event(1, 0.0, 7, 8, 9, 10, 50.0), True)
    assert mem.score(Event(2, 1.0, 100, 7, 101, 102, 50.0)) == (0.0, None)   # requester 7 is not worker 7
    assert mem.score(Event(3, 1.0, 100, 101, 7, 102, 50.0)) == (0.0, None)   # device 7 is not worker 7
    assert mem.score(Event(4, 1.0, 7, 101, 102, 103, 50.0))[1] == "worker"


def test_taint_decays_with_the_half_life_on_simulated_time():
    p = MemoryParams(strength_requester=0.8, half_life_h=48.0, honest_relief=0.0)
    mem = EntityMemory(p)
    mem.observe(ev(1, 0.0, requester=500), True)
    key = ("requester", 500)
    t0 = mem.taint(key, 0.0)
    assert t0 == pytest.approx(0.8)
    assert mem.taint(key, 48.0) == pytest.approx(0.4)
    assert mem.taint(key, 96.0) == pytest.approx(0.2)
    assert mem.score(ev(2, 10_000.0, worker=2, requester=500))[0] == 0.0  # long gone: abstains


def test_a_late_label_is_aged_from_the_request_time_not_the_arrival_time():
    p = MemoryParams(strength_requester=0.8, half_life_h=48.0, honest_relief=0.0)
    mem = EntityMemory(p)
    mem.observe(ev(1, 0.0, requester=500), True)      # the label of a request made at t=0
    assert mem.taint(("requester", 500), 72.0) == pytest.approx(0.8 * 0.5 ** 1.5)


def test_repeated_attack_labels_accumulate_and_saturate_at_one():
    mem = EntityMemory(MemoryParams(strength_requester=0.3, half_life_h=1e6, honest_relief=0.0))
    scores = []
    for i in range(6):
        mem.observe(ev(i, float(i), worker=i, requester=500), True)
        scores.append(mem.taint(("requester", 500), float(i)))
    assert scores == sorted(scores)
    assert scores[-1] == pytest.approx(1.0)


def test_out_of_order_labels_are_handled():
    p = MemoryParams(strength_requester=0.5, half_life_h=10.0, honest_relief=0.0)
    a, b = EntityMemory(p), EntityMemory(p)
    a.observe(ev(1, 5.0, worker=1), True)
    a.observe(ev(2, 20.0, worker=2), True)
    b.observe(ev(2, 20.0, worker=2), True)
    b.observe(ev(1, 5.0, worker=1), True)
    assert a.taint(("requester", 500), 30.0) == pytest.approx(b.taint(("requester", 500), 30.0))


def test_honest_labels_never_taint():
    mem = EntityMemory(MemoryParams())
    for i in range(20):
        mem.observe(ev(i, float(i), worker=i % 3, requester=500), False)
    assert mem.attack == {}
    assert mem.score(ev(99, 30.0, worker=1, requester=500)) == (0.0, None)


def test_honest_history_weakens_a_requesters_taint_even_if_it_came_first():
    def requester_taint(n_honest):
        mem = EntityMemory(MemoryParams(strength_requester=0.8, honest_relief=2.0, half_life_h=1e6))
        for i in range(n_honest):
            mem.observe(ev(i, float(i), worker=100 + i, requester=500), False)
        mem.observe(ev(1000, 50.0, worker=1, requester=500), True)
        return mem.taint(("requester", 500), 60.0)

    t0, t5, t50 = requester_taint(0), requester_taint(5), requester_taint(50)
    assert t0 > t5 > t50 > 0.0
    assert t50 < 0.05


def test_honest_relief_zero_ignores_honest_history():
    mem = EntityMemory(MemoryParams(strength_requester=0.8, honest_relief=0.0, half_life_h=1e6))
    for i in range(50):
        mem.observe(ev(i, float(i), worker=100 + i, requester=500), False)
    mem.observe(ev(1000, 50.0, worker=1, requester=500), True)
    assert mem.taint(("requester", 500), 60.0) == pytest.approx(0.8, rel=1e-3)


def test_weight_zero_switches_a_kind_off():
    p = MemoryParams(weight_requester=0.0)
    mem = EntityMemory(p)
    mem.observe(ev(1, 0.0, worker=1, requester=500), True)
    assert mem.score(ev(2, 1.0, worker=2, requester=500)) == (0.0, None)


def test_score_is_clipped_to_unit_interval_and_takes_the_strongest_entity():
    p = MemoryParams(strength_worker=1.0, strength_requester=0.2, weight_worker=1.0, weight_requester=1.0)
    mem = EntityMemory(p)
    mem.observe(ev(1, 0.0, worker=1, requester=500), True)
    s, kind = mem.score(ev(2, 0.0, worker=1, requester=500))
    assert kind == "worker" and 0.0 < s <= 1.0


# --- swarm integration ---

def test_memory_abstains_with_no_labels_so_decisions_equal_the_plain_swarm():
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    log_a, log_b = PlatformLog(ep.accounts), PlatformLog(ep.accounts)
    plain, mem = make_swarm(), make_memory_swarm()
    for e in ep.events[:600]:  # decide only; no label is ever delivered
        assert mem.decide(e, log_b) == plain.decide(e, log_a)
        assert mem.last_signals["memory"] == 0.0
        assert mem.last_signals["risk"] == pytest.approx(plain.last_signals["risk"])
        log_a.add(e)
        log_b.add(e)


def test_attack_label_makes_a_sibling_payout_riskier_and_reports_the_match():
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    log = PlatformLog(ep.accounts)
    sw = make_memory_swarm(None, MemoryParams(strength_requester=1.0, weight_requester=1.0))
    first = ep.events[0]
    sw.decide(first, log)
    log.add(first)
    other = next(w for w in ep.accounts if w != first.worker_id)  # a different, known account
    sibling = Event(10**6, first.t + 1.0, other, first.requester_id, 9_999_991, 9_999_992, 50.0)
    sw.decide(sibling, log)
    before = sw.last_signals["risk"]
    sw.learn(first, True)
    sw.decide(Event(10**6 + 1, first.t + 1.0, other, first.requester_id, 9_999_991, 9_999_992, 50.0), log)
    after = sw.last_signals
    assert after["risk"] > before
    assert after["memory"] > 0.0 and after["memory_match"] == "requester"
    assert after["verdict"] in {Verdict.HOLD.value, Verdict.STEP_UP.value}


def test_parent_l1_weight_update_is_unchanged_by_memory():
    ep = episode_for(TRAIN_SEEDS[1], TRAIN_FAMILIES)
    plain, mem = make_swarm(), make_memory_swarm()
    # Drive both with the SAME verdict-independent script: decide every event, then label it later.
    log_a, log_b = PlatformLog(ep.accounts), PlatformLog(ep.accounts)
    batch = ep.events[:400]
    for e in batch:
        plain.decide(e, log_a)
        mem.decide(e, log_b)
        log_a.add(e)
        log_b.add(e)
    # Specialist scores are memory-free and identical (no label yet, memory abstains), so the pending
    # scores - and therefore the L1 update from the labels - must be identical too.
    for e in batch:
        truth = ep.truth[e.id].is_attack
        plain.learn(e, truth)
        mem.learn(e, truth)
        assert mem.weights == plain.weights
    assert set(mem.weights) == set(NAMES)  # memory is not a learned reputation voice


def test_memory_swarm_is_a_swarm_with_the_documented_name():
    sw = make_memory_swarm()
    assert isinstance(sw, Swarm) and isinstance(sw, MemorySwarm)
    assert sw.name == "swarm_memory"
    assert make_memory_swarm(name="x").name == "x"


def test_memory_is_per_episode_state():
    a, b = make_memory_swarm(), make_memory_swarm()
    a.learn(ev(1, 0.0), True)
    assert a.memory.attack and not b.memory.attack


def test_runs_are_deterministic():
    ep = episode_for(TRAIN_SEEDS[2], TRAIN_FAMILIES)
    r1 = run_episode(ep, make_memory_swarm())
    r2 = run_episode(ep, make_memory_swarm())
    assert r1.verdicts == r2.verdicts
    assert {k: (o.paid, o.paid_at) for k, o in r1.outcomes.items()} == {k: (o.paid, o.paid_at) for k, o in r2.outcomes.items()}


def test_memory_with_everything_switched_off_matches_the_plain_swarm_end_to_end():
    ep = episode_for(TRAIN_SEEDS[3], TRAIN_FAMILIES)
    off = MemoryParams(weight_worker=0.0, weight_device=0.0, weight_destination=0.0, weight_requester=0.0)
    assert run_episode(ep, make_memory_swarm(None, off)).verdicts == run_episode(ep, make_swarm()).verdicts


def test_memory_never_reads_ground_truth():
    tree = ast.parse(inspect.getsource(memory_module))
    forbidden = {"truth", "Truth", "family", "campaign_id", "Episode", "episode"}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not used & forbidden
    # labels enter only through learn(): the only function that passes `is_attack` on is MemorySwarm.learn
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "decide":
            assert "is_attack" not in {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


# --- held-out data is never touched ---

BANNED = ("TEST_SEEN_SEEDS", "TEST_HELDOUT_SEEDS", "HELDOUT_FAMILIES", "sleeper", "TEST_HELDOUT", "TEST_SEEN")


@pytest.mark.parametrize("rel", ["payguard/swarm/memory.py", "scripts/run_memory.py"])
def test_new_files_never_touch_heldout_data(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    for word in BANNED:
        assert word not in src, (rel, word)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            assert not 2000 <= node.value <= 3999, (rel, node.value, "seed in the held-out range")
