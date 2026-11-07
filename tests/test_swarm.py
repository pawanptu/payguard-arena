"""The swarm must behave as designed: bounded and deterministic specialists, a graph that separates rings from
honest hard negatives, and an L1 coordinator whose weights are well-behaved."""
import ast
import dataclasses
import inspect
import math

import numpy as np
import pytest

import payguard.swarm.coordinator as coordinator_module
import payguard.swarm.defenders as defenders_module
from payguard.baselines.static_rules import StaticRules
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for, episodes
from payguard.sim.attacks import RingParams
from payguard.sim.engine import LABEL_DELAY_H, build_episode
from payguard.sim.features import PlatformLog
from payguard.sim.metrics import evaluate
from payguard.sim.runner import run_episode
from payguard.sim.types import Event, Verdict
from payguard.swarm.coordinator import NAMES, Swarm, make_swarm, normalize_with_floor, verdict_for
from payguard.swarm.defenders import SPECIALISTS, BaselineDrift, Graph, Policy, Velocity, make_specialist
from payguard.swarm.objective import describe, objective
from payguard.swarm.params import (
    BOUNDS,
    INT_FIELDS,
    SwarmParams,
    default_params,
    mutate,
    sample_params,
)
from payguard.swarm.single import SPECIALIST_NAMES, SingleDefender, make_single

N_POPULAR_REQUESTERS = 60  # SimConfig default: requester ids 0..59 are popular, 60..65 are small shops


def scores_over(ep, defender, until_t=None):
    """Score every event of an episode with `defender`, adding each event to the log AFTER scoring it."""
    log, out = PlatformLog(ep.accounts), {}
    for e in ep.events:
        if until_t is not None and e.t > until_t:
            break
        out[e.id] = defender.score(e, log)
        log.add(e)
    return out


def ev(i, t, worker, requester=0, device=None, destination=None, amount=50.0):
    return Event(i, t, worker, requester, worker if device is None else device,
                 worker if destination is None else destination, amount)


# --- params ---

def test_bounds_cover_every_field_and_defaults_sit_inside():
    names = {f.name for f in dataclasses.fields(SwarmParams)}
    assert set(BOUNDS) == names
    assert INT_FIELDS <= names
    p = default_params()
    for name, (lo, hi) in BOUNDS.items():
        assert lo < hi
        assert lo <= getattr(p, name) <= hi, name
        assert isinstance(getattr(p, name), int) == (name in INT_FIELDS), name
    assert all(isinstance(getattr(p, n), (int, float)) for n in names)  # numeric genome only


def test_sampled_and_mutated_params_respect_bounds_and_keep_ints_as_ints():
    rng = np.random.default_rng(3)
    p = sample_params(rng)
    for _ in range(100):
        p = mutate(p, rng, scale=0.5)
        for name, (lo, hi) in BOUNDS.items():
            v = getattr(p, name)
            assert lo <= v <= hi, name
            assert isinstance(v, int) == (name in INT_FIELDS), name
        assert p.step_up_threshold < p.hold_threshold  # disjoint bounds keep the verdict order valid
        assert p.min_weight < 0.25                      # a floor above 1/4 would pin the four weights


def test_sampling_and_mutation_are_deterministic_given_the_generator():
    a, b = sample_params(np.random.default_rng(11)), sample_params(np.random.default_rng(11))
    assert a == b
    assert mutate(a, np.random.default_rng(5)) == mutate(b, np.random.default_rng(5))
    assert mutate(a, np.random.default_rng(5)) != mutate(a, np.random.default_rng(6))


def test_objective_is_the_preregistered_formula():
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    m = evaluate([run_episode(ep, make_single("policy"))])
    assert objective(m) == pytest.approx(m.leakage_pct + 0.5 * m.honest_block_rate + 1.0 * m.honest_unpaid_rate)
    assert "leak" in describe(m) and "block" in describe(m) and "unpaid" in describe(m)


# --- specialists ---

def test_specialists_are_bounded_and_deterministic_on_a_train_episode():
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    p = default_params()
    for name in SPECIALIST_NAMES:
        first = scores_over(ep, make_specialist(name, p))
        again = scores_over(ep, make_specialist(name, p))
        assert first == again, name
        assert all(0.0 <= s <= 1.0 and math.isfinite(s) for s in first.values()), name
        assert max(first.values()) > 0.0, f"{name} never fires on a TRAIN episode"


def test_specialists_stay_bounded_under_extreme_params():
    ep = episode_for(TRAIN_SEEDS[1], TRAIN_FAMILIES)
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = sample_params(rng)
        for name in SPECIALIST_NAMES:
            s = scores_over(ep, make_specialist(name, p), until_t=24 * 4)
            assert all(0.0 <= v <= 1.0 and math.isfinite(v) for v in s.values()), name


def test_graph_scores_a_ring_far_above_small_shops_and_popular_requesters():
    """Once the ring has accumulated history, its payouts must stand out from honest hard negatives."""
    ep = build_episode(5, [("ring", RingParams())])
    graph = scores_over(ep, Graph(default_params()))
    late = lambda i: ep.events[i].t >= 6 * 24  # noqa: E731
    ring = [graph[e.id] for e in ep.events if ep.truth[e.id].family == "ring" and late(e.id)]
    shops = [graph[e.id] for e in ep.events if e.requester_id >= N_POPULAR_REQUESTERS
             and ep.truth[e.id].family == "honest" and late(e.id)]
    popular = [graph[e.id] for e in ep.events if e.requester_id < N_POPULAR_REQUESTERS
               and ep.truth[e.id].family == "honest" and late(e.id)]
    assert len(ring) > 50 and len(shops) > 20 and len(popular) > 500
    assert np.median(ring) > 0.9
    assert np.mean(np.array(ring) >= 0.7) > 0.8
    assert max(shops) < 0.1                       # shops serve about 3 workers: not a ring
    assert np.mean(np.array(popular) >= 0.5) < 0.02  # popular requesters pay many unrelated workers


def test_graph_cannot_see_the_first_payouts_of_a_ring():
    """Documented limit: with fewer than graph_min_workers distinct workers on the requester there is nothing collective."""
    ep = build_episode(5, [("ring", RingParams())])
    graph = scores_over(ep, Graph(default_params()))
    first = [e.id for e in ep.events if ep.truth[e.id].family == "ring"][:3]
    assert all(graph[i] == 0.0 for i in first)


def test_graph_flags_a_device_shared_by_many_workers_but_not_a_pair():
    g = Graph(default_params())
    log = PlatformLog({w: -24.0 * 100 for w in range(10)})
    for i, w in enumerate((1, 2)):
        log.add(ev(i, 1.0 + i, w, requester=w, device=77))
    assert g.score(ev(10, 5.0, 2, requester=2, device=77), log) == 0.0   # two workers on a device: family
    log.add(ev(11, 6.0, 3, requester=3, device=77))
    log.add(ev(12, 7.0, 4, requester=4, device=77))
    assert g.score(ev(13, 8.0, 5, requester=5, device=77), log) > 0.9    # five workers on one device


def test_velocity_sees_a_burst_that_a_single_payout_hides():
    v = Velocity(default_params())
    log = PlatformLog({1: -24.0 * 100})
    scores = []
    for i in range(5):
        e = ev(i, 100.0 + i, 1, amount=480.0)
        scores.append(v.score(e, log))
        log.add(e)
    assert scores[0] == 0.0
    assert scores == sorted(scores) and scores[-1] > 0.5


def test_policy_limits_scale_with_account_age():
    p = Policy(default_params())
    young, old = PlatformLog({1: -24.0 * 15}), PlatformLog({1: -24.0 * 400})
    big = ev(0, 0.0, 1, amount=450.0)
    assert p.score(big, young) > 0.8
    assert p.score(big, old) < p.score(big, young)
    assert p.score(ev(1, 0.0, 1, amount=60.0), young) == 0.0
    assert p.cap(0.0) < p.cap(10.0) < p.cap(1000.0) == default_params().policy_cap_max_usd


def test_drift_needs_history_and_then_flags_a_jump_against_the_workers_own_baseline():
    d = BaselineDrift(default_params())
    log = PlatformLog({1: -24.0 * 60})
    jump = ev(100, 200.0, 1, amount=400.0)
    assert d.score(jump, log) == 0.0               # no baseline yet: cannot drift from nothing
    for i in range(8):
        log.add(ev(i, 10.0 + 24.0 * i, 1, amount=40.0 + i))
    assert d.score(ev(101, 200.0, 1, amount=44.0), log) == 0.0
    assert d.score(jump, log) > 0.9


# --- coordinator and L1 ---

def test_normalize_with_floor_sums_to_one_and_respects_the_floor():
    out = normalize_with_floor({"a": 1e-9, "b": 5.0, "c": 5.0, "d": 1e-12}, floor=0.05)
    assert sum(out.values()) == pytest.approx(1.0)
    assert min(out.values()) >= 0.05 - 1e-12
    assert out["a"] == pytest.approx(0.05) and out["d"] == pytest.approx(0.05)
    assert out["b"] == pytest.approx(out["c"])


def test_initial_weights_follow_the_params_and_are_normalised():
    s = Swarm(dataclasses.replace(default_params(), w0_graph=0.9, w0_drift=0.3))
    assert set(s.weights) == set(NAMES)
    assert sum(s.weights.values()) == pytest.approx(1.0)
    assert s.weights["graph"] > s.weights["drift"] > 0


def _score_row(**kw):
    return {n: kw.get(n, 0.0) for n in NAMES}


def test_weights_stay_valid_and_move_toward_the_more_accurate_defender():
    p = dataclasses.replace(default_params(), eta=0.1, min_weight=0.05)
    s = Swarm(p)
    start = dict(s.weights)

    def feed(i):
        # Graph is right every time (alarm on attacks, silent on honest); Velocity is always wrong;
        # Policy and Drift are lukewarm (half right) either way.
        attack = i % 3 == 0
        s._pending[i] = _score_row(graph=1.0 if attack else 0.0, velocity=0.0 if attack else 1.0,
                                   drift=0.5, policy=0.5)
        s.learn(ev(i, float(i), 1), attack)
        assert sum(s.weights.values()) == pytest.approx(1.0)
        assert min(s.weights.values()) >= p.min_weight - 1e-12

    for i in range(30):
        feed(i)
    assert s.weights["graph"] > start["graph"]
    assert s.weights["velocity"] < start["velocity"]
    assert s.weights["graph"] > s.weights["policy"] > s.weights["velocity"]
    assert not s._pending                               # every remembered event was consumed by its label
    for i in range(30, 400):
        feed(i)
    assert s.weights["velocity"] == pytest.approx(p.min_weight)  # pinned at the floor, never below


def test_frozen_swarm_never_moves_its_weights():
    s = Swarm(default_params(), learn_weights=False)
    start = dict(s.weights)
    ep = episode_for(TRAIN_SEEDS[2], TRAIN_FAMILIES)
    run_episode(ep, s)
    assert s.weights == start
    assert not s._pending                          # a frozen swarm remembers nothing


def test_learn_ignores_labels_for_events_it_never_decided():
    s = Swarm(default_params())
    before = dict(s.weights)
    s.learn(ev(999, 0.0, 1), True)
    assert s.weights == before


def test_pending_memory_is_released_when_labels_arrive():
    ep = episode_for(TRAIN_SEEDS[3], TRAIN_FAMILIES)
    s = Swarm(default_params())
    run_episode(ep, s)
    last_t = ep.events[-1].t
    in_flight = [e for e in ep.events if e.t + LABEL_DELAY_H > last_t]  # at most these can still be unlabelled
    assert 0 < len(s._pending) <= len(in_flight)


def test_swarm_runs_end_to_end_and_exposes_signals_for_the_dashboard():
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    s = make_swarm(default_params())
    res = run_episode(ep, s)
    assert set(res.verdicts) == {e.id for e in ep.events}
    assert {Verdict.RELEASE, Verdict.HOLD} <= set(res.verdicts.values())
    m = evaluate([res])
    assert math.isfinite(objective(m)) and m.leakage_pct < 100.0
    sig = s.last_signals
    assert sig["event_id"] == ep.events[-1].id
    assert set(sig["scores"]) == set(NAMES) and set(sig["weights"]) == set(NAMES)
    assert sig["verdict"] == res.verdicts[ep.events[-1].id].value
    assert sum(s.weights.values()) == pytest.approx(1.0)
    assert s.weights != Swarm(default_params()).weights   # L1 actually learned something from delayed labels


def test_swarm_is_deterministic():
    ep = episode_for(TRAIN_SEEDS[4], TRAIN_FAMILIES)
    a, b = run_episode(ep, make_swarm()), run_episode(ep, make_swarm())
    assert a.verdicts == b.verdicts


def test_single_defender_wraps_one_specialist_with_the_same_thresholds():
    assert set(SPECIALIST_NAMES) == set(SPECIALISTS) == set(NAMES)
    with pytest.raises(ValueError):
        SingleDefender("nope")
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    p = default_params()
    single = make_single("graph", p)
    assert single.name == "single_graph"
    res = run_episode(ep, single)
    expected = scores_over(ep, Graph(p))
    assert {i: verdict_for(s, p) for i, s in expected.items()} == res.verdicts


def test_verdict_mapping_respects_the_two_thresholds():
    p = default_params()
    assert verdict_for(0.0, p) is Verdict.RELEASE
    assert verdict_for(p.step_up_threshold, p) is Verdict.STEP_UP
    assert verdict_for(p.hold_threshold, p) is Verdict.HOLD
    assert verdict_for(1.0, p) is Verdict.HOLD


# --- isolation and value ---

GROUND_TRUTH_NAMES = {"truth", "Truth", "is_attack", "family", "campaign_id"}


def _names_used(func_node):
    return {n.id for n in ast.walk(func_node) if isinstance(n, ast.Name)} |            {n.attr for n in ast.walk(func_node) if isinstance(n, ast.Attribute)}


def test_decisions_never_touch_ground_truth():
    """Specialists and Swarm.decide may use Event + PlatformLog only; labels enter through learn() alone."""
    tree = ast.parse(inspect.getsource(defenders_module))
    assert not _names_used(tree) & GROUND_TRUTH_NAMES
    swarm_tree = ast.parse(inspect.getsource(coordinator_module))
    for node in ast.walk(swarm_tree):
        if isinstance(node, ast.FunctionDef) and node.name in {"decide", "aggregate", "_pending_add"}:
            assert not _names_used(node) & GROUND_TRUTH_NAMES, node.name


def test_swarm_clearly_beats_static_rules_and_every_single_specialist_on_train():
    eps = episodes(TRAIN_SEEDS[:4], TRAIN_FAMILIES)

    def fitness(make):
        return objective(evaluate([run_episode(ep, make()) for ep in eps]))

    swarm = fitness(make_swarm)
    others = {"static_rules": fitness(StaticRules), **{n: fitness(lambda n=n: make_single(n)) for n in SPECIALIST_NAMES}}
    assert swarm < 0.6 * min(others.values()), (swarm, others)
