"""The simulator must behave as designed, or the benchmark proves nothing."""
import dataclasses
from itertools import combinations

import numpy as np
import pytest

from payguard.protocol import (
    HELDOUT_FAMILIES,
    TEST_HELDOUT_SEEDS,
    TEST_SEEN_SEEDS,
    TRAIN_FAMILIES,
    TRAIN_SEEDS,
    episode_for,
)
from payguard.sim.attacks import (
    BOUNDS,
    INT_FIELDS,
    PARAMS,
    REVIEW_THRESHOLD_USD,
    RingParams,
    SleeperParams,
    StructuringParams,
    mutate,
    sample_params,
)
from payguard.sim.engine import HOLD_DELAY_H, LABEL_DELAY_H, build_episode, resolve
from payguard.sim.features import PlatformLog
from payguard.sim.runner import run_episode
from payguard.sim.types import Event, Verdict


def features_for(ep):
    log, out = PlatformLog(ep.accounts), {}
    for e in ep.events:
        out[e.id] = log.features(e)
        log.add(e)
    return out, log


def ids_of(ep, family):
    return [i for i, t in ep.truth.items() if t.family == family]


def test_episode_is_deterministic_and_sorted():
    a = build_episode(5, [("ring", RingParams())])
    b = build_episode(5, [("ring", RingParams())])
    assert a.events == b.events and a.truth == b.truth
    assert [e.t for e in a.events] == sorted(e.t for e in a.events)
    assert [e.id for e in a.events] == list(range(len(a.events)))


def test_defenders_cannot_see_ground_truth():
    names = {f.name for f in dataclasses.fields(Event)}
    assert not names & {"is_attack", "family", "campaign_id", "truth"}
    ep = build_episode(5, [("ring", RingParams())])
    feats, _ = features_for(ep)
    assert not set(next(iter(feats.values()))) & {"is_attack", "family", "campaign_id"}


def test_ring_is_clean_per_account():
    """Per-account rules must not see the ring: aged accounts, normal amounts and rate, own device."""
    ep = build_episode(5, [("ring", RingParams())])
    feats, _ = features_for(ep)
    ring, honest = ids_of(ep, "ring"), ids_of(ep, "honest")
    assert len(ring) > 50
    # Structural per-account signals must never fire on the ring.
    structural = [i for i in ring if feats[i]["age_days"] < 7 or feats[i]["amount"] >= REVIEW_THRESHOLD_USD
                  or feats[i]["window_n"] > 5 or feats[i]["window_usd"] >= 800
                  or feats[i]["device_workers"] > 1 or feats[i]["destination_workers"] > 1]
    assert not structural

    # The amount-drift signal is lognormal noise for everyone: the ring must not stand out from honest traffic.
    def drift_rate(ids):
        return sum(feats[i]["amount_ratio"] > 3.0 for i in ids) / len(ids)

    assert drift_rate(ring) <= drift_rate(honest) + 0.02


def test_ring_is_detectable_collectively():
    """Not undetectable by construction: ring workers share their requester set, honest workers do not."""
    ep = build_episode(5, [("ring", RingParams())])
    _, log = features_for(ep)
    ring_workers = {ep.events[i].worker_id for i in ids_of(ep, "ring")}
    honest = [w for w in range(300) if len(log.requesters_by_worker[w]) >= 1]

    def jaccard(a, b):
        sa, sb = log.requesters_by_worker[a], log.requesters_by_worker[b]
        return len(sa & sb) / len(sa | sb)

    ring_j = np.mean([jaccard(a, b) for a, b in combinations(sorted(ring_workers), 2)])
    rng = np.random.default_rng(0)
    honest_j = np.mean([jaccard(*rng.choice(honest, 2, replace=False)) for _ in range(500)])
    assert ring_j > 0.8
    assert honest_j < 0.25


def test_structuring_dodges_single_threshold_but_not_velocity():
    ep = build_episode(7, [("structuring", StructuringParams())])
    feats, _ = features_for(ep)
    ids = ids_of(ep, "structuring")
    assert ids and all(ep.events[i].amount < REVIEW_THRESHOLD_USD for i in ids)
    caught = [i for i in ids if feats[i]["window_usd"] >= 800]
    assert len(caught) / len(ids) >= 0.5


def test_sleeper_cashout_is_far_above_own_baseline_but_below_threshold():
    p = SleeperParams(base_rate_per_day=1.2)
    ep = build_episode(9, [("sleeper", p)])
    feats, _ = features_for(ep)
    by_worker = {}
    for i in ids_of(ep, "sleeper"):
        by_worker.setdefault(ep.events[i].worker_id, []).append(i)
    assert len(by_worker) == p.n_accounts
    for ids in by_worker.values():
        assert len(ids) <= p.cashout_count  # only the cash-out is labelled attack, never the trust-building phase
        cash = ids
        assert all(ep.events[i].amount < REVIEW_THRESHOLD_USD for i in ids)
        assert all(feats[i]["amount_ratio"] >= 3.0 for i in cash)  # drift is visible to a baseline defender
        assert all(feats[i]["window_n"] <= 3 for i in cash)         # but spaced out to dodge velocity rules


def test_resolve_is_deterministic_and_matches_review_accuracy():
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    first = [resolve(ep.seed, e, ep.truth[e.id], Verdict.HOLD) for e in ep.events]
    again = [resolve(ep.seed, e, ep.truth[e.id], Verdict.HOLD) for e in ep.events]
    assert first == again
    honest_paid = [o.paid for o, e in zip(first, ep.events) if not ep.truth[e.id].is_attack]
    attack_paid = [o.paid for o, e in zip(first, ep.events) if ep.truth[e.id].is_attack]
    assert 0.85 < np.mean(honest_paid) < 0.95
    assert 0.0 < np.mean(attack_paid) < 0.25
    assert all(o.paid_at - e.t == pytest.approx(HOLD_DELAY_H) for o, e in zip(first, ep.events) if o.paid)


def test_release_pays_immediately_and_label_arrives_late():
    ep = episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES)
    e = ep.events[10]
    oc = resolve(ep.seed, e, ep.truth[e.id], Verdict.RELEASE)
    assert oc.paid and oc.paid_at == e.t and oc.label_at == e.t + LABEL_DELAY_H


def test_labels_reach_the_defender_only_after_their_delay():
    class Recorder:
        name = "recorder"

        def __init__(self, ep):
            self.ep, self.fresh, self.learned = ep, [], 0

        def decide(self, event, log):
            # The runner delivers due labels just before each decision, so event.t is the delivery time.
            assert all(event.t >= old.t + LABEL_DELAY_H for old in self.fresh)
            self.fresh.clear()
            return Verdict.RELEASE

        def learn(self, event, is_attack):
            assert is_attack == self.ep.truth[event.id].is_attack
            self.fresh.append(event)
            self.learned += 1

    ep = episode_for(TRAIN_SEEDS[1], TRAIN_FAMILIES)
    rec = Recorder(ep)
    run_episode(ep, rec)
    assert 0 < rec.learned < len(ep.events)  # the most recent events are still unlabelled


def test_protocol_splits_are_disjoint_and_family_pure():
    assert not set(TRAIN_SEEDS) & set(TEST_SEEN_SEEDS) | set(TRAIN_SEEDS) & set(TEST_HELDOUT_SEEDS)
    assert not set(TEST_SEEN_SEEDS) & set(TEST_HELDOUT_SEEDS)
    assert {t.family for t in episode_for(TRAIN_SEEDS[0], TRAIN_FAMILIES).truth.values()} <= {"honest", *TRAIN_FAMILIES}
    held = {t.family for t in episode_for(TEST_HELDOUT_SEEDS[0], HELDOUT_FAMILIES).truth.values()}
    assert held == {"honest", "sleeper"}


def test_sampled_and_mutated_params_respect_bounds():
    rng = np.random.default_rng(3)
    for family, cls in PARAMS.items():
        p = sample_params(family, rng)
        for _ in range(50):
            p = mutate(p, family, rng, scale=0.5)
            for name, (lo, hi) in BOUNDS[family].items():
                v = getattr(p, name)
                assert lo <= v <= hi
                assert isinstance(v, int) == (name in INT_FIELDS)
        assert isinstance(p, cls)
