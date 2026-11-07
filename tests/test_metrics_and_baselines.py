import math

from payguard.baselines.static_rules import StaticRules
from payguard.baselines.trivial import AlwaysHold, AlwaysRelease
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episodes
from payguard.sim.metrics import _campaign_leak_before_containment, evaluate
from payguard.sim.runner import run_episode
from payguard.sim.types import Verdict


def run(make, eps):
    return evaluate([run_episode(ep, make()) for ep in eps])


def test_always_release_is_the_zero_friction_maximum_leakage_bound():
    m = run(AlwaysRelease, episodes(TRAIN_SEEDS[:3], TRAIN_FAMILIES))
    assert m.leakage_pct == 100.0
    assert m.honest_block_rate == 0.0 and m.honest_unpaid_rate == 0.0
    assert m.honest_median_ttp_h == 0.0
    assert m.never_contained_rate == 100.0


def test_always_hold_is_bounded_by_review_accuracy_and_delay():
    m = run(AlwaysHold, episodes(TRAIN_SEEDS[:3], TRAIN_FAMILIES))
    assert 5.0 < m.leakage_pct < 15.0          # review is 90% accurate
    assert 5.0 < m.honest_unpaid_rate < 15.0   # and wrongly cancels about 10% of honest payouts
    assert m.honest_median_ttp_h == 24.0
    assert m.honest_block_rate == 100.0


def test_static_rules_sit_between_the_bounds_and_leak_on_the_clean_ring():
    eps = episodes(TRAIN_SEEDS[:10], TRAIN_FAMILIES)
    results = [run_episode(ep, StaticRules()) for ep in eps]
    m = evaluate(results)
    assert math.isfinite(m.honest_median_ttp_h)
    assert m.leakage_pct < 100.0
    assert m.leakage_pct_by_family["ring"] > 75.0
    # Invariant of the benchmark: per-account rules cannot see a ring that is clean per account, so they
    # block ring payouts no more often than they block honest payouts (generic friction, not detection).
    ring = [r.verdicts[e.id] is not Verdict.RELEASE for r in results for e in r.episode.events
            if r.episode.truth[e.id].family == "ring"]
    assert 100.0 * sum(ring) / len(ring) <= m.honest_block_rate + 5.0


def test_containment_counts_leaks_before_three_consecutive_blocks():
    assert _campaign_leak_before_containment([False, False, True, True, True, True],
                                             [True, True, False, False, False, False]) == (2, True)
    assert _campaign_leak_before_containment([False, True, False, True], [True, False, True, False]) == (2, False)
    assert _campaign_leak_before_containment([True, True], [False, False]) == (0, True)
