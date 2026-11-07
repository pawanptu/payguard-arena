"""Generation-level evolution (L2): splits, evaluation, GA operators, the gate, the abstention gene, the loop's
wiring (promote / reject / rollback), the replay contract and the held-out isolation."""
import ast
import dataclasses
import json
import math
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import payguard.evolve.loop as loop_module
from payguard.evolve import attackers as atk
from payguard.evolve import replay_export as rx
from payguard.evolve.evaluation import (
    Evaluator,
    arena_spec,
    evaluate,
    get_episode,
    proto_spec,
    swarm_job,
)
from payguard.evolve.ga import (
    crossover,
    initial_population,
    mutate_sparse,
    mutation_scale,
    params_key,
    tournament,
)
from payguard.evolve.gate import gate_decision, rollback_decision
from payguard.evolve.loop import EvolveConfig, make_plan, run_evolution
from payguard.evolve.splits import FAMILIES, TRAIN_A, TRAIN_B
from payguard.protocol import TRAIN_FAMILIES, TRAIN_SEEDS, episode_for, episodes
from payguard.sim.attacks import RingParams, StructuringParams
from payguard.sim.attacks import sample_params as sample_attacker
from payguard.sim.engine import build_episode
from payguard.sim.metrics import Metrics
from payguard.swarm.coordinator import ACTIVE_SCORE, NAMES, Swarm, make_swarm, normalize_with_floor
from payguard.swarm.objective import objective
from payguard.swarm.params import (
    BOUNDS,
    INT_FIELDS,
    SwarmParams,
    default_params,
    mutate,
    sample_params,
)
from scripts.evolve import main as evolve_main
from scripts.run_baselines import run

ROOT = Path(__file__).resolve().parent.parent


# --- splits ---

def test_splits_are_slices_of_the_protocol_train_seeds():
    assert TRAIN_A == TRAIN_SEEDS[:10] and TRAIN_B == TRAIN_SEEDS[10:]
    assert not set(TRAIN_A) & set(TRAIN_B)
    assert FAMILIES == TRAIN_FAMILIES == ("ring", "structuring")


# --- the abstention-aware L1 gene ---

def test_l1_abstain_is_the_last_int_gene_with_binary_bounds_and_default_off():
    assert [f.name for f in dataclasses.fields(SwarmParams)][-1] == "l1_abstain"
    assert BOUNDS["l1_abstain"] == (0, 1) and "l1_abstain" in INT_FIELDS
    assert default_params().l1_abstain == 0 and isinstance(default_params().l1_abstain, int)


def test_sampling_mutation_crossover_keep_bounds_and_ints_including_l1_abstain():
    rng = np.random.default_rng(4)
    seen = set()
    p, q = sample_params(rng), sample_params(rng)
    for _ in range(200):
        child = mutate_sparse(crossover(p, q, rng), rng, 0.5)
        for name, (lo, hi) in BOUNDS.items():
            v = getattr(child, name)
            assert lo <= v <= hi, name
            assert isinstance(v, int) == (name in INT_FIELDS), name
        assert child.step_up_threshold < child.hold_threshold
        seen.add(child.l1_abstain)
        p = mutate(child, rng, 0.3)
        assert p.l1_abstain in (0, 1) and isinstance(p.l1_abstain, int)
    assert seen == {0, 1}   # the gene really flips


def test_crossover_takes_every_gene_from_one_of_the_parents_and_mutation_is_sparse():
    rng = np.random.default_rng(0)
    a, b = sample_params(rng), sample_params(rng)
    c = crossover(a, b, rng)
    for f in dataclasses.fields(a):
        assert getattr(c, f.name) in (getattr(a, f.name), getattr(b, f.name))
    moved = [sum(getattr(m, f.name) != getattr(a, f.name) for f in dataclasses.fields(a))
             for m in (mutate_sparse(a, np.random.default_rng(i), 0.3) for i in range(60))]
    assert min(moved) >= 1 and 3 < np.mean(moved) < 16   # about 30% of the 29 genes, never zero


def test_ga_operators_are_deterministic_and_the_scale_decays():
    a = sample_params(np.random.default_rng(1))
    assert mutate_sparse(a, np.random.default_rng(2), 0.2) == mutate_sparse(a, np.random.default_rng(2), 0.2)
    assert initial_population(8, np.random.default_rng(3)) == initial_population(8, np.random.default_rng(3))
    assert mutation_scale(0, 10, 0.25, 0.05) == 0.25 and mutation_scale(9, 10, 0.25, 0.05) == pytest.approx(0.05)
    assert mutation_scale(0, 1, 0.25, 0.05) == 0.25
    fit = [5.0, 1.0, 3.0]
    wins = [tournament(fit, 3, np.random.default_rng(i)) for i in range(50)]
    assert set(wins) <= {0, 1, 2} and wins.count(1) > 25


def test_initial_population_holds_the_default_the_abstain_variant_mutants_and_fresh_draws():
    pop = initial_population(12, np.random.default_rng(0))
    assert len(pop) == 12 and len({cid for _, cid, _ in pop}) == 12
    assert pop[0][0] == default_params() and pop[0][1] == "default"
    assert pop[1][0] == replace(default_params(), l1_abstain=1)
    assert {o for _, _, o in pop} == {"default", "default+l1_abstain", "mutant_of_default", "fresh_sample"}


def _swarm_with(p: SwarmParams, pending: dict[str, float]) -> Swarm:
    s = Swarm(p)
    s._pending[1] = pending
    return s


def test_l1_abstain_loss_is_exact_on_a_hand_made_case():
    p = replace(default_params(), eta=0.5, min_weight=0.01, l1_abstain=1)
    scores = {"velocity": 0.0, "graph": 0.9, "drift": 0.04, "policy": ACTIVE_SCORE}
    # attack: velocity abstained (0), graph spoke up and was right (1-0.9), drift below ACTIVE_SCORE abstained (0),
    # policy exactly at ACTIVE_SCORE counts as speaking up (1-0.05)
    expect_loss = {"velocity": 0.0, "graph": 0.1, "drift": 0.0, "policy": 0.95}
    s = _swarm_with(p, scores)
    start = dict(s.weights)
    s.learn(_ev(1), True)
    raw = {n: start[n] * math.exp(-0.5 * expect_loss[n]) for n in NAMES}
    want = normalize_with_floor(raw, 0.01)
    for n in NAMES:
        assert s.weights[n] == pytest.approx(want[n], abs=1e-15)
    assert s.weights["velocity"] == s.weights["drift"] > s.weights["graph"] > s.weights["policy"]
    # honest payout: every specialist pays its own score (crying wolf), abstaining or not
    s2 = _swarm_with(p, scores)
    s2.learn(_ev(1), False)
    want2 = normalize_with_floor({n: start[n] * math.exp(-0.5 * scores[n]) for n in NAMES}, 0.01)
    for n in NAMES:
        assert s2.weights[n] == pytest.approx(want2[n], abs=1e-15)
    assert Swarm.abstain_loss(0.04, True) == 0.0 and Swarm.abstain_loss(0.5, False) == 0.5


def test_l1_abstain_differs_from_the_old_loss_exactly_where_a_specialist_abstained():
    scores = {"velocity": 0.0, "graph": 0.9, "drift": 0.0, "policy": 0.0}
    old = _swarm_with(replace(default_params(), eta=0.5), scores)
    new = _swarm_with(replace(default_params(), eta=0.5, l1_abstain=1), scores)
    old.learn(_ev(1), True)
    new.learn(_ev(1), True)
    assert new.weights["graph"] < old.weights["graph"] or new.weights["velocity"] > old.weights["velocity"]
    assert new.weights["velocity"] == new.weights["drift"] == new.weights["policy"]  # nobody blamed for silence


def test_l1_abstain_zero_is_exactly_the_old_update_formula():
    p = replace(default_params(), eta=0.3, min_weight=0.07)
    s = Swarm(p)
    rng = np.random.default_rng(5)
    old = dict(s.weights)
    for i in range(300):
        scores = {n: float(rng.random()) * (rng.random() < 0.7) for n in NAMES}
        attack = bool(rng.random() < 0.3)
        s._pending[i] = scores
        s.learn(_ev(i), attack)
        y = 1.0 if attack else 0.0
        old = normalize_with_floor({n: old[n] * math.exp(-p.eta * abs(scores[n] - y)) for n in NAMES}, p.min_weight)
        assert s.weights == old   # bit for bit, not approx


def test_default_swarm_results_are_unchanged_by_the_new_gene():
    eps = episodes(TRAIN_SEEDS[:2], TRAIN_FAMILIES)
    m_l1 = evaluate(default_params(), eps, True)
    m_frozen = evaluate(default_params(), eps, False)
    assert objective(m_l1) == objective(run(lambda: make_swarm(default_params(), True), eps))
    assert objective(m_frozen) == objective(run(lambda: make_swarm(default_params(), False), eps))


def _ev(i):
    from payguard.sim.types import Event
    return Event(i, float(i), 1, 0, 1, 1, 50.0)


# --- evaluation ---

def test_evaluate_matches_run_baselines_aggregation_exactly():
    eps = episodes(TRAIN_SEEDS[:2], TRAIN_FAMILIES)
    p = replace(default_params(), eta=0.2)
    assert evaluate(p, eps, True) == run(lambda: make_swarm(p, True), eps)
    assert evaluate(p, eps, False) == run(lambda: make_swarm(p, False), eps)


def test_episode_specs_rebuild_the_same_episodes_once():
    ep = get_episode(proto_spec(TRAIN_SEEDS[1]))
    assert ep is get_episode(proto_spec(TRAIN_SEEDS[1]))                      # cached
    assert ep.events == episode_for(TRAIN_SEEDS[1], TRAIN_FAMILIES).events
    camp = [("ring", RingParams(ring_size=8)), ("structuring", StructuringParams(k=4))]
    assert get_episode(arena_spec(1003, camp)).events == build_episode(1003, camp).events


def test_same_seed_different_attackers_gives_the_same_honest_traffic():
    a = get_episode(arena_spec(1004, [("ring", RingParams(ring_size=6))]))
    b = get_episode(arena_spec(1004, [("ring", RingParams(ring_size=20, n_requesters=4))]))

    def honest(ep):
        return [(e.t, e.worker_id, e.requester_id, e.amount) for e in ep.events if not ep.truth[e.id].is_attack]
    assert honest(a) == honest(b) and len(a.events) != len(b.events)


def test_pool_results_do_not_depend_on_the_worker_count():
    specs = [proto_spec(TRAIN_SEEDS[0]), proto_spec(TRAIN_SEEDS[1])]
    jobs = [swarm_job(default_params(), specs), swarm_job(replace(default_params(), eta=0.4, l1_abstain=1), specs),
            ("static", tuple(specs))]
    with Evaluator(1) as e1, Evaluator(2, (TRAIN_SEEDS[0],)) as e2:
        r1, r2 = e1.run(jobs), e2.run(jobs)
    assert r1 == r2


# --- gate (pure) ---

def test_gate_promotes_only_with_enough_gain_and_bounded_friction():
    assert gate_decision(20.0, 19.5, 6.0, 6.0)                     # exactly 0.5 better: promote
    assert not gate_decision(20.0, 19.51, 6.0, 6.0)                # not enough gain
    assert gate_decision(20.0, 10.0, 6.0, 7.0)                     # +1.0 point of honest block is allowed
    assert not gate_decision(20.0, 10.0, 6.0, 7.01)                # more friction: rejected despite the big gain
    assert not gate_decision(20.0, 21.0, 6.0, 5.0)                 # worse objective
    assert gate_decision(20.0, 19.0, 6.0, 2.0)                     # less friction is always fine


def test_rollback_triggers_only_when_worse_than_the_predecessor_by_more_than_the_margin():
    assert not rollback_decision(18.0, 18.5)                       # exactly 0.5 worse: keep
    assert rollback_decision(18.0, 18.51)
    assert not rollback_decision(18.0, 17.0)                       # better: keep
    assert not rollback_decision(18.0, 18.0)


# --- attackers ---

def test_attacker_selection_keeps_the_top_leakers_per_family_and_arena_specs_are_deterministic():
    rng = np.random.default_rng(0)
    cands = [(f, sample_attacker(f, rng)) for f in FAMILIES for _ in range(5)]

    def fake(leak):
        return (_metrics(leak), None)
    results = [fake(float((i * 7) % 10)) for i in range(len(cands))]
    pools, recs = atk.select_strongest(cands, results, 3)
    for f in FAMILIES:
        leaks = [r["leaked_usd"] for r in recs[f]]
        assert len(pools[f]) == 3 and leaks == sorted(leaks, reverse=True)
    sp1 = atk.arena_specs(pools, TRAIN_A, 2, 3)
    assert sp1 == atk.arena_specs(pools, TRAIN_A, 2, 3) and len(sp1) == 3
    assert {s[1] for s in sp1} <= set(TRAIN_A)
    assert all(len(s[2]) == 2 and {c[0] for c in s[2]} == set(FAMILIES) for s in sp1)
    assert not {s[1] for s in atk.arena_specs(pools, TRAIN_B, 0, 3)} & set(TRAIN_A)


# --- the loop's wiring, with a fake evaluator (fast) ---

def _metrics(leak: float, block: float = 5.0) -> Metrics:
    return Metrics(n_episodes=1, attack_usd=100000.0, leaked_usd=leak * 1000.0, leakage_pct=leak,
                   leakage_pct_by_family={}, honest_n=1000, honest_block_rate=block, honest_unpaid_rate=0.0,
                   honest_median_ttp_h=0.0, honest_p95_ttp_h=0.0, immunity_attempts=0.0, never_contained_rate=0.0,
                   llm_calls=0)


class FakeEvaluator:
    """Objective = 20 - 10 * l1_abstain + noise(params, specs): the abstain variants are clearly better."""

    def __init__(self, block_of=lambda pd: 5.0):
        self.block_of = block_of
        self.calls = 0
        self.seeds: set[int] = set()

    def run(self, jobs):
        self.calls += len(jobs)
        out = []
        for job in jobs:
            assert job[0] == "swarm"
            pd, specs = job[1], job[3]
            self.seeds |= {sp[1] for sp in specs}
            noise = zlib.crc32(repr((sorted(pd.items()), specs)).encode()) % 1000 / 20000.0  # < 0.05: no gate can fire on noise
            out.append((_metrics(20.0 - 10.0 * pd["l1_abstain"] + noise, self.block_of(pd)), {"graph": 0.25}))
        return out


FAST = EvolveConfig(seed=1, pop=6, gens=4, elites=2, pool_size=2, attacker_seeds=1, arena_episodes=2, name="fake")


def test_loop_promotes_a_clearly_better_challenger_and_keeps_a_stable_champion():
    res = run_evolution(FAST, FakeEvaluator())
    lin = res["lineage"]
    assert [e["generation"] for e in lin] == [0, 1, 2, 3]
    assert lin[0]["promoted"] and lin[0]["champion_id"] == "default"
    assert res["champion"]["params"]["l1_abstain"] == 1
    assert not any(e["rolled_back"] for e in lin)
    assert all(e["champion_id_after"] != "default" for e in lin)
    for e in lin:
        assert set(e["strongest_attacker"]) == set(FAMILIES) and e["strongest_attacker_family"] in FAMILIES
        assert len(e["attacker_pools"]["ring"]) == 2 and set(e["champion_weights"]) == {"graph"}
        assert e["challenger"] is None or {"id", "params", "objective_a", "objective_b"} <= set(e["challenger"])
    ids = [c["id"] for c in res["candidates"]]
    assert len(ids) == len(set(ids)) and all(p in ids for c in res["candidates"] for p in c["parents"])


def test_loop_rejects_a_challenger_that_adds_too_much_friction():
    block = lambda pd: 5.0 + 3.0 * pd["l1_abstain"]  # noqa: E731 - the better candidates all block 3 points more
    res = run_evolution(FAST, FakeEvaluator(block))
    assert not any(e["promoted"] for e in res["lineage"])
    assert res["champion"]["id"] == "default" and res["champion"]["params"] == dataclasses.asdict(default_params())
    assert all(e["challenger"] is not None for e in res["lineage"])


def test_loop_rolls_back_and_never_re_promotes_the_rolled_back_candidate(monkeypatch):
    monkeypatch.setattr(loop_module, "gate_decision", lambda *a, **k: True)
    monkeypatch.setattr(loop_module, "rollback_decision", lambda *a, **k: True)
    res = run_evolution(replace(FAST, gens=5), FakeEvaluator())
    lin = res["lineage"]
    assert lin[0]["promoted"] and not lin[0]["rolled_back"]
    first = lin[0]["champion_id_after"]
    assert first != "default"
    assert lin[1]["rolled_back"] and lin[1]["rollback"]["from"] == first and lin[1]["rollback"]["to"] == "default"
    assert lin[1]["champion_id"] == "default"                       # the gate in generation 1 faced the restored champion
    assert lin[1]["challenger"]["id"] != first
    params_of = {c["id"]: c["params"] for c in res["candidates"]}
    rolled_params = []
    for e in lin:
        if e["challenger"] is not None:
            assert e["challenger"]["params"] not in rolled_params   # a rolled-back champion never returns as challenger
        if e["rolled_back"]:
            rolled_params.append(params_of[e["rollback"]["from"]])
    assert len(rolled_params) == 4
    for e in lin[1:]:
        assert e["rolled_back"] == (e["generation"] >= 1)           # promoted every generation, rolled back the next


def test_loop_without_rollback_trigger_keeps_the_promoted_champion(monkeypatch):
    monkeypatch.setattr(loop_module, "rollback_decision", lambda *a, **k: False)
    res = run_evolution(FAST, FakeEvaluator())
    assert not any(e["rolled_back"] for e in res["lineage"])


def test_plan_sizes_generations_from_the_measured_episode_time():
    cfg = EvolveConfig(pop=28)
    fast = make_plan(replace(cfg, gens=0), workers=10, t_ep=0.3)
    slow = make_plan(replace(cfg, gens=0), workers=10, t_ep=1.5)
    assert slow["gens"] <= fast["gens"] and 6 <= slow["gens"] and fast["gens"] <= 40
    assert slow["estimated_minutes"] <= 21 or slow["gens"] == 6
    assert make_plan(replace(cfg, gens=7), 10, 0.5)["gens"] == 7


# --- end to end (--quick), replay contract, determinism ---

@pytest.fixture(scope="module")
def quick_cli(tmp_path_factory):
    out = tmp_path_factory.mktemp("evo")
    rc = evolve_main(["--quick", "--workers", "3", "--out", str(out / "evolution"), "--replay", str(out / "replay.json")])
    assert rc == 0
    return out


def test_quick_lineage_is_deterministic_for_a_seed_whatever_the_worker_count(quick_cli):
    cfg = EvolveConfig.quick(0)
    with Evaluator(2, tuple(cfg.fitness_seeds) + tuple(cfg.gate_seeds)) as ev:
        again = run_evolution(cfg, ev)["lineage"]
    saved = json.loads((quick_cli / "evolution" / "lineage.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(again)) == saved and len(saved) == cfg.gens


def test_quick_outputs_champion_lineage_candidates_and_provenance(quick_cli):
    d = quick_cli / "evolution"
    champ = json.loads((d / "champion.json").read_text(encoding="utf-8"))
    assert set(champ["params"]) == {f.name for f in dataclasses.fields(SwarmParams)}
    prov = champ["provenance"]
    assert {"generation", "objective_a", "objective_b", "git_rev", "seed"} <= set(prov)
    assert len(prov["git_rev"]) == 40 or prov["git_rev"] == "unknown"
    lin = json.loads((d / "lineage.json").read_text(encoding="utf-8"))
    assert isinstance(lin, list) and all({"generation", "champion_id", "challenger", "promoted", "rolled_back",
                                          "strongest_attacker"} <= set(e) for e in lin)
    cands = json.loads((d / "candidates.json").read_text(encoding="utf-8"))
    assert all({"id", "parents", "params"} <= set(c) for c in cands)
    analysis = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
    assert {"tables", "attacker_evolution", "robustness_final_pool_train_b", "champion"} <= set(analysis)
    assert set(analysis["tables"]) == {"TRAIN_A", "TRAIN_B"}


def test_replay_json_follows_the_schema_version_1_contract(quick_cli):
    doc = json.loads((quick_cli / "replay.json").read_text(encoding="utf-8"))
    rx.validate_replay(doc)
    assert set(doc) == {"schema_version", "simulated", "synthetic_fixture", "generated_at", "objective", "results",
                        "evolution", "episode", "payouts"}
    assert doc["schema_version"] == 1 and doc["simulated"] is True and doc["synthetic_fixture"] is False
    assert doc["payouts"] == [] and isinstance(doc["generated_at"], str) and doc["generated_at"].endswith("+00:00")
    assert "leakage_pct" in doc["objective"] and "honest_block_rate" in doc["objective"]
    names = [r["defender"] for r in doc["results"]]
    assert names[:3] == ["always_release", "always_hold", "static_rules"]
    assert names[3].startswith("single_") and names[4:] == ["swarm_frozen", "swarm_l1", "swarm_evolved"]
    for r in doc["results"]:
        assert r["split"] == "TRAIN" and set(r) == set(rx.RESULT_KEYS)
        assert all(isinstance(r[k], (int, float)) for k in rx.RESULT_KEYS[2:])
        assert r["objective"] == pytest.approx(r["leakage_pct"] + 0.5 * r["honest_block_rate"] + r["honest_unpaid_rate"],
                                               abs=0.01)
    assert len(doc["evolution"]) == EvolveConfig.quick(0).gens
    for e in doc["evolution"]:
        assert set(e) == set(rx.EVOLUTION_KEYS) and isinstance(e["promoted"], bool)
        assert e["attacker_family"] in FAMILIES and set(e["weights"]) == set(NAMES)
    ep = doc["episode"]
    assert ep["seed"] == TRAIN_SEEDS[0] and ep["split"] == "TRAIN" and ep["defender"] == "swarm_evolved"
    ev0 = ep["events"][0]
    assert set(ev0) == set(rx.EVENT_KEYS) and set(ev0["signals"]) == {"scores", "risk", "top", "active"}
    assert set(ev0["weights"]) == set(NAMES) and ev0["verdict"] in {"RELEASE", "HOLD", "STEP_UP"}
    assert any(e["is_attack"] for e in ep["events"]) and {e["family"] for e in ep["events"]} >= {"honest", "ring"}
    assert all(len(repr(e["t"]).split(".")[-1]) <= 3 for e in ep["events"][:200])   # rounded to 3 decimals
    assert (quick_cli / "replay.json").stat().st_size < rx.MAX_BYTES


def test_replay_keeps_every_attack_event_and_samples_honest_ones_when_too_big():
    lin = [{"generation": 0, "champion_gate": {"objective": 10.0}, "challenger": None, "promoted": False,
            "strongest_attacker_family": "ring", "strongest_attacker": {"ring": {"leakage_pct": 5.0}},
            "champion_weights": {n: 0.25 for n in NAMES}}]
    full = rx.build_replay([], lin, default_params())
    n_attack = sum(e["is_attack"] for e in full["episode"]["events"])
    small = rx.build_replay([], lin, default_params(), max_bytes=400_000)
    ep = small["episode"]
    assert ep["honest_sample_every"] >= 2
    assert sum(e["is_attack"] for e in ep["events"]) == n_attack
    assert len(ep["events"]) < len(full["episode"]["events"])
    assert len(json.dumps(small, separators=(",", ":"))) <= 400_000
    assert "honest_sample_every" not in full["episode"]


# --- held-out isolation ---

FORBIDDEN = ("TEST_SEEN_SEEDS", "TEST_HELDOUT_SEEDS", "HELDOUT_FAMILIES", "sleeper", "TEST_SEEN", "TEST_HELDOUT")


def _evolve_sources():
    files = sorted((ROOT / "payguard" / "evolve").glob("*.py")) + [ROOT / "scripts" / "evolve.py"]
    assert len(files) >= 9
    return files


def test_evolution_code_never_references_the_held_out_data():
    for path in _evolve_sources():
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in text, f"{path.name} mentions {token}"
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                assert not 2000 <= node.value <= 3999, f"{path.name}: seed-like constant {node.value}"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "sleeper" not in node.value.lower()
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {a.name for a in node.names} | {getattr(node, "module", None) or ""}
                assert not any("TEST_" in n for n in names)


def test_evolution_only_ever_asks_for_train_seeds_and_keeps_fitness_and_gate_seeds_apart():
    ev = FakeEvaluator()
    run_evolution(replace(FAST, gens=3, fitness_seeds=TRAIN_A, gate_seeds=TRAIN_B), ev)
    assert ev.seeds <= set(TRAIN_SEEDS) and ev.seeds & set(TRAIN_A) and ev.seeds & set(TRAIN_B)
    pools = atk.initial_pools(np.random.default_rng(0), 2)
    for g in range(4):
        assert {sp[1] for sp in loop_module.fitness_specs(FAST, pools, g)} <= set(TRAIN_A)
        assert {sp[1] for sp in loop_module.gate_specs(FAST, pools, g)} <= set(TRAIN_B)
