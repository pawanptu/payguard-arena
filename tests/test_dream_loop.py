"""Dream hooks in the L2 loop (P5): default path unchanged, family-restricted evolution, hook order, observation
timing, real gate, calibration signs. TRAIN seeds only. Most tests use a fake evaluator (no simulation)."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import payguard.evolve.loop as loop_module
from payguard.dream.contract import DreamConfig, DreamHooks, Observation, observe
from payguard.evolve import attackers as atk
from payguard.evolve.evaluation import Evaluator, get_episode, proto_spec
from payguard.evolve.loop import DreamLoopConfig, EvolveConfig, run_evolution
from payguard.evolve.splits import FAMILIES, TRAIN_A, TRAIN_B
from payguard.protocol import TRAIN_FAMILIES
from payguard.sim.attacks import RingParams, StructuringParams
from payguard.sim.metrics import Metrics
from payguard.swarm.params import SwarmParams, default_params

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "evolve_quick_base_f4f922d.json"

SMALL = EvolveConfig(seed=1, pop=4, gens=3, elites=1, tournament_k=2, pool_size=1, fresh_attackers=1,
                     attacker_seeds=1, arena_episodes=1, fitness_seeds=TRAIN_A[:2], gate_seeds=TRAIN_B[:2], name="small")


def _metrics(leak: float, block: float = 5.0) -> Metrics:
    return Metrics(n_episodes=1, attack_usd=100000.0, leaked_usd=leak * 1000.0, leakage_pct=leak,
                   leakage_pct_by_family={}, honest_n=1000, honest_block_rate=block, honest_unpaid_rate=0.0,
                   honest_median_ttp_h=0.0, honest_p95_ttp_h=0.0, immunity_attempts=0.0, never_contained_rate=0.0,
                   llm_calls=0)


def real_obj(pd: dict) -> float:
    """Real objective of a candidate: the abstain gene helps by 10 points (clearly better challengers exist)."""
    return 20.0 - 10.0 * pd["l1_abstain"]


class RecEval:
    """Fake evaluator: objective = real_obj(params) (+ a tiny spec-dependent noise), block 5 %. Logs every batch as "ev"
    in `log` (a shared list with the hooks), and keeps every spec it was asked to play."""

    def __init__(self, log=None, block_of=lambda pd: 5.0):
        self.log = log if log is not None else []
        self.block_of = block_of
        self.specs: list[tuple] = []

    def run(self, jobs):
        self.log.append("ev")
        out = []
        for job in jobs:
            assert job[0] == "swarm"
            pd, specs = job[1], job[3]
            self.specs += list(specs)
            out.append((_metrics(real_obj(pd) + 0.001, self.block_of(pd)), {"graph": 0.25}))
        return out


class RecordingHooks:
    """Implements DreamHooks. Dream objective = the real objective of the candidate (so predicted == realized)."""

    def __init__(self, log, dream_of=None):
        self.log = log
        self.dream_of = dream_of or (lambda p: real_obj(p.__dict__))
        self.begins: list[tuple[int, list]] = []
        self.ends: list[tuple] = []
        self.dreams: list[list] = []
        self.blends: list[tuple] = []
        self.records = 0

    def begin_generation(self, g, observations):
        self.log.append("begin")
        self.begins.append((g, list(observations)))

    def dream_fitness(self, candidates, evaluator):
        self.log.append("dream")
        out = [float(self.dream_of(c)) for c in candidates]
        self.dreams.append(out)
        return out

    def blend(self, real_fitness, dream_fitness):
        self.log.append("blend")
        self.blends.append((list(real_fitness), list(dream_fitness)))
        return [(r + d) / 2 for r, d in zip(real_fitness, dream_fitness)]

    def end_generation(self, predicted_gain, realized_gain):
        self.log.append("end")
        self.ends.append((predicted_gain, realized_gain))

    def record(self):
        self.log.append("record")
        self.records += 1
        return {"g": len(self.ends) - 1, "lam": 0.3}


class AbsurdHooks(RecordingHooks):
    """Dreams that say the opposite of reality: the candidate that is worst in reality looks best, by a huge margin."""

    def __init__(self, log):
        super().__init__(log, dream_of=lambda p: -1e6 * real_obj(p.__dict__))

    def blend(self, real_fitness, dream_fitness):
        self.log.append("blend")
        return list(dream_fitness)                   # lam = 1: selection by dreams only


# --- default path --------------------------------------------------------------------------------------------

def test_default_families_and_specs_are_the_old_ones():
    cfg = EvolveConfig()
    assert cfg.families == FAMILIES == TRAIN_FAMILIES == ("ring", "structuring")
    assert proto_spec(1005) == ("proto", 1005) == proto_spec(1005, TRAIN_FAMILIES)
    assert [sp for sp in cfg.fit_protocol_specs()] == [("proto", s) for s in cfg.fitness_seeds]
    pools = atk.initial_pools(np.random.default_rng(3), 2)
    assert atk.arena_specs(pools, TRAIN_A, 4, 3) == atk.arena_specs(pools, TRAIN_A, 4, 3, FAMILIES)
    assert loop_module.fitness_specs(cfg, pools, 2) == loop_module.fitness_specs(replace(cfg, families=FAMILIES), pools, 2)


def test_default_quick_lineage_is_byte_for_byte_the_base_commit_lineage():
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cfg = EvolveConfig.quick(0)
    with Evaluator(1) as ev:
        res = run_evolution(cfg, ev)
    now = json.loads(json.dumps({"lineage": res["lineage"], "champion": res["champion"],
                                 "candidate_ids": [c["id"] for c in res["candidates"]],
                                 "gen_count": res["generations_run"]}))
    assert now == base
    assert all("dream" not in e for e in now["lineage"])


def test_invalid_families_are_refused():
    with pytest.raises(ValueError):
        EvolveConfig(families=("sleeper",))
    with pytest.raises(ValueError):
        EvolveConfig(families=())
    with pytest.raises(ValueError):
        EvolveConfig(families=("ring", "ring"))
    with pytest.raises(ValueError):
        DreamLoopConfig(obs_window=0)


# --- family-restricted evolution ----------------------------------------------------------------------------

def test_family_restricted_evolution_builds_only_the_requested_family():
    cfg = replace(SMALL, families=("ring",))
    assert cfg.fit_protocol_specs() == [("proto", s, ("ring",)) for s in TRAIN_A[:2]]
    ep = get_episode(("proto", TRAIN_A[0], ("ring",)))
    assert {t.family for t in ep.truth.values() if t.is_attack} <= {"ring"} and ep.truth
    pools = atk.initial_pools(np.random.default_rng(0), 2, ("ring",))
    assert set(pools) == {"ring"} and all(isinstance(p, RingParams) for p in pools["ring"])
    specs = atk.arena_specs(pools, TRAIN_A, 0, 2, ("ring",))
    assert all([fam for fam, _ in s[2]] == ["ring"] for s in specs)

    ev = RecEval()
    res = run_evolution(cfg, ev)
    assert set(res["lineage"][0]["strongest_attacker"]) == {"ring"}
    assert all(set(e["attacker_pools"]) == {"ring"} and e["strongest_attacker_family"] == "ring" for e in res["lineage"])
    assert set(res["initial_pools"]) == set(res["final_pools"]) == {"ring"}
    arena = [s for s in ev.specs if s[0] == "arena"]
    proto = [s for s in ev.specs if s[0] == "proto"]
    assert arena and proto
    assert all({fam for fam, _ in s[2]} == {"ring"} for s in arena)
    assert all(s == ("proto", s[1], ("ring",)) for s in proto)
    assert not any(isinstance(p, StructuringParams) for v in res["final_pools"].values() for p in v)


def test_structuring_only_evolution_attacks_structuring_only():
    res = run_evolution(replace(SMALL, families=("structuring",), gens=1), RecEval())
    assert set(res["final_pools"]) == {"structuring"}
    assert res["lineage"][0]["strongest_attacker_family"] == "structuring"


# --- dream hooks: order, timing, selection, calibration -----------------------------------------------------

def test_hooks_are_called_in_the_documented_order():
    log: list[str] = []
    hooks = RecordingHooks(log)
    run_evolution(SMALL, RecEval(log), hooks=hooks)
    assert isinstance(hooks, DreamHooks)
    # log segments: one per generation, starting right after the previous generation's record
    segs, cur = [], []
    for tok in log:
        cur.append(tok)
        if tok == "record":
            segs.append(cur)
            cur = []
    assert not cur and len(segs) == SMALL.gens
    for seg in segs:
        assert seg[0] == "ev" and seg[1] == "begin", seg           # attackers scored, then begin_generation
        i_dream = seg.index("dream")
        assert seg.index("begin") < i_dream < seg.index("blend") < seg.index("end") < seg.index("record")
        assert seg[2:i_dream].count("ev") == 1                     # the fitness+gate batch comes before dreams
        assert seg[i_dream - 1] == "ev" and set(seg[seg.index("blend") + 1:seg.index("end")]) <= {"ev"}
        assert seg.count("begin") == seg.count("dream") == seg.count("blend") == seg.count("end") == 1
    assert [g for g, _ in hooks.begins] == list(range(SMALL.gens))
    assert hooks.records == SMALL.gens


def test_observations_at_g_are_the_most_recent_episodes_played_before_g(monkeypatch):
    """Every observe() call is logged with the number of begin_generation calls made so far. The observations handed
    to generation g must be the last `obs_window` of those made before its begin_generation: nothing from generation g
    (its fitness set is observed only after it was played) and nothing from later generations."""
    created: list[Observation] = []                 # observe() outputs in call order
    real_observe = loop_module.observe

    def spy(ep):
        o = real_observe(ep)
        created.append(o)
        return o

    monkeypatch.setattr(loop_module, "observe", spy)
    window = 3
    cfg = replace(SMALL, gens=3)
    before_begin: list[list[Observation]] = []
    hooks = RecordingHooks([])
    orig_begin = hooks.begin_generation

    def begin(g, observations):
        before_begin.append(list(created))
        orig_begin(g, observations)

    hooks.begin_generation = begin
    run_evolution(cfg, RecEval(hooks.log), hooks=hooks, loop_cfg=DreamLoopConfig(obs_window=window))
    assert len(hooks.begins) == cfg.gens
    for (g, obs), seen in zip(hooks.begins, before_begin):
        assert len(obs) == min(window, len(seen))
        assert all(a is b for a, b in zip(obs, seen[len(seen) - len(obs):]))
    # generation 0: the protocol TRAIN_A history only, nothing else has been played
    g0_obs = hooks.begins[0][1]
    assert [o.seed for o in g0_obs] == list(cfg.fitness_seeds) and all(o.seed in TRAIN_A for o in g0_obs)
    assert len(before_begin[0]) == len(cfg.fitness_seeds)
    # from generation 1 on, the arena episodes of earlier generations have joined the history (window permitting)
    assert len(before_begin[1]) > len(before_begin[0])


def test_calibration_gets_predicted_and_realized_gain_only_on_promotion():
    log: list[str] = []
    hooks = RecordingHooks(log)
    res = run_evolution(SMALL, RecEval(log), hooks=hooks)
    promoted = [e["promoted"] for e in res["lineage"]]
    assert any(promoted)
    for e, (pred, real) in zip(res["lineage"], hooks.ends):
        if e["promoted"]:
            # predicted (dreams) and realized (real gate) gains: champion minus challenger, both lower-is-better;
            # here dreams equal the real objective, so they must agree and be positive
            assert pred == pytest.approx(real) and real > 0
            assert real == pytest.approx(e["champion_gate"]["objective"] - e["challenger"]["gate"]["objective"], abs=1e-3)
        else:
            assert (pred, real) == (None, None)
    assert "dream" in res["lineage"][0] and res["lineage"][0]["dream"]["real_fitness"]
    assert len(res["lineage"][0]["dream"]["blended_fitness"]) == SMALL.pop


def test_gate_uses_the_real_objective_whatever_the_dreams_say():
    log: list[str] = []
    res = run_evolution(SMALL, RecEval(log), hooks=AbsurdHooks(log))
    for e in res["lineage"]:
        if e["promoted"]:
            ch, ch_gate = e["challenger"], e["challenger"]["gate"]
            assert ch_gate["objective"] <= e["champion_gate"]["objective"] - 0.5           # real MIN_GAIN
            assert ch_gate["honest_block_rate"] <= e["champion_gate"]["honest_block_rate"] + 1.0
    assert any(e["challenger"] is not None for e in res["lineage"])


def test_no_real_gain_means_no_promotion_however_absurd_the_dreams():
    class Flat(RecEval):
        def run(self, jobs):
            self.log.append("ev")
            return [(_metrics(20.0), {"graph": 0.25}) for _ in jobs]

    log: list[str] = []
    hooks = AbsurdHooks(log)
    res = run_evolution(replace(SMALL, gens=3), Flat(log), hooks=hooks)
    assert not any(e["promoted"] or e["rolled_back"] for e in res["lineage"])
    assert res["champion"]["id"] == "default"
    assert hooks.ends == [(None, None)] * 3


def test_the_rollback_rule_still_uses_real_objectives(monkeypatch):
    monkeypatch.setattr(loop_module, "rollback_decision", lambda prev, champ, **k: False)
    log: list[str] = []
    res = run_evolution(replace(SMALL, gens=3), RecEval(log), hooks=AbsurdHooks(log))
    assert not any(e["rolled_back"] for e in res["lineage"])


def test_dream_candidates_include_champion_and_predecessor_after_promotion():
    log: list[str] = []
    hooks = RecordingHooks(log)
    res = run_evolution(replace(SMALL, gens=3), RecEval(log), hooks=hooks)
    sizes = [len(d) for d in hooks.dreams]
    assert sizes[0] == SMALL.pop + 1                       # population + champion
    assert sizes[1] == SMALL.pop + 1 + (1 if res["lineage"][0]["promoted"] else 0)
