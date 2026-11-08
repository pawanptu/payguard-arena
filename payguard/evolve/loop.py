"""The L2 evolution loop: defender GA + attacker co-evolution + champion/challenger gate with rollback.

One generation g:
  1. ATTACKERS   per TRAIN family: pool + mutants + fresh draws are played alone against the current champion
                 (TRAIN_A seeds); the strongest `pool_size` per family (by USD leaked) become the new pool.
  2. FITNESS     every defender candidate is evaluated on the fitness set = the protocol TRAIN_A episodes
                 + `arena_episodes` arena episodes built from the new pool on TRAIN_A seeds. In the same batch the
                 champion (and, right after a promotion, the champion it replaced) is evaluated on the GATE set =
                 protocol TRAIN_B episodes + `arena_episodes` arena episodes from the new pool on TRAIN_B seeds.
  3. ROLLBACK    if the champion was promoted in g-1 and now scores worse than its predecessor by more than
                 ROLLBACK_MARGIN on the new gate set, the predecessor returns (and the rolled-back candidate may
                 never be a challenger again).
  4. GATE        the best fitness candidate that is not the champion becomes the challenger; it is evaluated on the
                 gate set and promoted only if `gate.gate_decision` says so.
  5. BREED       elites (top `elites` + the champion) survive; the rest are children of tournament-selected
                 parents: uniform crossover, then a sparse mutation whose scale decays over the generations.
Everything is seeded from `cfg.seed`; results do not depend on the worker count. TRAIN seeds only.

DREAM HOOKS (optional, `run_evolution(..., hooks=engine)`; `hooks=None` leaves every number above unchanged):
  * OBSERVATION TIMING. The swarm only learns from episodes it has already played, with labels delivered. At
    generation g the world model sees `observe(...)` of the fitness-set episodes of generations < g, i.e. the
    protocol TRAIN_A episodes (the known training history, the only observations in generation 0) followed by the
    arena episodes of each earlier generation, de-duplicated, in the order they were played. Only the most recent
    `DreamLoopConfig.obs_window` of them are handed over. Attackers selected in generation g are never observed in g.
    The attacker pools are not i.i.d. (they co-evolve against the champion), so past episodes carry information about
    the next attackers; that information is what the world model is for.
  * DREAM FITNESS. `begin_generation` runs after attacker selection (step 1). Dream scores are computed for the
    population, the champion, and the predecessor (if any), so the champion that faces the gate is always scored.
  * SELECTION. Challenger choice, elites and tournaments rank by `hooks.blend(real, dream)`. The promotion gate
    and the rollback use REAL gate objectives only.
  * CALIBRATION. On promotion, predicted gain = dream(champion) - dream(challenger) and realized gain = real gate
    objective(champion) - objective(challenger) (positive = challenger better). `end_generation(None, None)` otherwise.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from ..dream.contract import DreamHooks, Observation, observe
from ..sim.metrics import Metrics
from ..swarm.objective import objective
from ..swarm.params import SwarmParams
from . import attackers as atk
from .evaluation import Evaluator, get_episode, proto_spec, swarm_job
from .ga import crossover, initial_population, mutate_sparse, mutation_scale, params_key, tournament
from .gate import gate_decision, rollback_decision
from .splits import FAMILIES, TRAIN_A, TRAIN_B

ROUND = 6


@dataclass(frozen=True)
class EvolveConfig:
    seed: int = 0
    pop: int = 20
    gens: int = 20
    elites: int = 3
    tournament_k: int = 3
    pool_size: int = 3              # attackers kept per family
    fresh_attackers: int = 1        # new uniform attacker draws per family and generation
    attacker_seeds: int = 2         # TRAIN_A seeds on which each attacker is scored against the champion
    arena_episodes: int = 3         # arena episodes added to the fitness set and to the gate set
    fitness_seeds: tuple[int, ...] = TRAIN_A
    gate_seeds: tuple[int, ...] = TRAIN_B
    mut_start: float = 0.25
    mut_end: float = 0.05
    atk_mut_start: float = 0.25
    atk_mut_end: float = 0.10
    budget_min: float = 28.0        # hard wall-clock stop: no new generation if it would likely overrun
    families: tuple[str, ...] = FAMILIES   # attacked families (leave-one-family-out uses one); TRAIN families only
    name: str = "full"

    def __post_init__(self) -> None:
        if not self.families or any(f not in FAMILIES for f in self.families) or len(set(self.families)) != len(self.families):
            raise ValueError(f"families must be distinct TRAIN families from {FAMILIES}, got {self.families}")

    @classmethod
    def quick(cls, seed: int = 0) -> "EvolveConfig":
        return cls(seed=seed, pop=4, gens=2, elites=1, tournament_k=2, pool_size=2, fresh_attackers=1,
                   attacker_seeds=1, arena_episodes=1, fitness_seeds=TRAIN_A[:2], gate_seeds=TRAIN_B[:2], name="quick")

    def fit_protocol_specs(self) -> list[tuple]:
        return [proto_spec(s, self.families) for s in self.fitness_seeds]

    def gate_protocol_specs(self) -> list[tuple]:
        return [proto_spec(s, self.families) for s in self.gate_seeds]


@dataclass(frozen=True)
class DreamLoopConfig:
    """How the dream hooks see the history: `obs_window` = number of most recent observations the world model fits on."""
    obs_window: int = 20

    def __post_init__(self) -> None:
        if self.obs_window < 1:
            raise ValueError("obs_window must be >= 1")


@dataclass
class Candidate:
    id: str
    params: SwarmParams
    generation: int
    parents: list[str]
    origin: str
    history: list[dict]


def r(x: float) -> float:
    return round(float(x), ROUND)


def rng_for(cfg: EvolveConfig, stage: int, g: int = 0) -> np.random.Generator:
    return np.random.default_rng([cfg.seed, stage, g])


def score_dict(m: Metrics) -> dict:
    return {"objective": r(objective(m)), "leakage_pct": r(m.leakage_pct), "honest_block_rate": r(m.honest_block_rate),
            "honest_unpaid_rate": r(m.honest_unpaid_rate)}


def fitness_specs(cfg: EvolveConfig, pools: atk.Pools, g: int) -> list[tuple]:
    return cfg.fit_protocol_specs() + atk.arena_specs(pools, cfg.fitness_seeds, g, cfg.arena_episodes, cfg.families)


def gate_specs(cfg: EvolveConfig, pools: atk.Pools, g: int) -> list[tuple]:
    return cfg.gate_protocol_specs() + atk.arena_specs(pools, cfg.gate_seeds, g, cfg.arena_episodes, cfg.families)


def make_plan(cfg: EvolveConfig, workers: int, t_ep: float, t_build: float = 0.25, budget_min: float = 20.0) -> dict:
    """Wall-time model from one measured per-episode run time (s, under full parallel load).
    If cfg.gens <= 0 the number of generations is sized to fit `budget_min` minutes."""
    W = max(1, workers)
    n_fit = len(cfg.fitness_seeds) + cfg.arena_episodes
    n_gate = len(cfg.gate_seeds) + cfg.arena_episodes
    n_atk_jobs = len(cfg.families) * (2 * cfg.pool_size + cfg.fresh_attackers)
    t_atk = math.ceil(n_atk_jobs / W) * cfg.attacker_seeds * (t_ep + t_build)
    t_fit = math.ceil((cfg.pop + 2) / W) * (n_fit * t_ep + cfg.arena_episodes * t_build)
    t_gate = n_gate * t_ep + cfg.arena_episodes * t_build
    per_gen = t_atk + t_fit + t_gate
    post = (2 * 20 + 2 * 10 + 4) * t_ep + 20            # train table (2 rounds of 20), A/B rows, robustness, replay run
    gens = cfg.gens
    if gens <= 0:
        gens = int(max(6, min(40, (budget_min * 60 - post) // per_gen)))
    return {"pop": cfg.pop, "gens": gens, "workers": W, "t_episode_s": round(t_ep, 3),
            "est_seconds_per_generation": round(per_gen, 1), "est_post_seconds": round(post, 1),
            "estimated_minutes": round((gens * per_gen + post) / 60, 1)}


def run_evolution(cfg: EvolveConfig, ev: Evaluator, log: Callable[[str], None] = lambda s: None,
                  clock: Callable[[], float] = time.time, *, hooks: DreamHooks | None = None,
                  loop_cfg: DreamLoopConfig = DreamLoopConfig()) -> dict:
    t0 = clock()
    A = list(cfg.fitness_seeds)
    registry: dict[str, Candidate] = {}

    pop: list[Candidate] = []
    for p, cid, origin in initial_population(cfg.pop, rng_for(cfg, 1)):
        c = Candidate(cid, p, 0, [], origin, [])
        registry[cid] = c
        pop.append(c)

    pools = atk.initial_pools(rng_for(cfg, 2), cfg.pool_size, cfg.families)
    initial_pools = {f: list(v) for f, v in pools.items()}
    champion = pop[0]                                  # the verified D-023 swarm
    champion_prov = {"generation": -1, "objective_a": None, "objective_b": None}
    prev: Candidate | None = None                      # predecessor of a champion promoted in the last generation
    prev_prov: dict | None = None
    blacklist: set[tuple] = set()                      # params keys of rolled-back champions
    lineage: list[dict] = []
    gen_seconds: list[float] = []
    stopped_early = False

    # Dream observations (only with hooks): the fitness-set episodes played so far, in play order, de-duplicated.
    # Generation 0 knows only the protocol TRAIN_A episodes; each generation's fitness set joins AFTER it was played.
    history: list[Observation] = []
    seen_specs: set[tuple] = set()

    def learn_from(specs: list[tuple]) -> None:
        for sp in specs:
            if sp not in seen_specs:
                seen_specs.add(sp)
                history.append(observe(get_episode(sp)))

    if hooks is not None:
        learn_from(cfg.fit_protocol_specs())

    for g in range(cfg.gens):
        tg = clock()
        elapsed = tg - t0
        if gen_seconds and elapsed + 1.25 * max(gen_seconds) > cfg.budget_min * 60:
            stopped_early = True
            log(f"  budget guard: stopping before generation {g} (elapsed {elapsed / 60:.1f} min)")
            break

        # 1. attackers vs the current champion
        scale = mutation_scale(g, cfg.gens, cfg.atk_mut_start, cfg.atk_mut_end)
        cands = atk.candidate_attackers(pools, rng_for(cfg, 3, g), scale, cfg.fresh_attackers, cfg.families)
        atk_seeds = [A[(g * cfg.attacker_seeds + j) % len(A)] for j in range(cfg.attacker_seeds)]
        atk_res = ev.run(atk.scoring_jobs(champion.params, cands, atk_seeds))
        pools, atk_records = atk.select_strongest(cands, atk_res, cfg.pool_size, cfg.families)
        scored_against = champion.id
        if hooks is not None:
            # the world model sees only generations < g (the fitness set of g is appended below, after it was played)
            hooks.begin_generation(g, history[-loop_cfg.obs_window:])

        # 2. fitness of every candidate + gate score of the champion (and of its predecessor)
        fit_sp, gate_sp = fitness_specs(cfg, pools, g), gate_specs(cfg, pools, g)
        gate_subjects = [champion] + ([prev] if prev is not None else [])
        jobs = [swarm_job(c.params, fit_sp) for c in pop] + [swarm_job(c.params, gate_sp) for c in gate_subjects]
        res = ev.run(jobs)
        fit_m = [m for m, _ in res[:len(pop)]]
        gate_res = res[len(pop):]
        fits = [objective(m) for m in fit_m]
        for c, m in zip(pop, fit_m):
            c.history.append({"generation": g, **score_dict(m)})
        dream: list[float] | None = None
        champ_dream_i = len(pop)                       # index in `dream` of the champion that faces the gate
        if hooks is not None:
            learn_from(fit_sp)
            dream = hooks.dream_fitness([c.params for c in pop] + [champion.params]
                                        + ([prev.params] if prev is not None else []), ev)
            blended = hooks.blend(fits, dream[:len(pop)])
        else:
            blended = fits                             # selection ranks by the real fitness

        # 3. rollback check
        rolled_back, rollback = False, None
        champ_m, champ_w = gate_res[0]
        if prev is not None:
            prev_m, prev_w = gate_res[1]
            if rollback_decision(objective(prev_m), objective(champ_m)):
                rolled_back = True
                rollback = {"from": champion.id, "to": prev.id, "from_gate_objective": r(objective(champ_m)),
                            "to_gate_objective": r(objective(prev_m))}
                blacklist.add(params_key(champion.params))
                champion, champion_prov = prev, prev_prov
                champ_m, champ_w = prev_m, prev_w
                champ_dream_i = len(pop) + 1           # the predecessor's dream score sits right after the champion's
                log(f"  gen {g}: ROLLBACK {rollback['from']} -> {rollback['to']}")
            prev, prev_prov = None, None

        # 4. challenger and gate (the gate uses real objectives only)
        champ_key = params_key(champion.params)
        order = sorted(range(len(pop)), key=lambda i: (blended[i], i))
        challenger_i = next((i for i in order if params_key(pop[i].params) not in blacklist
                             and params_key(pop[i].params) != champ_key), None)
        challenger = pop[challenger_i] if challenger_i is not None else None
        promoted, ch_entry, ch_m = False, None, None
        if challenger is not None:
            ch_m, _ = ev.run([swarm_job(challenger.params, gate_sp)])[0]
            promoted = gate_decision(objective(champ_m), objective(ch_m), champ_m.honest_block_rate,
                                     ch_m.honest_block_rate)
            ch_entry = {"id": challenger.id, "params": asdict(challenger.params),
                        "objective_a": r(fits[challenger_i]), "objective_b": r(objective(ch_m)),
                        "gate": score_dict(ch_m)}
        champion_before = champion
        if hooks is not None:
            # gain convention (engine.predicted_gain): champion minus challenger, positive = challenger better
            if promoted:
                hooks.end_generation(dream[champ_dream_i] - dream[challenger_i], objective(champ_m) - objective(ch_m))
            else:
                hooks.end_generation(None, None)
        if promoted:
            prev, prev_prov = champion, champion_prov
            champion = challenger
            champion_prov = {"generation": g, "objective_a": r(fits[challenger_i]), "objective_b": r(objective(ch_m))}
            log(f"  gen {g}: PROMOTED {challenger.id} (gate {objective(ch_m):.2f} vs champion {objective(champ_m):.2f})")

        strongest = {f: atk_records[f][0] for f in cfg.families}
        top_fam = max(cfg.families, key=lambda f: strongest[f]["leaked_usd"])
        entry = {
            "generation": g,
            "champion_id": champion_before.id,
            "champion_gate": score_dict(champ_m),
            "challenger": ch_entry,
            "promoted": promoted,
            "rolled_back": rolled_back,
            "rollback": rollback,
            "champion_id_after": champion.id,
            "attackers_scored_against": scored_against,
            "strongest_attacker": strongest,
            "strongest_attacker_family": top_fam,
            "attacker_pools": atk_records,
            "champion_weights": {k: r(v) for k, v in champ_w.items()},
            "fitness_best": r(min(fits)), "fitness_mean": r(sum(fits) / len(fits)),
        }
        if hooks is not None:
            entry["dream"] = {**hooks.record(),
                              "real_fitness": [r(x) for x in fits],
                              "blended_fitness": [r(x) for x in blended],
                              "dream_fitness": [r(x) for x in dream[:len(pop)]]}
        lineage.append(entry)

        # 5. breed the next generation
        if g < cfg.gens - 1:
            rng = rng_for(cfg, 5, g)
            sc = mutation_scale(g + 1, cfg.gens, cfg.mut_start, cfg.mut_end)
            elite_idx = [i for i in order if params_key(pop[i].params) not in blacklist][:cfg.elites]
            champ_idx = next((i for i, c in enumerate(pop) if c.id == champion.id), None)
            if champ_idx is not None and champ_idx not in elite_idx:
                elite_idx.append(champ_idx)
            nxt = [pop[i] for i in elite_idx]
            for j in range(cfg.pop - len(nxt)):
                ia, ib = tournament(blended, cfg.tournament_k, rng), tournament(blended, cfg.tournament_k, rng)
                child = mutate_sparse(crossover(pop[ia].params, pop[ib].params, rng), rng, sc)
                c = Candidate(f"g{g + 1}-{j}", child, g + 1, [pop[ia].id, pop[ib].id], "offspring", [])
                registry[c.id] = c
                nxt.append(c)
            pop = nxt
        gen_seconds.append(clock() - tg)
        ch_obj = objective(ch_m) if ch_m is not None else float("nan")
        log(f"  gen {g:2d} {gen_seconds[-1]:5.1f}s  champion {champion.id:8s} gate {objective(champ_m):6.2f}"
            f"  challenger {challenger.id if challenger else '-':8s} {ch_obj:6.2f}  best fitness {min(fits):6.2f}"
            f"  top attacker {top_fam} {strongest[top_fam]['leakage_pct']:5.1f}%")

    return {
        "lineage": lineage,
        "candidates": [{"id": c.id, "generation": c.generation, "parents": c.parents, "origin": c.origin,
                        "params": asdict(c.params), "history": c.history} for c in registry.values()],
        "champion": {"id": champion.id, "params": asdict(champion.params), **champion_prov},
        "initial_pools": initial_pools,
        "final_pools": pools,
        "stopped_early": stopped_early,
        "generations_run": len(lineage),
        "seconds": clock() - t0,
        "gen_seconds": gen_seconds,
    }
