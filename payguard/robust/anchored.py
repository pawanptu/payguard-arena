"""C1: prior-anchored evolution of `SwarmParams` (MAP estimate with a quadratic prior at the default swarm).

Fitness (lower is better) on the FIT episodes:

    objective + lam * d(theta, default) + 100 * max(0, honest_block - honest_block_of_static_rules)

d = RMS over genes of the deviation from the default in units of the gene's BOUNDS range. A gene that the fit
episodes give no evidence about (a lens that is silent on the fit attacks) can only move if the honest friction it
saves outweighs lam * d, so by default it stays at its prior instead of being loosened. The default swarm is in the
first population, so the evolved result is never worse than the default on the fit fitness (elitism). Operators are
the existing ones of `payguard.evolve.ga`; the loop is a plain GA (no attacker co-evolution, no gate): the
out-of-family protocol is the gate here.
"""
from __future__ import annotations

from dataclasses import asdict, fields

import numpy as np

from ..evolve.ga import crossover, initial_population, mutate_sparse, mutation_scale, params_key, tournament
from ..swarm.objective import objective
from ..swarm.params import BOUNDS, SwarmParams, default_params
from .folds import Runner, Spec

BLOCK_PENALTY = 100.0
POP, GENS, ELITES, TOURNAMENT_K = 20, 10, 3, 3
MUT_START, MUT_END = 0.25, 0.05


def prior_distance(p: SwarmParams, prior: SwarmParams | None = None) -> float:
    prior = prior if prior is not None else default_params()
    dev = []
    for f in fields(p):
        lo, hi = BOUNDS[f.name]
        dev.append((getattr(p, f.name) - getattr(prior, f.name)) / (hi - lo))
    return float(np.sqrt(np.mean(np.square(dev))))


def fitness(m, p: SwarmParams, lam: float, static_block: float) -> float:
    return (objective(m) + lam * prior_distance(p)
            + BLOCK_PENALTY * max(0.0, m.honest_block_rate - static_block))


def evolve(runner: Runner, fit_specs: tuple[Spec, ...], lam: float, seed: int, tag: int,
           pop: int = POP, gens: int = GENS, log=lambda s: None) -> dict:
    """GA on `fit_specs`. Deterministic given (seed, tag, fit_specs); independent of the worker count."""
    static_block = runner.metrics({"kind": "static"}, fit_specs).honest_block_rate
    rng0 = np.random.default_rng([seed, tag, 1])
    population = [p for p, _, _ in initial_population(pop, rng0)]
    seen: dict[tuple, dict] = {}
    trace = []
    for g in range(gens):
        todo = {}
        for p in population:
            k = params_key(p)
            if k not in seen and k not in todo:
                todo[k] = p
        if todo:
            res = runner.map([("metrics", {"kind": "swarm", "params": asdict(p), "learn": True}, fit_specs)
                              for p in todo.values()])
            for (k, p), m in zip(todo.items(), res):
                seen[k] = {"params": p, "fit": fitness(m, p, lam, static_block), "obj": objective(m),
                           "block": m.honest_block_rate, "leak": m.leakage_pct, "d": prior_distance(p)}
        fits = [seen[params_key(p)]["fit"] for p in population]
        order = sorted(range(len(population)), key=lambda i: (fits[i], i))
        top = seen[params_key(population[order[0]])]
        trace.append({"gen": g, "best_fit": fits[order[0]], "mean_fit": float(np.mean(fits)),
                      "best_obj": top["obj"], "best_d": top["d"]})
        log(f"    gen {g:2d} best fit {fits[order[0]]:7.2f} (obj {top['obj']:6.2f} d {top['d']:.3f})")
        if g == gens - 1:
            break
        rng = np.random.default_rng([seed, tag, 5, g])
        sc = mutation_scale(g + 1, gens, MUT_START, MUT_END)
        nxt = [population[i] for i in order[:ELITES]]
        while len(nxt) < pop:
            a, b = tournament(fits, TOURNAMENT_K, rng), tournament(fits, TOURNAMENT_K, rng)
            nxt.append(mutate_sparse(crossover(population[a], population[b], rng), rng, sc))
        population = nxt
    best = min(seen.values(), key=lambda r: (r["fit"], params_key(r["params"])))
    return {"params": best["params"], "fit": best["fit"], "obj_fit_set": best["obj"], "block_fit_set": best["block"],
            "d": best["d"], "static_block_fit_set": static_block, "trace": trace, "evaluated": len(seen)}
