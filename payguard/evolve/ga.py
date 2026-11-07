"""Defender genetic algorithm operators over `SwarmParams` (all deterministic given the Generator)."""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from ..swarm.params import INT_FIELDS, SwarmParams, default_params, mutate, sample_params

GENE_RATE = 0.3   # share of genes a mutation actually moves (params.mutate jitters ALL of them; we keep ~30%)


def params_key(p: SwarmParams) -> tuple:
    return tuple(getattr(p, f.name) for f in fields(p))


def crossover(a: SwarmParams, b: SwarmParams, rng: np.random.Generator) -> SwarmParams:
    """Uniform crossover: every gene comes from either parent with probability 1/2."""
    pick = rng.random(len(fields(a))) < 0.5
    return SwarmParams(**{f.name: getattr(a if pick[i] else b, f.name) for i, f in enumerate(fields(a))})


def mutate_sparse(p: SwarmParams, rng: np.random.Generator, scale: float, gene_rate: float = GENE_RATE) -> SwarmParams:
    """`params.mutate` proposes a move for every gene; each gene keeps the proposal with probability gene_rate
    (at least one gene always moves), otherwise it stays at the parent's value."""
    proposal = mutate(p, rng, scale)
    flags = rng.random(len(fields(p))) < gene_rate
    if not flags.any():
        flags[int(rng.integers(0, len(flags)))] = True
    return replace(p, **{f.name: getattr(proposal, f.name) for i, f in enumerate(fields(p)) if flags[i]})


def tournament(fitness: list[float], k: int, rng: np.random.Generator) -> int:
    """Index of the best (lowest objective) of k random contestants (ties: lower index)."""
    idx = rng.integers(0, len(fitness), size=k)
    return int(min(idx, key=lambda i: (fitness[i], i)))


def mutation_scale(generation: int, generations: int, start: float, end: float) -> float:
    """Linear decay from `start` (generation 0) to `end` (last generation)."""
    if generations <= 1:
        return start
    return start + (end - start) * generation / (generations - 1)


def initial_population(size: int, rng: np.random.Generator) -> list[tuple[SwarmParams, str, str]]:
    """(params, id, origin). Always: the default swarm and the default with the abstention gene on; then
    mutants of the default (about half of the rest) and fresh uniform draws."""
    d = default_params()
    pop = [(d, "default", "default"), (replace(d, l1_abstain=1), "g0-1", "default+l1_abstain")]
    base = len(pop)
    n_mut = (size - base + 1) // 2
    for i in range(base, size):
        if i - base < n_mut:
            pop.append((mutate_sparse(d, rng, 0.2), f"g0-{i}", "mutant_of_default"))
        else:
            pop.append((sample_params(rng), f"g0-{i}", "fresh_sample"))
    return pop[:size]


def int_genes_ok(p: SwarmParams) -> bool:
    return all(isinstance(getattr(p, n), int) for n in INT_FIELDS)
