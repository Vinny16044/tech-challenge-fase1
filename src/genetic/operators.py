"""
Operadores genéticos:
- seleção por torneio;
- seleção por roleta;
- crossover uniforme;
- mutação.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

try:
    from .encoding import get_search_space, validate_individual
except ImportError:
    from encoding import get_search_space, validate_individual


Individual = Dict[str, Any]
EvaluatedIndividual = Tuple[Individual, float]


def tournament_selection(
    evaluated_population: List[EvaluatedIndividual],
    tournament_size: int = 3,
    seed: int | None = None,
) -> Individual:
    """
    Seleciona um indivíduo usando torneio.

    O operador sorteia alguns indivíduos e escolhe o de maior fitness.
    """
    if not evaluated_population:
        raise ValueError("A população avaliada não pode estar vazia.")

    rng = random.Random(seed)
    tournament_size = min(tournament_size, len(evaluated_population))

    competitors = rng.sample(evaluated_population, tournament_size)
    winner = max(competitors, key=lambda item: item[1])

    return winner[0].copy()


def roulette_selection(
    evaluated_population: List[EvaluatedIndividual],
    seed: int | None = None,
) -> Individual:
    """
    Seleciona um indivíduo por roleta proporcional ao fitness.
    """
    if not evaluated_population:
        raise ValueError("A população avaliada não pode estar vazia.")

    rng = random.Random(seed)
    total_fitness = sum(max(fitness, 0) for _, fitness in evaluated_population)

    if total_fitness == 0:
        return rng.choice(evaluated_population)[0].copy()

    pick = rng.uniform(0, total_fitness)
    current = 0.0

    for individual, fitness in evaluated_population:
        current += max(fitness, 0)
        if current >= pick:
            return individual.copy()

    return evaluated_population[-1][0].copy()


def uniform_crossover(
    parent_1: Individual,
    parent_2: Individual,
    crossover_rate: float = 0.8,
    seed: int | None = None,
) -> Tuple[Individual, Individual]:
    """
    Realiza crossover uniforme entre dois pais.
    """
    if set(parent_1.keys()) != set(parent_2.keys()):
        raise ValueError("Os pais devem possuir os mesmos genes.")

    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate deve estar entre 0 e 1.")

    rng = random.Random(seed)

    child_1 = parent_1.copy()
    child_2 = parent_2.copy()

    if rng.random() > crossover_rate:
        return child_1, child_2

    for gene in parent_1.keys():
        if rng.random() < 0.5:
            child_1[gene], child_2[gene] = child_2[gene], child_1[gene]

    return child_1, child_2


def mutate(
    model_name: str,
    individual: Individual,
    mutation_rate: float = 0.1,
    seed: int | None = None,
) -> Individual:
    """
    Aplica mutação nos genes do indivíduo.
    """
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate deve estar entre 0 e 1.")

    if not validate_individual(model_name, individual):
        raise ValueError("Indivíduo inválido para o modelo informado.")

    rng = random.Random(seed)
    search_space = get_search_space(model_name)
    mutated = individual.copy()

    for gene, possible_values in search_space.items():
        if rng.random() < mutation_rate:
            alternative_values = [
                value for value in possible_values
                if value != mutated[gene]
            ]

            if alternative_values:
                mutated[gene] = rng.choice(alternative_values)

    return mutated