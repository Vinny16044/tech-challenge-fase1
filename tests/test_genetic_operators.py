import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from genetic.encoding import create_individual, create_population, validate_individual
from genetic.operators import (
    tournament_selection,
    roulette_selection,
    uniform_crossover,
    mutate,
)


def test_create_population_returns_valid_individuals():
    population = create_population(
        model_name="Random Forest",
        population_size=5,
        seed=42,
    )

    assert len(population) == 5
    assert all(validate_individual("Random Forest", individual) for individual in population)


def test_create_population_rejects_invalid_size():
    with pytest.raises(ValueError):
        create_population(
            model_name="Random Forest",
            population_size=0,
            seed=42,
        )


def test_tournament_selection_returns_valid_individual():
    individual_1 = create_individual("Árvore de Decisão")
    individual_2 = create_individual("Árvore de Decisão")

    evaluated_population = [
        (individual_1, 0.70),
        (individual_2, 0.95),
    ]

    selected = tournament_selection(
        evaluated_population=evaluated_population,
        tournament_size=2,
        seed=42,
    )

    assert validate_individual("Árvore de Decisão", selected)


def test_roulette_selection_returns_valid_individual():
    population = create_population("KNN", population_size=3, seed=42)
    evaluated_population = [
        (population[0], 0.10),
        (population[1], 0.80),
        (population[2], 0.30),
    ]

    selected = roulette_selection(
        evaluated_population=evaluated_population,
        seed=42,
    )

    assert validate_individual("KNN", selected)


def test_uniform_crossover_preserves_genes():
    parent_1 = create_individual("Regressão Logística")
    parent_2 = create_individual("Regressão Logística")

    child_1, child_2 = uniform_crossover(
        parent_1=parent_1,
        parent_2=parent_2,
        crossover_rate=1.0,
        seed=42,
    )

    assert set(child_1.keys()) == set(parent_1.keys())
    assert set(child_2.keys()) == set(parent_2.keys())
    assert validate_individual("Regressão Logística", child_1)
    assert validate_individual("Regressão Logística", child_2)


def test_mutate_returns_valid_individual():
    individual = create_individual("Random Forest")

    mutated = mutate(
        model_name="Random Forest",
        individual=individual,
        mutation_rate=1.0,
        seed=42,
    )

    assert validate_individual("Random Forest", mutated)


def test_mutate_rejects_invalid_rate():
    individual = create_individual("Random Forest")

    with pytest.raises(ValueError):
        mutate(
            model_name="Random Forest",
            individual=individual,
            mutation_rate=1.5,
            seed=42,
        )
