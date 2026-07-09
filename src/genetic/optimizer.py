"""
Loop principal do Algoritmo Genético para otimização de hiperparâmetros.

Este módulo executa as gerações do AG:
- cria população inicial;
- avalia fitness com validação cruzada;
- aplica seleção;
- aplica crossover;
- aplica mutação;
- preserva elites;
- retorna o melhor conjunto de hiperparâmetros encontrado.

Responsável: Thamy Lais / Paola
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from sklearn.base import BaseEstimator

from genetic.encoding import create_population
from genetic.operators import (
    tournament_selection,
    roulette_selection,
    uniform_crossover,
    mutate,
)
from genetic.fitness import evaluate_individual

from models import create_model_by_name
from evaluation import evaluate_model


Individual = Dict[str, Any]
EvaluatedIndividual = Tuple[Individual, float]


@dataclass
class GeneticAlgorithmConfig:
    """
    Configurações principais do Algoritmo Genético.
    """

    population_size: int = 12
    generations: int = 8
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elitism_size: int = 2
    tournament_size: int = 3
    selection_method: str = "tournament"
    cv_splits: int = 3
    seed: int = 42
    early_stopping_rounds: int | None = 4


@dataclass
class GeneticAlgorithmResult:
    """
    Resultado final da otimização genética.
    """

    best_individual: Individual
    best_fitness: float
    history: List[Dict[str, Any]]
    final_population: List[Individual]


class GeneticHyperparameterOptimizer:
    """
    Otimizador de hiperparâmetros baseado em Algoritmo Genético.
    """

    def __init__(
        self,
        model_name: str,
        model_factory: Callable[..., BaseEstimator],
        config: GeneticAlgorithmConfig | None = None,
    ):
        self.model_name = model_name
        self.model_factory = model_factory
        self.config = config or GeneticAlgorithmConfig()
        self.rng = random.Random(self.config.seed)

    def _evaluate_population(self, population: List[Individual], X, y) -> List[EvaluatedIndividual]:
        """
        Avalia todos os indivíduos da população e ordena por fitness decrescente.
        """
        evaluated_population = []

        for individual in population:
            fitness = evaluate_individual(
                model_factory=self.model_factory,
                hyperparameters=individual,
                X=X,
                y=y,
                cv_splits=self.config.cv_splits,
                seed=self.config.seed,
            )
            evaluated_population.append((individual, fitness))

        return sorted(evaluated_population, key=lambda item: item[1], reverse=True)

    def _select_parent(self, evaluated_population: List[EvaluatedIndividual]) -> Individual:
        """
        Seleciona um pai usando o método configurado.
        """
        selection_seed = self.rng.randint(0, 10_000_000)

        if self.config.selection_method == "tournament":
            return tournament_selection(
                evaluated_population=evaluated_population,
                tournament_size=self.config.tournament_size,
                seed=selection_seed,
            )

        if self.config.selection_method == "roulette":
            return roulette_selection(
                evaluated_population=evaluated_population,
                seed=selection_seed,
            )

        raise ValueError("selection_method deve ser 'tournament' ou 'roulette'.")

    def optimize(self, X, y) -> GeneticAlgorithmResult:
        """
        Executa o loop completo do Algoritmo Genético.
        """
        population = create_population(
            model_name=self.model_name,
            population_size=self.config.population_size,
            seed=self.config.seed,
        )

        best_individual = None
        best_fitness = -1.0
        history = []
        rounds_without_improvement = 0

        for generation in range(1, self.config.generations + 1):
            print(f"\nGeração {generation}/{self.config.generations} - {self.model_name}")

            evaluated_population = self._evaluate_population(population, X, y)

            generation_best_individual, generation_best_fitness = evaluated_population[0]
            generation_avg_fitness = sum(fitness for _, fitness in evaluated_population) / len(evaluated_population)

            if generation_best_fitness > best_fitness:
                best_individual = generation_best_individual.copy()
                best_fitness = generation_best_fitness
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            history.append({
                "generation": generation,
                "best_fitness": generation_best_fitness,
                "avg_fitness": generation_avg_fitness,
                "global_best_fitness": best_fitness,
                "best_individual": generation_best_individual.copy(),
            })

            print(f"Melhor fitness da geração: {generation_best_fitness:.4f}")
            print(f"Fitness médio da geração: {generation_avg_fitness:.4f}")
            print(f"Melhor fitness global: {best_fitness:.4f}")

            if (
                self.config.early_stopping_rounds is not None
                and rounds_without_improvement >= self.config.early_stopping_rounds
            ):
                print("Parada antecipada acionada: sem melhora nas últimas gerações.")
                break

            elites = [
                individual.copy()
                for individual, _ in evaluated_population[: self.config.elitism_size]
            ]

            next_population = elites.copy()

            while len(next_population) < self.config.population_size:
                parent_1 = self._select_parent(evaluated_population)
                parent_2 = self._select_parent(evaluated_population)

                child_1, child_2 = uniform_crossover(
                    parent_1=parent_1,
                    parent_2=parent_2,
                    crossover_rate=self.config.crossover_rate,
                    seed=self.rng.randint(0, 10_000_000),
                )

                child_1 = mutate(
                    model_name=self.model_name,
                    individual=child_1,
                    mutation_rate=self.config.mutation_rate,
                    seed=self.rng.randint(0, 10_000_000),
                )

                child_2 = mutate(
                    model_name=self.model_name,
                    individual=child_2,
                    mutation_rate=self.config.mutation_rate,
                    seed=self.rng.randint(0, 10_000_000),
                )

                next_population.extend([child_1, child_2])

            population = next_population[: self.config.population_size]

        return GeneticAlgorithmResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            history=history,
            final_population=population,
        )


def optimize_hyperparameters(
    model_name: str,
    X_train,
    y_train,
    config: GeneticAlgorithmConfig | None = None,
) -> GeneticAlgorithmResult:
    """
    Função auxiliar para rodar o AG usando o padrão de modelos do projeto.
    """

    def model_factory(**params):
        return create_model_by_name(model_name, params)

    optimizer = GeneticHyperparameterOptimizer(
        model_name=model_name,
        model_factory=model_factory,
        config=config,
    )

    return optimizer.optimize(X_train, y_train)


def train_and_evaluate_optimized_model(
    model_name: str,
    params: dict,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """
    Treina e avalia um modelo com os hiperparâmetros encontrados pelo AG.
    """

    print("\n" + "=" * 60)
    print(f"MODELO OTIMIZADO - {model_name}")
    print("=" * 60)
    print(f"Hiperparâmetros: {params}")

    model = create_model_by_name(model_name, params)

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=f"{model_name} Otimizado",
    )

    return {
        "model_name": model_name,
        "params": params,
        "model": model,
        "metrics": metrics,
    }


def run_optimized_models(
    best_params_by_model: dict,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    """
    Executa o treinamento e avaliação de vários modelos já otimizados.
    Mantido para compatibilidade com o restante do projeto.
    """

    results = {}

    for model_name, params in best_params_by_model.items():
        result = train_and_evaluate_optimized_model(
            model_name=model_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results[model_name] = result

    return results


def run_genetic_optimization_for_models(
    model_names: List[str],
    X_train,
    y_train,
    config: GeneticAlgorithmConfig | None = None,
) -> dict:
    """
    Roda o Algoritmo Genético para uma lista de modelos.

    Retorna:
        {
            "Random Forest": {
                "best_params": {...},
                "best_fitness": 0.91,
                "history": [...]
            }
        }
    """

    optimization_results = {}

    for model_name in model_names:
        result = optimize_hyperparameters(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            config=config,
        )

        optimization_results[model_name] = {
            "best_params": result.best_individual,
            "best_fitness": result.best_fitness,
            "history": result.history,
        }

    return optimization_results
