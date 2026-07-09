"""
Representação dos genes e criação da população inicial.

Compatível com os modelos usados na Fase 1:
- Regressão Logística
- Random Forest
- KNN
- Árvore de Decisão
- XGBoost

Cada indivíduo é um dicionário em que cada chave representa um hiperparâmetro.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "Regressão Logística": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "class_weight": [None, "balanced"],
        "max_iter": [10000],
        "random_state": [42],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
        "random_state": [42],
        "n_jobs": [-1],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],
        "p": [1, 2],
        "n_jobs": [-1],
    },
    "Árvore de Decisão": {
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy", "log_loss"],
        "class_weight": [None, "balanced"],
        "random_state": [42],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 4, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "eval_metric": ["logloss"],
        "random_state": [42],
    },
}


def get_search_space(model_name: str) -> Dict[str, List[Any]]:
    """
    Retorna o espaço de busca do modelo informado.
    """
    if model_name not in SEARCH_SPACES:
        available = ", ".join(SEARCH_SPACES.keys())
        raise ValueError(f"Modelo '{model_name}' não suportado. Opções: {available}")

    return SEARCH_SPACES[model_name]


def create_individual(model_name: str, rng: random.Random | None = None) -> Dict[str, Any]:
    """
    Cria um indivíduo aleatório para o modelo informado.
    """
    rng = rng or random
    search_space = get_search_space(model_name)

    return {
        hyperparameter: rng.choice(possible_values)
        for hyperparameter, possible_values in search_space.items()
    }


def create_population(
    model_name: str,
    population_size: int,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Cria a população inicial do Algoritmo Genético.
    """
    if population_size <= 0:
        raise ValueError("population_size deve ser maior que zero.")

    rng = random.Random(seed)

    return [
        create_individual(model_name=model_name, rng=rng)
        for _ in range(population_size)
    ]


def validate_individual(model_name: str, individual: Dict[str, Any]) -> bool:
    """
    Valida se todos os genes do indivíduo pertencem ao espaço de busca.
    """
    search_space = get_search_space(model_name)

    if set(individual.keys()) != set(search_space.keys()):
        return False

    return all(
        individual[gene] in possible_values
        for gene, possible_values in search_space.items()
    )