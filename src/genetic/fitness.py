"""

Função fitness baseada em validação cruzada.

A prioridade é Recall, pois em diagnóstico médico falsos negativos
são mais críticos que falsos positivos.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate


def weighted_medical_fitness(
    recall: float,
    f1: float,
    accuracy: float,
    recall_weight: float = 0.60,
    f1_weight: float = 0.30,
    accuracy_weight: float = 0.10,
) -> float:
    """
    Calcula a nota final do indivíduo.
    """
    weights_sum = recall_weight + f1_weight + accuracy_weight

    if not np.isclose(weights_sum, 1.0):
        raise ValueError("A soma dos pesos deve ser igual a 1.0.")

    return (
        recall_weight * recall
        + f1_weight * f1
        + accuracy_weight * accuracy
    )


def evaluate_individual(
    model_factory: Callable[..., BaseEstimator],
    hyperparameters: Dict[str, Any],
    X,
    y,
    cv_splits: int = 5,
    seed: int = 42,
) -> float:
    """
    Avalia um indivíduo com validação cruzada estratificada.

    Usa métricas weighted para manter compatibilidade com a avaliação
    já usada na Fase 1.
    """
    model = model_factory(**hyperparameters)

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed,
    )

    scoring = {
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "accuracy": "accuracy",
    }

    try:
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        recall = float(np.mean(scores["test_recall"]))
        f1 = float(np.mean(scores["test_f1"]))
        accuracy = float(np.mean(scores["test_accuracy"]))

        return weighted_medical_fitness(
            recall=recall,
            f1=f1,
            accuracy=accuracy,
        )

    except Exception as exc:
        print(f"Indivíduo inválido ou falha no treino: {hyperparameters}. Erro: {exc}")
        return 0.0