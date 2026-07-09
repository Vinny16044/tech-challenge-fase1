import os
import sys

import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from genetic.fitness import weighted_medical_fitness, evaluate_individual


def test_weighted_medical_fitness_calculates_expected_value():
    fitness = weighted_medical_fitness(
        recall=0.90,
        f1=0.80,
        accuracy=0.70,
    )

    expected = (0.60 * 0.90) + (0.30 * 0.80) + (0.10 * 0.70)

    assert fitness == pytest.approx(expected)


def test_weighted_medical_fitness_rejects_invalid_weights():
    with pytest.raises(ValueError):
        weighted_medical_fitness(
            recall=0.90,
            f1=0.80,
            accuracy=0.70,
            recall_weight=0.60,
            f1_weight=0.30,
            accuracy_weight=0.20,
        )


def test_evaluate_individual_returns_float_between_zero_and_one():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    fitness = evaluate_individual(
        model_factory=DecisionTreeClassifier,
        hyperparameters={"max_depth": 3, "random_state": 42},
        X=X,
        y=y,
        cv_splits=2,
        seed=42,
    )

    assert isinstance(fitness, float)
    assert 0.0 <= fitness <= 1.0
