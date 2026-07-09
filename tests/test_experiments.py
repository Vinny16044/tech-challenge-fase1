import os
import sys

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import genetic.experiments as experiments
from genetic.experiments import run_phase2_experiments


def test_run_phase2_experiments(tmp_path, monkeypatch):
    """
    Teste leve do pipeline de experimentos.

    Usa apenas Árvore de Decisão e uma configuração mínima do AG para validar
    o fluxo sem transformar o teste automatizado em execução completa de benchmark.
    """
    monkeypatch.setattr(
        experiments,
        "EXPERIMENT_CONFIGS",
        [
            {
                "experiment": "Teste rápido",
                "description": "Configuração mínima para teste automatizado",
                "population_size": 4,
                "generations": 2,
                "mutation_rate": 0.10,
                "crossover_rate": 0.80,
                "elitism_size": 1,
                "cv_splits": 2,
                "early_stopping_rounds": 1,
            }
        ],
    )

    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline_results = pd.DataFrame(
        {
            "accuracy": [0.80],
            "recall": [0.80],
            "f1_score": [0.80],
        },
        index=["Árvore de Decisão"],
    )

    df_result = run_phase2_experiments(
        dataset_name="Teste",
        baseline_results=baseline_results,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=str(tmp_path),
        model_names=["Árvore de Decisão"],
    )

    comparison_file = tmp_path / "teste_phase2_comparison.csv"
    history_file = tmp_path / "teste_fitness_history.csv"

    assert df_result is not None
    assert not df_result.empty
    assert comparison_file.exists()
    assert history_file.exists()
    assert "dataset" in df_result.columns
    assert "experiment" in df_result.columns
    assert "model" in df_result.columns
    assert "optimized_recall" in df_result.columns
    assert "best_fitness" in df_result.columns
