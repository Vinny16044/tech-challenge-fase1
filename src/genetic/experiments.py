"""
Módulo de experimentos da Fase 2.
Responsável: Paola
"""

import os
import pandas as pd

from genetic.optimizer import run_optimized_models
from monitoring import setup_logger, log_experiment_result
from genetic.visualization import plot_baseline_vs_optimized


EXPERIMENT_CONFIGS = [
    {
        "experiment": "Experimento 1",
        "description": "Configuração inicial conservadora",
        "population_size": 20,
        "generations": 30,
        "mutation_rate": 0.10,
        "crossover_rate": 0.80,
    },
    {
        "experiment": "Experimento 2",
        "description": "Configuração intermediária",
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.20,
        "crossover_rate": 0.90,
    },
    {
        "experiment": "Experimento 3",
        "description": "Configuração exploratória",
        "population_size": 100,
        "generations": 80,
        "mutation_rate": 0.05,
        "crossover_rate": 0.70,
    },
]


def get_mock_best_params() -> dict:
    return {
        "Regressão Logística": {
            "C": 0.5,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
        },
        "Random Forest": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        "KNN": {
            "n_neighbors": 7,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
        },
        "Árvore de Decisão": {
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "criterion": "gini",
        },
        "XGBoost": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    }


def run_phase2_experiments(
    dataset_name: str,
    baseline_results: pd.DataFrame,
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir: str = "results/fase2"
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)
    log_path = setup_logger()

    print("\n" + "=" * 70)
    print(f"EXPERIMENTOS FASE 2 - {dataset_name}")
    print("=" * 70)
    print(f"Log criado em: {log_path}")

    all_rows = []

    for config in EXPERIMENT_CONFIGS:
        print("\n" + "-" * 70)
        print(config["experiment"])
        print(config["description"])
        print("-" * 70)

        best_params_by_model = get_mock_best_params()

        optimized_results = run_optimized_models(
            best_params_by_model=best_params_by_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        for model_name, result in optimized_results.items():
            metrics = result["metrics"]
            params = result["params"]

            log_experiment_result(model_name, params, metrics)

            baseline_recall = baseline_results.loc[model_name, "recall"] if model_name in baseline_results.index else None
            baseline_f1 = baseline_results.loc[model_name, "f1_score"] if model_name in baseline_results.index else None
            baseline_accuracy = baseline_results.loc[model_name, "accuracy"] if model_name in baseline_results.index else None

            all_rows.append({
                "dataset": dataset_name,
                "experiment": config["experiment"],
                "population_size": config["population_size"],
                "generations": config["generations"],
                "mutation_rate": config["mutation_rate"],
                "crossover_rate": config["crossover_rate"],
                "model": model_name,
                "baseline_accuracy": baseline_accuracy,
                "optimized_accuracy": metrics.get("accuracy"),
                "baseline_recall": baseline_recall,
                "optimized_recall": metrics.get("recall"),
                "baseline_f1": baseline_f1,
                "optimized_f1": metrics.get("f1_score"),
                "recall_gain": metrics.get("recall") - baseline_recall if baseline_recall is not None else None,
                "f1_gain": metrics.get("f1_score") - baseline_f1 if baseline_f1 is not None else None,
                "best_params": str(params),
            })

    df_comparison = pd.DataFrame(all_rows)

    output_path = os.path.join(
        output_dir,
        f"{dataset_name.lower().replace(' ', '_')}_phase2_comparison.csv"
    )

    df_comparison.to_csv(output_path, index=False, encoding="utf-8-sig")

    try:
        plot_baseline_vs_optimized(df_comparison, metric="recall")
        plot_baseline_vs_optimized(df_comparison, metric="accuracy")
        plot_baseline_vs_optimized(df_comparison, metric="f1")
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")

    print("\n" + "=" * 70)
    print(f"Comparativo salvo em: {output_path}")
    print("=" * 70)

    return df_comparison