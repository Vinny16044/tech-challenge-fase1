"""
Módulo de experimentos da Fase 2.
Responsável: Paola
"""
from __future__ import annotations
import os
import pandas as pd

from typing import Iterable

from genetic.optimizer import (
    GeneticAlgorithmConfig,
    optimize_hyperparameters,
    run_optimized_models,
)
from monitoring import setup_logger, log_experiment_result
from genetic.visualization import plot_baseline_vs_optimized


EXPERIMENT_CONFIGS = [
    {
        "experiment": "Experimento 1",
        "description": "Configuração inicial conservadora",
        "population_size": 8,
        "generations": 5,
        "mutation_rate": 0.10,
        "crossover_rate": 0.80,
        "elitism_size": 2,
        "cv_splits": 3,
        "early_stopping_rounds": 3,
    },
    {
        "experiment": "Experimento 2",
        "description": "Configuração intermediária",
        "population_size": 12,
        "generations": 8,
        "mutation_rate": 0.20,
        "crossover_rate": 0.85,
        "elitism_size": 2,
        "cv_splits": 3,
        "early_stopping_rounds": 4,
    },
    {
        "experiment": "Experimento 3",
        "description": "Configuração exploratória",
        "population_size": 16,
        "generations": 10,
        "mutation_rate": 0.15,
        "crossover_rate": 0.90,
        "elitism_size": 2,
        "cv_splits": 3,
        "early_stopping_rounds": 4,
    },
]


DEFAULT_MODEL_NAMES = [
    "Regressão Logística",
    "Random Forest",
    "KNN",
    "Árvore de Decisão",
    "XGBoost",
]


def _resolve_model_names(
    baseline_results: pd.DataFrame,
    model_names: Iterable[str] | None = None,
) -> list[str]:
    """
    Define quais modelos serão otimizados.

    Prioridade:
    1. lista passada no argumento model_names;
    2. modelos presentes no índice do baseline_results;
    3. lista padrão com os 5 modelos do projeto.
    """

    if model_names is not None:
        return list(model_names)

    if baseline_results is not None and not baseline_results.empty:
        return [
            model_name
            for model_name in baseline_results.index.tolist()
            if model_name in DEFAULT_MODEL_NAMES
        ]

    return DEFAULT_MODEL_NAMES.copy()


def _build_ga_config(config: dict, seed: int = 42) -> GeneticAlgorithmConfig:
    """
    Converte o dicionário do experimento em configuração do AG.
    """

    return GeneticAlgorithmConfig(
        population_size=config["population_size"],
        generations=config["generations"],
        crossover_rate=config["crossover_rate"],
        mutation_rate=config["mutation_rate"],
        elitism_size=config.get("elitism_size", 2),
        tournament_size=config.get("tournament_size", 3),
        selection_method=config.get("selection_method", "tournament"),
        cv_splits=config.get("cv_splits", 3),
        seed=seed,
        early_stopping_rounds=config.get("early_stopping_rounds", 4),
    )


def _get_baseline_metric(
    baseline_results: pd.DataFrame,
    model_name: str,
    metric: str,
):
    """
    Busca uma métrica no resultado baseline sem quebrar caso ela não exista.
    """

    if (
        baseline_results is not None
        and model_name in baseline_results.index
        and metric in baseline_results.columns
    ):
        return baseline_results.loc[model_name, metric]

    return None


def run_phase2_experiments(
    dataset_name: str,
    baseline_results: pd.DataFrame,
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir: str = "results/fase2",
    model_names: Iterable[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Executa os experimentos da Fase 2.

    Fluxo correto:
    1. monta uma configuração de AG;
    2. roda o AG para cada modelo;
    3. usa os melhores hiperparâmetros encontrados;
    4. treina os modelos otimizados;
    5. compara baseline vs. otimizado.
    """

    os.makedirs(output_dir, exist_ok=True)
    log_path = setup_logger()

    selected_models = _resolve_model_names(
        baseline_results=baseline_results,
        model_names=model_names,
    )

    print("\n" + "=" * 70)
    print(f"EXPERIMENTOS FASE 2 - {dataset_name}")
    print("=" * 70)
    print(f"Log criado em: {log_path}")
    print(f"Modelos selecionados: {selected_models}")

    all_rows = []
    all_fitness_history_rows = []

    for experiment_index, config in enumerate(EXPERIMENT_CONFIGS, start=1):
        print("\n" + "-" * 70)
        print(config["experiment"])
        print(config["description"])
        print("-" * 70)

        ga_config = _build_ga_config(config=config, seed=seed + experiment_index)

        best_params_by_model = {}
        optimization_metadata_by_model = {}

        for model_name in selected_models:
            print("\n" + "#" * 70)
            print(f"Otimizando hiperparâmetros com AG: {model_name}")
            print("#" * 70)

            optimization_result = optimize_hyperparameters(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                config=ga_config,
            )

            best_params_by_model[model_name] = optimization_result.best_individual
            optimization_metadata_by_model[model_name] = {
                "best_fitness": optimization_result.best_fitness,
                "history": optimization_result.history,
            }

            for history_row in optimization_result.history:
                all_fitness_history_rows.append({
                    "dataset": dataset_name,
                    "experiment": config["experiment"],
                    "model": model_name,
                    "generation": history_row["generation"],
                    "best_fitness": history_row["best_fitness"],
                    "avg_fitness": history_row["avg_fitness"],
                    "global_best_fitness": history_row["global_best_fitness"],
                    "best_individual": str(history_row["best_individual"]),
                })

        optimized_results = run_optimized_models(
            best_params_by_model=best_params_by_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        for model_name, result in optimized_results.items():
            metrics = result["metrics"]
            params = result["params"]

            log_experiment_result(model_name, params, metrics)

            baseline_recall = _get_baseline_metric(baseline_results, model_name, "recall")
            baseline_f1 = _get_baseline_metric(baseline_results, model_name, "f1_score")
            baseline_accuracy = _get_baseline_metric(baseline_results, model_name, "accuracy")

            optimized_recall = metrics.get("recall")
            optimized_f1 = metrics.get("f1_score")
            optimized_accuracy = metrics.get("accuracy")

            all_rows.append({
                "dataset": dataset_name,
                "experiment": config["experiment"],
                "population_size": config["population_size"],
                "generations": config["generations"],
                "mutation_rate": config["mutation_rate"],
                "crossover_rate": config["crossover_rate"],
                "model": model_name,
                "baseline_accuracy": baseline_accuracy,
                "optimized_accuracy": optimized_accuracy,
                "baseline_recall": baseline_recall,
                "optimized_recall": optimized_recall,
                "baseline_f1": baseline_f1,
                "optimized_f1": optimized_f1,
                "recall_gain": optimized_recall - baseline_recall if baseline_recall is not None else None,
                "f1_gain": optimized_f1 - baseline_f1 if baseline_f1 is not None else None,
                "best_fitness": optimization_metadata_by_model[model_name]["best_fitness"],
                "best_params": str(params),
            })

    df_comparison = pd.DataFrame(all_rows)

    output_path = os.path.join(
        output_dir,
        f"{dataset_name.lower().replace(' ', '_')}_phase2_comparison.csv",
    )

    df_comparison.to_csv(output_path, index=False, encoding="utf-8-sig")

    history_df = pd.DataFrame(all_fitness_history_rows)
    history_output_path = os.path.join(
        output_dir,
        f"{dataset_name.lower().replace(' ', '_')}_fitness_history.csv",
    )
    history_df.to_csv(history_output_path, index=False, encoding="utf-8-sig")

    try:
        plot_baseline_vs_optimized(df_comparison, metric="recall")
        plot_baseline_vs_optimized(df_comparison, metric="accuracy")
        plot_baseline_vs_optimized(df_comparison, metric="f1")
    except Exception as exc:
        print(f"Erro ao gerar gráficos: {exc}")

    print("\n" + "=" * 70)
    print(f"Comparativo salvo em: {output_path}")
    print(f"Histórico de fitness salvo em: {history_output_path}")
    print("=" * 70)

    return df_comparison
