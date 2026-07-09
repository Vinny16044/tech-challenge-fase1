"""
Módulo de integração entre Algoritmo Genético e modelos de Machine Learning.
Responsável: Paola

Este módulo recebe hiperparâmetros gerados pelo Algoritmo Genético,
cria o modelo correspondente, treina, avalia e retorna as métricas.
"""

from models import create_model_by_name
from evaluation import evaluate_model


def train_and_evaluate_optimized_model(
    model_name: str,
    params: dict,
    X_train,
    y_train,
    X_test,
    y_test
) -> dict:
    """
    Treina e avalia um modelo otimizado com hiperparâmetros definidos pelo AG.
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
        model_name=f"{model_name} Otimizado"
    )

    return {
        "model_name": model_name,
        "params": params,
        "model": model,
        "metrics": metrics
    }


def run_optimized_models(
    best_params_by_model: dict,
    X_train,
    y_train,
    X_test,
    y_test
) -> dict:
    """
    Executa o treinamento e avaliação de vários modelos otimizados.

    Args:
        best_params_by_model: dicionário no formato:
            {
                "Random Forest": {"n_estimators": 200, "max_depth": 10},
                "KNN": {"n_neighbors": 7}
            }

    Returns:
        Dicionário com modelos otimizados e métricas.
    """

    results = {}

    for model_name, params in best_params_by_model.items():
        result = train_and_evaluate_optimized_model(
            model_name=model_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        results[model_name] = result

    return results