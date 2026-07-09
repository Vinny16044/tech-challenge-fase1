"""
Módulo de Modelos de Machine Learning
Responsável: Rodrigo (Desenvolvedor)

Implementação dos modelos de classificação para diagnóstico médico.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import os


def create_models() -> dict:
    models = {
        'Regressão Logística': LogisticRegression(
            max_iter=10000,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        'Árvore de Decisão': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
            max_depth=10
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )
    }

    print(f"Modelos criados: {list(models.keys())}")
    return models


def create_model_by_name(model_name: str, params: dict = None):
    if params is None:
        params = {}

    if model_name == "Regressão Logística":
        return LogisticRegression(
            max_iter=params.get("max_iter", 10000),
            C=params.get("C", 1.0),
            penalty=params.get("penalty", "l2") or "l2",
            solver=params.get("solver", "lbfgs"),
            class_weight=params.get("class_weight", "balanced"),
            random_state=42
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            metric=params.get("metric", "minkowski"),
            p=params.get("p", 2),
            n_jobs=-1
        )

    if model_name == "Árvore de Decisão":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            criterion=params.get("criterion", "gini"),
            class_weight="balanced",
            random_state=42
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            eval_metric="logloss",
            random_state=42
        )

    raise ValueError(f"Modelo '{model_name}' não reconhecido.")


def train_model(model, X_train, y_train, model_name: str = "modelo"):
    """
    Treina um modelo com os dados de treinamento.

    Args:
        model: Instância do modelo sklearn
        X_train: Features de treinamento
        y_train: Target de treinamento
        model_name: Nome do modelo para log

    Returns:
        Modelo treinado
    """
    print(f"Treinando {model_name}...", end=" ")
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    print(f"Acurácia no treino: {train_score:.4f}")

    return model


def train_all_models(models: dict, X_train, y_train) -> dict:
    """
    Treina todos os modelos.

    Args:
        models: Dicionário de modelos
        X_train: Features de treinamento
        y_train: Target de treinamento

    Returns:
        Dicionário com modelos treinados
    """
    print("\n" + "=" * 50)
    print("TREINAMENTO DOS MODELOS")
    print("=" * 50)

    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_model(model, X_train, y_train, name)

    print("=" * 50)
    print(f"Todos os {len(trained_models)} modelos treinados com sucesso!")
    return trained_models


def save_model(model, filepath: str):
    """
    Salva um modelo treinado em disco.

    Args:
        model: Modelo treinado
        filepath: Caminho para salvar
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Modelo salvo em: {filepath}")


def load_model(filepath: str):
    """
    Carrega um modelo salvo do disco.

    Args:
        filepath: Caminho do modelo salvo

    Returns:
        Modelo carregado
    """
    model = joblib.load(filepath)
    print(f"Modelo carregado de: {filepath}")
    return model
