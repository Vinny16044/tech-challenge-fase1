"""
Módulo de Modelos de Machine Learning
Responsável: Rodrigo (Desenvolvedor)

Implementação dos modelos de classificação para diagnóstico médico.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import os


def create_models() -> dict:
    """
    Cria e retorna os modelos de classificação a serem treinados.

    Returns:
        Dicionário com nome e instância de cada modelo
    """
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
    }

    print(f"Modelos criados: {list(models.keys())}")
    return models


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
