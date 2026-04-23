"""
Módulo de Avaliação e Explicabilidade
Responsável: Rodrigo (Desenvolvedor)

Avaliação dos modelos com métricas adequadas para diagnóstico médico
e explicabilidade com SHAP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import os


def evaluate_model(model, X_test, y_test, model_name: str = "modelo") -> dict:
    """
    Avalia um modelo com múltiplas métricas.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        model_name: Nome do modelo

    Returns:
        Dicionário com as métricas
    """
    y_pred = model.predict(X_test)

    # Probabilidades para AUC-ROC (se disponível)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except AttributeError:
        y_prob = None
        auc = None

    metrics = {
        'modelo': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'auc_roc': auc,
    }

    print(f"\n--- {model_name} ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if auc:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Avalia todos os modelos e retorna um DataFrame comparativo.

    Args:
        models: Dicionário de modelos treinados
        X_test: Features de teste
        y_test: Target de teste

    Returns:
        DataFrame com métricas comparativas
    """
    print("\n" + "=" * 60)
    print("AVALIAÇÃO DOS MODELOS")
    print("=" * 60)

    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    df_results = pd.DataFrame(results).set_index('modelo')
    df_results = df_results.sort_values('recall', ascending=False)

    print("\n" + "=" * 60)
    print("RANKING DOS MODELOS (ordenado por Recall)")
    print("=" * 60)
    print(df_results.to_string())

    return df_results


def plot_confusion_matrix(model, X_test, y_test, model_name: str,
                          labels: list = None, save_path: str = None):
    """
    Gera e plota a matriz de confusão.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        model_name: Nome do modelo
        labels: Nomes das classes
        save_path: Caminho para salvar a figura
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if labels is None:
        labels = ['Maligno', 'Benigno']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Matriz de Confusão - {model_name}', fontsize=14)
    ax.set_xlabel('Predição', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura salva em: {save_path}")

    plt.show()


def plot_roc_curves(models: dict, X_test, y_test, save_path: str = None):
    """
    Plota as curvas ROC de todos os modelos.

    Args:
        models: Dicionário de modelos treinados
        X_test: Features de teste
        y_test: Target de teste
        save_path: Caminho para salvar a figura
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in models.items():
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
        except AttributeError:
            print(f"  {name}: predict_proba não disponível, pulando curva ROC")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aleatório')
    ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    ax.set_title('Curvas ROC - Comparação de Modelos', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura salva em: {save_path}")

    plt.show()


def explain_with_shap(model, X_test, feature_names: list,
                       model_name: str = "modelo", save_dir: str = None):
    """
    Gera explicações SHAP para o modelo.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        feature_names: Nomes das features
        model_name: Nome do modelo
        save_dir: Diretório para salvar figuras
    """
    try:
        import shap

        print(f"\nGerando explicações SHAP para {model_name}...")

        # Escolhe o explainer adequado
        if hasattr(model, 'estimators_'):
            # Tree-based models (Random Forest, XGBoost, etc.)
            explainer = shap.TreeExplainer(model)
        else:
            # Modelos lineares ou genéricos
            explainer = shap.LinearExplainer(model, X_test)

        shap_values = explainer.shap_values(X_test)

        # Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'shap_summary_{model_name.replace(" ", "_")}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SHAP Summary salvo em: {save_path}")

        plt.show()

        # Feature Importance Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14)
        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'shap_importance_{model_name.replace(" ", "_")}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SHAP Importance salvo em: {save_path}")

        plt.show()

        return shap_values

    except ImportError:
        print("SHAP não instalado. Execute: pip install shap")
        return None
    except Exception as e:
        print(f"Erro ao gerar SHAP para {model_name}: {e}")
        return None


def get_feature_importance(model, feature_names: list,
                           model_name: str = "modelo", save_path: str = None):
    """
    Extrai e plota a importância das features (para modelos baseados em árvore).

    Args:
        model: Modelo treinado
        feature_names: Nomes das features
        model_name: Nome do modelo
        save_path: Caminho para salvar a figura
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} não suporta feature_importances_")
        return None

    importance = model.feature_importances_
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df_imp['feature'], df_imp['importance'], color='steelblue')
    ax.set_xlabel('Importância', fontsize=12)
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura salva em: {save_path}")

    plt.show()

    return df_imp
