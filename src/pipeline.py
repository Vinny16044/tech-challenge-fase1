"""
Módulo de Pipeline End-to-End
Responsável: Vinicius (P.O. e Desenvolvedor)

Orquestra todo o fluxo: carregamento → pré-processamento → modelagem → avaliação.
Suporta os dois datasets: SEER Clínico e Wisconsin Complementar Tumoral.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATASET_SEER_CLINICO, DATASET_WISCONSIN_TUMORAL, DATASET_PCOS,
    SEER_PARAMS, WISCONSIN_PARAMS, MODELING, FIGURES_DIR, MODELS_DIR,
    ensure_dirs
)
from preprocessing import (
    check_data_quality, handle_missing_values, encode_categorical,
    scale_features, split_data
)
from models import create_models, train_all_models, save_model
from evaluation import (
    evaluate_all_models, plot_confusion_matrix, plot_roc_curves,
    explain_with_shap, get_feature_importance
)


# ============================================================
# CARREGAMENTO DOS DATASETS
# ============================================================

def load_seer_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Carrega o Dataset Principal Clínico (SEER Breast Cancer).
    
    Args:
        filepath: Caminho para o CSV (default: config.DATASET_SEER_CLINICO)
    
    Returns:
        DataFrame limpo com colunas padronizadas
    """
    filepath = filepath or DATASET_SEER_CLINICO
    
    df = pd.read_csv(
        filepath,
        sep=SEER_PARAMS['separator'],
        encoding=SEER_PARAMS['encoding']
    )
    
    # Remover coluna vazia
    for col in SEER_PARAMS['drop_cols']:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Remover espaços extras nos nomes
    df.columns = [c.strip() for c in df.columns]
    
    # Criar target binário
    if SEER_PARAMS['target_col'] in df.columns:
        df['target'] = df[SEER_PARAMS['target_col']].map(SEER_PARAMS['target_map'])
    
    print(f"Dataset SEER Clínico carregado: {df.shape}")
    print(f"  Classes (Status): {df['Status'].value_counts().to_dict()}")
    print(f"  Target binário: {df['target'].value_counts().to_dict()}")
    
    return df


def load_wisconsin_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Carrega o Dataset Complementar Tumoral (Wisconsin variant).
    Corrige o problema de valores inteiros (ex: 1354 → 13.54).
    
    Args:
        filepath: Caminho para o CSV (default: config.DATASET_WISCONSIN_TUMORAL)
    
    Returns:
        DataFrame com valores numéricos corrigidos
    """
    filepath = filepath or DATASET_WISCONSIN_TUMORAL
    
    df = pd.read_csv(
        filepath,
        sep=WISCONSIN_PARAMS['separator'],
        encoding=WISCONSIN_PARAMS['encoding']
    )
    
    df.columns = [c.strip() for c in df.columns]
    
    # Criar target binário
    if WISCONSIN_PARAMS['target_col'] in df.columns:
        df['target'] = df[WISCONSIN_PARAMS['target_col']].map(WISCONSIN_PARAMS['target_map'])
    
    # Corrigir valores inteiros → floats (dividir por 100)
    # Todas as features numéricas estão multiplicadas por 100 (ex: 1354 = 13.54)
    feature_cols = [c for c in df.columns
                    if c not in ['id', 'diagnosis', 'target', 'Diagnóstico',
                                 'Radius Ajustado', 'Area Ajustada', 'Concavidade Ajustada']
                    and df[c].dtype in ['int64', 'float64']]

    scale = WISCONSIN_PARAMS['scale_factor']
    for col in feature_cols:
        df[col] = df[col] / scale
    
    # Remover colunas extras em português
    drop_cols = [c for c in WISCONSIN_PARAMS['drop_cols'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    print(f"Dataset Wisconsin Tumoral carregado: {df.shape}")
    print(f"  Classes (diagnosis): {df['diagnosis'].value_counts().to_dict() if 'diagnosis' in df.columns else 'N/A'}")
    print(f"  Target binário: {df['target'].value_counts().to_dict()}")
    
    return df


def load_pcos_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Carrega o dataset PCOS de infertilidade.
    
    Args:
        filepath: Caminho para o CSV
    
    Returns:
        DataFrame do PCOS
    """
    filepath = filepath or DATASET_PCOS
    df = pd.read_csv(filepath)
    print(f"Dataset PCOS carregado: {df.shape}")
    return df


# ============================================================
# PIPELINE POR DATASET
# ============================================================

def run_seer_pipeline(save_figures: bool = True, save_models: bool = True) -> dict:
    """
    Pipeline completo para o Dataset SEER Clínico.
    
    Returns:
        Dicionário com resultados, modelos treinados e métricas
    """
    print("\n" + "=" * 70)
    print("  PIPELINE - DATASET SEER CLÍNICO (Principal)")
    print("=" * 70)
    
    # 1. Carregar
    df = load_seer_dataset()
    
    # 2. Qualidade dos dados
    check_data_quality(df)
    
    # 3. Preparar features
    # Remover colunas que são tradução ou target direto
    cols_to_drop = ['Status', 'Status do Paciente', 'target']
    portuguese_cols = [c for c in SEER_PARAMS.get('portuguese_cols', []) if c in df.columns]
    cols_to_drop.extend(portuguese_cols)
    cols_to_drop = list(set(c for c in cols_to_drop if c in df.columns))
    
    y = df['target'].copy()
    X = df.drop(columns=cols_to_drop)
    
    # 4. Codificar categóricas
    X = encode_categorical(X)
    
    # 5. Tratar nulos
    X = handle_missing_values(X, strategy='median')
    
    # 6. Escalar
    X_scaled, scaler = scale_features(X, method='standard')
    
    # 7. Split
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y,
        test_size=MODELING['test_size'],
        random_state=MODELING['random_state']
    )
    
    # 8. Treinar modelos
    models = create_models()
    trained_models = train_all_models(models, X_train, y_train)
    
    # 9. Avaliar
    df_results = evaluate_all_models(trained_models, X_test, y_test)
    
    # 10. Visualizações
    fig_dir = os.path.join(FIGURES_DIR, 'seer') if save_figures else None
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
    
    best_model_name = df_results.index[0]
    best_model = trained_models[best_model_name]
    
    if save_figures:
        plot_confusion_matrix(best_model, X_test, y_test, best_model_name,
                              labels=['Óbito', 'Vivo'],
                              save_path=os.path.join(fig_dir, f'cm_{best_model_name.replace(" ", "_")}.png'))
        plot_roc_curves(trained_models, X_test, y_test,
                        save_path=os.path.join(fig_dir, 'roc_curves_all.png'))
    
    # 11. SHAP para o melhor modelo
    if save_figures:
        explain_with_shap(best_model, X_test, list(X.columns),
                          best_model_name, save_dir=fig_dir)
    
    # 12. Salvar modelos
    if save_models:
        os.makedirs(MODELS_DIR, exist_ok=True)
        for name, model in trained_models.items():
            save_model(model, os.path.join(MODELS_DIR, f'seer_{name.replace(" ", "_")}.pkl'))
    
    # 13. Salvar resultados
    results_path = os.path.join(os.path.dirname(FIGURES_DIR), 'seer_comparacao_modelos.csv')
    df_results.to_csv(results_path)
    print(f"\nResultados salvos em: {results_path}")
    
    return {
        'dataset': 'SEER Clínico',
        'df_original': df,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'trained_models': trained_models,
        'results': df_results,
        'best_model': (best_model_name, best_model),
    }


def run_wisconsin_pipeline(save_figures: bool = True, save_models: bool = True) -> dict:
    """
    Pipeline completo para o Dataset Wisconsin Complementar Tumoral.
    
    Returns:
        Dicionário com resultados, modelos treinados e métricas
    """
    print("\n" + "=" * 70)
    print("  PIPELINE - DATASET WISCONSIN COMPLEMENTAR TUMORAL")
    print("=" * 70)
    
    # 1. Carregar
    df = load_wisconsin_dataset()
    
    # 2. Qualidade dos dados
    check_data_quality(df)
    
    # 3. Preparar features
    cols_to_drop = ['diagnosis', 'target']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    y = df['target'].copy()
    X = df.drop(columns=cols_to_drop)
    
    # 4. Tratar nulos
    X = handle_missing_values(X, strategy='median')
    
    # 5. Escalar
    X_scaled, scaler = scale_features(X, method='standard')
    
    # 6. Split
    X_train, X_test, y_train, y_test = split_data(
        X_scaled, y,
        test_size=MODELING['test_size'],
        random_state=MODELING['random_state']
    )
    
    # 7. Treinar modelos
    models = create_models()
    trained_models = train_all_models(models, X_train, y_train)
    
    # 8. Avaliar
    df_results = evaluate_all_models(trained_models, X_test, y_test)
    
    # 9. Visualizações
    fig_dir = os.path.join(FIGURES_DIR, 'wisconsin') if save_figures else None
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
    
    best_model_name = df_results.index[0]
    best_model = trained_models[best_model_name]
    
    if save_figures:
        plot_confusion_matrix(best_model, X_test, y_test, best_model_name,
                              labels=['Benigno', 'Maligno'],
                              save_path=os.path.join(fig_dir, f'cm_{best_model_name.replace(" ", "_")}.png'))
        plot_roc_curves(trained_models, X_test, y_test,
                        save_path=os.path.join(fig_dir, 'roc_curves_all.png'))
    
    # 10. SHAP
    if save_figures:
        explain_with_shap(best_model, X_test, list(X.columns),
                          best_model_name, save_dir=fig_dir)
    
    # 11. Salvar modelos
    if save_models:
        os.makedirs(MODELS_DIR, exist_ok=True)
        for name, model in trained_models.items():
            save_model(model, os.path.join(MODELS_DIR, f'wisconsin_{name.replace(" ", "_")}.pkl'))
    
    # 12. Salvar resultados
    results_path = os.path.join(os.path.dirname(FIGURES_DIR), 'wisconsin_comparacao_modelos.csv')
    df_results.to_csv(results_path)
    print(f"\nResultados salvos em: {results_path}")
    
    return {
        'dataset': 'Wisconsin Complementar Tumoral',
        'df_original': df,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'trained_models': trained_models,
        'results': df_results,
        'best_model': (best_model_name, best_model),
    }


# ============================================================
# PIPELINE COMPLETO (AMBOS DATASETS)
# ============================================================

def run_full_pipeline(save_figures: bool = True, save_models: bool = True) -> dict:
    """
    Executa o pipeline completo para ambos os datasets.
    
    Returns:
        Dicionário com resultados de ambos os pipelines
    """
    ensure_dirs()
    
    print("\n" + "#" * 70)
    print("#  TECH CHALLENGE FASE 1 - PIPELINE COMPLETO")
    print("#  Saúde e Segurança da Mulher")
    print("#  FIAP POS TECH - IADT")
    print("#" * 70)
    
    results = {}
    
    # Pipeline SEER
    try:
        results['seer'] = run_seer_pipeline(save_figures, save_models)
        print(f"\n✓ Pipeline SEER concluído com sucesso!")
    except Exception as e:
        print(f"\n✗ Erro no pipeline SEER: {e}")
        import traceback
        traceback.print_exc()
    
    # Pipeline Wisconsin
    try:
        results['wisconsin'] = run_wisconsin_pipeline(save_figures, save_models)
        print(f"\n✓ Pipeline Wisconsin concluído com sucesso!")
    except Exception as e:
        print(f"\n✗ Erro no pipeline Wisconsin: {e}")
        import traceback
        traceback.print_exc()
    
    # Resumo comparativo
    if len(results) == 2:
        print("\n" + "#" * 70)
        print("#  RESUMO COMPARATIVO")
        print("#" * 70)
        for key, res in results.items():
            best_name, _ = res['best_model']
            best_recall = res['results'].loc[best_name, 'recall']
            best_f1 = res['results'].loc[best_name, 'f1_score']
            print(f"\n  {res['dataset']}:")
            print(f"    Melhor modelo: {best_name}")
            print(f"    Recall: {best_recall:.4f} | F1: {best_f1:.4f}")
    
    return results


if __name__ == '__main__':
    run_full_pipeline()
