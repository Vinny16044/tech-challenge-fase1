"""
Módulo de Configuração Central
Responsável: Vinicius (P.O. e Desenvolvedor)

Configurações centralizadas para paths, parâmetros e constantes do projeto.
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PHASE2_RESULTS_DIR = os.path.join(RESULTS_DIR, 'fase2')
PHASE2_LOGS_DIR = os.path.join(PHASE2_RESULTS_DIR, 'logs')

# Datasets
DATASET_SEER_CLINICO = os.path.join(RAW_DIR, 'DatasetPrincipalClinico.csv')
DATASET_WISCONSIN_TUMORAL = os.path.join(RAW_DIR, 'DatasetComplementarTumoral.csv')
DATASET_PCOS = os.path.join(RAW_DIR, 'PCOS_infertility.csv')

# ============================================================
# PARÂMETROS DOS DATASETS
# ============================================================
SEER_PARAMS = {
    'separator': ';',
    'encoding': 'latin-1',
    'target_col': 'Status',
    'target_map': {'Alive': 1, 'Dead': 0},  # 1=Vivo, 0=Óbito
    'target_col_pt': 'Status do Paciente',
    'drop_cols': ['Column1'],  # Coluna vazia
    'categorical_cols': [
        'Race', 'Marital Status', 'T Stage', 'N Stage',
        '6th Stage', 'Grade', 'A Stage', 'Estrogen Status',
        'Progesterone Status'
    ],
    'numeric_cols': ['Age', 'Tumor Size', 'Regional Node Examined',
                     'Reginal Node Positive', 'Survival Months'],
    'portuguese_cols': ['Status do Paciente', 'Raça', 'Estrogênio',
                        'Progesterona', 'Faixa Etária', 'Estado Civil'],
}

WISCONSIN_PARAMS = {
    'separator': ';',
    'encoding': 'latin-1',
    'target_col': 'diagnosis',
    'target_map': {'M': 1, 'B': 0},  # 1=Maligno, 0=Benigno
    'drop_cols': ['id', 'Diagnóstico', 'Radius Ajustado',
                  'Area Ajustada', 'Concavidade Ajustada'],
    'scale_factor': 100.0,  # Valores inteiros precisam ser divididos por 100
    'comma_decimal_cols': ['Radius Ajustado', 'Area Ajustada', 'Concavidade Ajustada'],
}

# ============================================================
# PARÂMETROS DE MODELAGEM
# ============================================================
MODELING = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'scoring_metric': 'recall',  # Prioridade: minimizar falsos negativos
}

# ============================================================
# MODELOS
# ============================================================
MODEL_CONFIGS = {
    'Regressão Logística': {
        'class': 'sklearn.linear_model.LogisticRegression',
        'params': {'max_iter': 10000, 'random_state': 42, 'class_weight': 'balanced'}
    },
    'Random Forest': {
        'class': 'sklearn.ensemble.RandomForestClassifier',
        'params': {'n_estimators': 100, 'random_state': 42, 'class_weight': 'balanced', 'n_jobs': -1}
    },
    'KNN': {
        'class': 'sklearn.neighbors.KNeighborsClassifier',
        'params': {'n_neighbors': 5, 'n_jobs': -1}
    },
    'Árvore de Decisão': {
        'class': 'sklearn.tree.DecisionTreeClassifier',
        'params': {'random_state': 42, 'class_weight': 'balanced', 'max_depth': 10}
    },
    'XGBoost': {
        'class': 'xgboost.XGBClassifier',
        'params': {
            'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'eval_metric': 'logloss', 'random_state': 42
        }
    },
}

# ============================================================
# VISUALIZAÇÃO
# ============================================================
VIZ = {
    'figsize_default': (12, 8),
    'figsize_small': (8, 6),
    'dpi': 150,
    'style': 'seaborn-v0_8-whitegrid',
    'palette': 'Set2',
}

def ensure_dirs():
    """Cria os diretórios necessários se não existirem."""
    for d in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        PHASE2_RESULTS_DIR,
        PHASE2_LOGS_DIR
    ]:
        os.makedirs(d, exist_ok=True)

if __name__ == '__main__':
    ensure_dirs()
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Datasets encontrados:")
    for name, path in [('SEER Clínico', DATASET_SEER_CLINICO),
                       ('Wisconsin Tumoral', DATASET_WISCONSIN_TUMORAL),
                       ('PCOS', DATASET_PCOS)]:
        exists = '✓' if os.path.exists(path) else '✗'
        print(f"  [{exists}] {name}: {path}")
