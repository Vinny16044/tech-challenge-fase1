"""
Módulo de Carregamento de Dados
Responsável: Vinicius (P.O. e Desenvolvedor)

Carrega e disponibiliza os datasets para análise e modelagem.
"""

import pandas as pd
import os
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Carrega o dataset Breast Cancer Wisconsin.

    Se filepath for fornecido, carrega do CSV local.
    Caso contrário, carrega diretamente do sklearn.

    Args:
        filepath: Caminho para o arquivo CSV (opcional)

    Returns:
        DataFrame com os dados e a coluna 'target' (0=maligno, 1=benigno)
    """
    # 1) Se filepath foi fornecido e existe, carregar diretamente
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Dataset carregado do arquivo: {filepath}")
        normalize_cols = False
    else:
        # 2) Procurar CSV bruto do Kaggle em data/raw (priorizar para SEER)
        raw_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
        kaggle_path = None
        if os.path.exists(raw_dir):
            for f in os.listdir(raw_dir):
                if f.lower().endswith('.csv') and 'breast' in f.lower():
                    kaggle_path = os.path.join(raw_dir, f)
                    break

        if kaggle_path and os.path.exists(kaggle_path):
            # Carregar o CSV bruto (Kaggle) e preservar cabeçalhos originais
            df = pd.read_csv(kaggle_path)
            # remover espaços extras nos nomes de coluna
            df.columns = [c.strip() for c in df.columns]
            print(f"Dataset SEER carregado do CSV bruto: {kaggle_path}")
            # Mapear automaticamente colunas para os nomes SEER esperados, quando possível
            try:
                import re

                desired = [
                    'Age', 'Race', 'Marital Status', 'T Stage', 'N Stage',
                    '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Tumor Size'
                ]

                def _norm(s: str) -> str:
                    return re.sub(r'[^a-z0-9]', '', str(s).lower())

                existing = { _norm(c): c for c in df.columns }
                rename_map = {}

                for d in desired:
                    dnorm = _norm(d)
                    match = None
                    # 1) exact normalized match
                    if dnorm in existing:
                        match = existing[dnorm]
                    else:
                        # 2) token-match: all tokens of desired appear in existing col
                        tokens = re.findall(r'\w+', d.lower())
                        for k, orig in existing.items():
                            if all(tok in k for tok in tokens):
                                match = orig
                                break

                    if match and match not in rename_map:
                        rename_map[match] = d

                if rename_map:
                    df = df.rename(columns=rename_map)
                    print(f"Colunas mapeadas automaticamente: {rename_map}")
                normalize_cols = False
            except Exception:
                normalize_cols = False
        else:
            # 3) Tentar carregar os dados processados (X_train + y_train) se existirem
            processed_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
            x_train_path = os.path.normpath(os.path.join(processed_dir, 'X_train.csv'))
            y_train_path = os.path.normpath(os.path.join(processed_dir, 'y_train.csv'))

            if os.path.exists(x_train_path) and os.path.exists(y_train_path):
                X = pd.read_csv(x_train_path)
                y = pd.read_csv(y_train_path)
                # Se y tiver uma coluna 'target' ou for uma Series
                if 'target' in y.columns:
                    y_col = y['target']
                else:
                    # pegar a primeira coluna como target
                    y_col = y.iloc[:, 0]

                df = X.copy()
                df['target'] = y_col.values
                # opcional: mapear diagnóstico quando possível
                if set(df['target'].unique()) <= {0, 1}:
                    df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})

                print(f"Dataset carregado de arquivos processados: {x_train_path}, {y_train_path}")
                normalize_cols = True
            else:
                # 4) Último recurso: carregar o dataset exemplo do sklearn (Breast Cancer Wisconsin)
                data = load_breast_cancer()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
                df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
                print("Dataset carregado do sklearn.datasets (fallback)")
                normalize_cols = True

    print(f"Shape: {df.shape}")
    # Normalizar nomes de colunas para snake_case apenas quando aplicável
    if 'normalize_cols' not in locals():
        normalize_cols = True

    if normalize_cols:
        def _normalize(col: str) -> str:
            return col.strip().lower().replace(' ', '_').replace('-', '_')

        df.columns = [_normalize(c) for c in df.columns]

    print(f"Colunas: {list(df.columns)}")

    return df


def load_pcos_dataset(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset PCOS (Polycystic Ovary Syndrome).

    Args:
        filepath: Caminho para o arquivo CSV

    Returns:
        DataFrame com os dados do PCOS
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {filepath}\n"
            f"Baixe o dataset em: https://www.kaggle.com/datasets/reihanenamdari/breast-cancer"
        )

    df = pd.read_csv(filepath)
    print(f"Dataset Breast carregado: {filepath}")
    print(f"Shape: {df.shape}")

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Retorna informações resumidas sobre o dataset.

    Args:
        df: DataFrame a ser analisado

    Returns:
        Dicionário com informações do dataset
    """
    info = {
        'n_amostras': df.shape[0],
        'n_features': df.shape[1],
        'tipos_dados': df.dtypes.value_counts().to_dict(),
        'valores_nulos': df.isnull().sum().sum(),
        'colunas_com_nulos': df.columns[df.isnull().any()].tolist(),
        'duplicatas': df.duplicated().sum(),
    }

    return info


if __name__ == "__main__":
    # Teste rápido do carregamento
    df = load_breast_cancer_dataset()
    info = get_dataset_info(df)
    print("\n--- Informações do Dataset ---")
    for key, value in info.items():
        print(f"{key}: {value}")
