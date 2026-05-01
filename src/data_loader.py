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
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Dataset carregado do arquivo: {filepath}")
    else:
        # Carrega do sklearn como fallback
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
        print("Dataset carregado do sklearn.datasets")

    print(f"Shape: {df.shape}")
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
            f"Baixe o dataset em: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
        )

    df = pd.read_csv(filepath)
    print(f"Dataset Breast Cancer carregado: {filepath}")
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
