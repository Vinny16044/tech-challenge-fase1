"""
Módulo de Pré-processamento de Dados
Responsável: Natalia Cabrera (Desenvolvedora)

Pipeline completo de limpeza, transformação e preparação dos dados
para modelagem de Machine Learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Verifica a qualidade dos dados: nulos, duplicatas, outliers.

    Args:
        df: DataFrame a ser analisado

    Returns:
        Relatório de qualidade dos dados
    """
    report = {
        'total_registros': len(df),
        'total_features': df.shape[1],
        'valores_nulos_por_coluna': df.isnull().sum().to_dict(),
        'percentual_nulos': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicatas': df.duplicated().sum(),
        'tipos_dados': df.dtypes.astype(str).to_dict(),
    }

    print("=" * 50)
    print("RELATÓRIO DE QUALIDADE DOS DADOS")
    print("=" * 50)
    print(f"Total de registros: {report['total_registros']}")
    print(f"Total de features: {report['total_features']}")
    print(f"Duplicatas: {report['duplicatas']}")
    print(f"Total de valores nulos: {df.isnull().sum().sum()}")
    print("=" * 50)

    return report


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Trata valores ausentes no DataFrame.

    Args:
        df: DataFrame com possíveis valores nulos
        strategy: Estratégia de imputação ('mean', 'median', 'drop')

    Returns:
        DataFrame sem valores nulos
    """
    df_clean = df.copy()

    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                fill_value = df_clean[col].mean() if strategy == 'mean' else df_clean[col].median()
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"  Coluna '{col}': {df[col].isnull().sum()} nulos preenchidos com {strategy} = {fill_value:.4f}")

    print(f"Valores nulos restantes: {df_clean.isnull().sum().sum()}")
    return df_clean


def encode_categorical(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Converte variáveis categóricas em numéricas.

    Args:
        df: DataFrame com variáveis categóricas
        columns: Lista de colunas para converter (None = detecta automaticamente)

    Returns:
        DataFrame com variáveis convertidas
    """
    df_encoded = df.copy()

    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

    le = LabelEncoder()
    for col in columns:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            print(f"  Coluna '{col}' codificada: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df_encoded


def scale_features(X: pd.DataFrame, method: str = 'standard') -> tuple:
    """
    Normaliza/padroniza as features numéricas.

    Args:
        X: DataFrame com as features
        method: 'standard' (StandardScaler) ou 'minmax' (MinMaxScaler)

    Returns:
        Tuple (X_scaled, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    print(f"Features padronizadas com {method}Scaler")
    return X_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Separa os dados em conjuntos de treino e teste.

    Args:
        X: Features
        y: Target
        test_size: Proporção do conjunto de teste
        random_state: Seed para reprodutibilidade

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dados separados:")
    print(f"  Treino: {X_train.shape[0]} amostras ({(1-test_size)*100:.0f}%)")
    print(f"  Teste:  {X_test.shape[0]} amostras ({test_size*100:.0f}%)")
    print(f"  Distribuição no treino: {y_train.value_counts().to_dict()}")
    print(f"  Distribuição no teste:  {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def full_preprocessing_pipeline(df: pd.DataFrame,
                                 target_col: str = 'target',
                                 drop_cols: list = None,
                                 test_size: float = 0.2) -> dict:
    """
    Pipeline completo de pré-processamento.

    Args:
        df: DataFrame original
        target_col: Nome da coluna target
        drop_cols: Colunas a remover (ex: 'id', 'diagnosis')
        test_size: Proporção do conjunto de teste

    Returns:
        Dicionário com dados processados e objetos do pipeline
    """
    print("\n" + "=" * 60)
    print("PIPELINE DE PRÉ-PROCESSAMENTO")
    print("=" * 60)

    # 1. Remover colunas desnecessárias
    df_proc = df.copy()
    if drop_cols:
        df_proc = df_proc.drop(columns=[c for c in drop_cols if c in df_proc.columns])
        print(f"\n1. Colunas removidas: {drop_cols}")

    # 2. Tratar valores ausentes
    print("\n2. Tratamento de valores ausentes:")
    df_proc = handle_missing_values(df_proc, strategy='median')

    # 3. Codificar variáveis categóricas
    print("\n3. Codificação de variáveis categóricas:")
    df_proc = encode_categorical(df_proc)

    # 4. Separar features e target
    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]
    print(f"\n4. Features: {X.shape[1]} | Target: '{target_col}'")

    # 5. Padronizar features
    print("\n5. Padronização:")
    X_scaled, scaler = scale_features(X, method='standard')

    # 6. Split treino/teste
    print("\n6. Separação treino/teste:")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=test_size)

    print("\n" + "=" * 60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns),
    }
