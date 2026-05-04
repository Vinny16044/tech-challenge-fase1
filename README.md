# Tech Challenge - Fase 1

## Sistema Inteligente de Suporte ao Diagnóstico - Saúde e Segurança da Mulher

**FIAP POS TECH - IADT (IA para Devs)**

---

## Sobre o Projeto

Este projeto desenvolve um sistema de Machine Learning para classificação de diagnóstico médico voltado à saúde feminina. Utilizamos dois datasets clínicos de câncer de mama para treinar, avaliar e comparar 5 modelos de classificação, com foco em minimizar falsos negativos (Recall como métrica principal) e garantir explicabilidade das predições via SHAP.

## Equipe

| Integrante | Função |
|---|---|
| Vinicius | P.O. e Desenvolvedor (integração, pipeline, CLI) |
| Natalia Cabrera | Desenvolvedora (EDA, pré-processamento) |
| Rodrigo | Desenvolvedor (modelos, avaliação, SHAP) |
| Paula | Dados, Documentação e Dashboards (Power BI) |
| Thamy | Dados, Documentação e Dashboards (Power BI) |

## Datasets

### Dataset Principal Clínico (SEER Breast Cancer)
- **Fonte:** SEER - Surveillance, Epidemiology, and End Results Program
- **Registros:** 4.024 pacientes
- **Features:** 18 (após feature engineering)
- **Target:** Status de sobrevivência (Alive/Dead)
- **Arquivo:** `data/raw/DatasetPrincipalClinico.csv`

### Dataset Complementar Tumoral (Wisconsin Breast Cancer)
- **Fonte:** UCI Machine Learning Repository (variante com dados tumorais)
- **Registros:** 569 pacientes
- **Features:** 30 medições de núcleos celulares
- **Target:** Diagnóstico (Maligno/Benigno)
- **Arquivo:** `data/raw/DatasetComplementarTumoral.csv`

### Dataset Complementar PCOS
- **Fonte:** Kaggle - Polycystic Ovary Syndrome
- **Registros:** 541
- **Arquivo:** `data/raw/PCOS_infertility.csv`

## Resultados

### Dataset SEER Clínico (Melhor: Random Forest)

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **Random Forest** | **89.69%** | **88.84%** | **89.69%** | **88.59%** | **83.81%** |
| XGBoost | 89.57% | 88.66% | 89.57% | 88.59% | 84.23% |
| KNN | 86.83% | 84.92% | 86.83% | 84.90% | 74.25% |
| Regressão Logística | 80.87% | 85.85% | 80.87% | 82.59% | 83.24% |
| Árvore de Decisão | 80.87% | 84.79% | 80.87% | 82.34% | 76.85% |

### Dataset Wisconsin Tumoral (Melhor: Random Forest)

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| **Random Forest** | **96.49%** | **96.68%** | **96.49%** | **96.45%** | **99.70%** |
| XGBoost | 95.61% | 95.69% | 95.61% | 95.58% | 99.64% |
| KNN | 92.98% | 93.31% | 92.98% | 92.85% | 97.54% |
| Regressão Logística | 89.47% | 89.88% | 89.47% | 89.56% | 96.16% |
| Árvore de Decisão | 87.72% | 87.89% | 87.72% | 87.78% | 87.30% |

## Estrutura do Projeto

```
tech-challenge-fase1/
├── main.py                              # CLI principal (python main.py --dataset all)
├── pyproject.toml                       # Dependências (Poetry)
├── Dockerfile                           # Container Docker
├── README.md                            # Este arquivo
│
├── src/
│   ├── __init__.py
│   ├── config.py                        # Configurações centralizadas
│   ├── pipeline.py                      # Pipeline end-to-end (orquestrador)
│   ├── data_load_breast_SEER.py         # Loader dataset SEER
│   ├── data_loader_breast_wisconsin.py  # Loader dataset Wisconsin
│   ├── preprocessing.py                 # Limpeza, encoding, scaling, split
│   ├── models.py                        # 5 modelos de classificação
│   └── evaluation.py                    # Métricas, plots, SHAP
│
├── notebooks/
│   ├── breast-cancer-SEER/
│   │   ├── 01_breast-cancer-SEER.ipynb      # EDA completo
│   │   ├── 02_preprocessing.ipynb           # Pipeline de pré-processamento
│   │   └── 03_modeling.ipynb                # Modelagem + SHAP
│   └── breast-cancer-wisconsin/
│       ├── 01_breast-cancer-wisconsin-data.ipynb  # EDA
│       ├── 02_preprocessing.ipynb                 # Pré-processamento
│       └── 03_modeling.ipynb                      # Modelagem + SHAP
│
├── data/
│   ├── raw/                             # Datasets originais (.csv)
│   └── processed/                       # Dados processados (train/test splits)
│
├── models/                              # Modelos treinados (.pkl)
│   ├── seer_Random_Forest.pkl
│   ├── seer_XGBoost.pkl
│   ├── wisconsin_Random_Forest.pkl
│   └── ...
│
├── reports/
│   ├── AI_Prediction_Results.xlsx       # Resultados em planilha
│   ├── seer_comparacao_modelos.csv      # Métricas comparativas SEER
│   ├── wisconsin_comparacao_modelos.csv # Métricas comparativas Wisconsin
│   └── figures/
│       ├── seer/                        # Figuras do pipeline SEER
│       └── wisconsin/                   # Figuras do pipeline Wisconsin
│
└── results/
    └── comparacao_modelos.csv           # Resultados consolidados
```

## Como Executar

### Pré-requisitos
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### Instalação

```bash
# Clone o repositório
git clone <URL_DO_REPOSITORIO>
cd tech-challenge-fase1

# Instale as dependências com Poetry
poetry install

# Ative o ambiente virtual
poetry shell
```

### Execução via CLI

```bash
# Pipeline completo (ambos datasets)
python main.py

# Apenas dataset SEER
python main.py --dataset seer

# Apenas dataset Wisconsin
python main.py --dataset wisconsin

# Sem gerar figuras (mais rápido)
python main.py --no-figures

# Sem salvar modelos em disco
python main.py --no-models
```

### Execução dos Notebooks

```bash
# Iniciar Jupyter
jupyter notebook

# Navegar para notebooks/breast-cancer-SEER/ ou notebooks/breast-cancer-wisconsin/
# Executar na ordem: 01_eda → 02_preprocessing → 03_modeling
```

### Com Docker

```bash
docker build -t tech-challenge-fase1 .
docker run -p 8888:8888 tech-challenge-fase1
```

## Modelos Utilizados

| # | Modelo | Tipo | Hiperparâmetros Chave |
|---|--------|------|----------------------|
| 1 | Regressão Logística | Linear | max_iter=10000, class_weight=balanced |
| 2 | Random Forest | Ensemble (Bagging) | n_estimators=100, class_weight=balanced |
| 3 | KNN | Instance-based | n_neighbors=5 |
| 4 | Árvore de Decisão | Árvore | max_depth=10, class_weight=balanced |
| 5 | XGBoost | Ensemble (Boosting) | n_estimators=100, max_depth=4, lr=0.1 |

## Métricas de Avaliação

- **Recall** (métrica principal) - Minimizar falsos negativos em diagnóstico oncológico
- **Accuracy** - Acurácia geral
- **Precision** - Evitar falsos positivos
- **F1-Score** - Equilíbrio precision/recall
- **AUC-ROC** - Capacidade discriminativa

## Explicabilidade (SHAP)

Utilizamos SHAP (SHapley Additive exPlanations) para garantir transparência nas predições. Os plots SHAP revelam quais features mais influenciam cada decisão, permitindo que profissionais de saúde entendam o raciocínio do modelo.

**Features mais importantes (SEER):** Survival Months, Node_Ratio, Tumor Size, N Stage, Estrogen Status

**Features mais importantes (Wisconsin):** concave points_worst, perimeter_worst, radius_worst, area_worst, concavity_mean

## Aviso Importante

Este sistema é uma ferramenta de **apoio ao diagnóstico médico**. O modelo não substitui a avaliação clínica profissional. O(a) médico(a) sempre deve ter a palavra final no diagnóstico.

## Licença

MIT - Projeto acadêmico FIAP POS TECH 2026
