# Tech Challenge - Fase 1

## Sistema Inteligente de Suporte ao Diagnóstico - Saúde e Segurança da Mulher

**FIAP POS TECH - IADT (IA para Devs)**

---

## 📋 Sobre o Projeto

Este projeto desenvolve um sistema de IA baseado em Machine Learning para classificação de exames médicos relacionados à saúde feminina. O foco principal é a detecção precoce de câncer de mama e outras condições ginecológicas.

## 👥 Equipe

| Integrante | Função |
|---|---|
| Vinicius | P.O. e Desenvolvedor |
| Natalia Cabrera | Desenvolvedora |
| Rodrigo | Desenvolvedor |
| Paola | Dados, Documentação e Dashboards |
| Thamy | Dados, Documentação e Dashboards |

## 📁 Estrutura do Projeto

```
tech-challenge-fase1/
├── README.md                         # Este arquivo
├── Dockerfile                        # Container Docker
├── requirements.txt                  # Dependências Python
├── data/
│   ├── raw/                          # Dados originais (não versionados)
│   └── processed/                    # Dados processados
├── notebooks/
│   ├── 01_eda.ipynb                  # Análise Exploratória de Dados
│   ├── 02_preprocessing.ipynb        # Pré-processamento
│   └── 03_modeling.ipynb             # Modelagem e Avaliação
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Carregamento de dados
│   ├── preprocessing.py              # Pipeline de pré-processamento
│   ├── models.py                     # Modelos de Machine Learning
│   └── evaluation.py                 # Avaliação e explicabilidade (SHAP)
├── dashboards/                       # Dashboards de visualização
├── reports/
│   ├── relatorio_tecnico.pdf         # Relatório final
│   └── figures/                      # Gráficos exportados
└── results/                          # Resultados dos modelos
```

## 📊 Dataset

**Principal:** [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- 569 amostras, 30 features numéricas
- Classificação binária: Maligno (M) vs. Benigno (B)

**Complementar:** [Polycystic Ovary Syndrome (PCOS)](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)

## 🚀 Como Executar

### ✅ Pré-requisitos
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### 💻 Instalação Local

```bash
# Clone o repositório
git clone <URL_DO_REPOSITORIO>
cd tech-challenge-fase1

# Instale as dependências com Poetry
poetry install

# Ative o ambiente virtual do Poetry
poetry shell

# Execute os notebooks
jupyter notebook
```

> **Nota:** Se preferir usar pip, gere o requirements.txt com:
> `poetry export -f requirements.txt --output requirements.txt`

### 🐳 Com Docker

```bash
# Build da imagem
docker build -t tech-challenge-fase1 .

# Executar o container
docker run -p 8888:8888 tech-challenge-fase1
```

Acesse o Jupyter em: http://localhost:8888

## 🤖 Modelos Utilizados

1. **Regressão Logística** - Baseline linear para classificação binária
2. **Random Forest** - Ensemble de árvores de decisão
3. **XGBoost** - Gradient boosting otimizado

## 📈 Métricas de Avaliação

- **Recall** (métrica principal) - Minimizar falsos negativos é crítico em diagnóstico médico
- **Accuracy** - Acurácia geral do modelo
- **F1-Score** - Equilíbrio entre precision e recall
- **AUC-ROC** - Capacidade de discriminação do modelo

## 🔍 Explicabilidade

Utilizamos SHAP (SHapley Additive exPlanations) para interpretar as predições dos modelos, garantindo transparência no processo de apoio ao diagnóstico.

## ⚠️ Aviso Importante

Este sistema é uma ferramenta de **apoio ao diagnóstico médico**. O(a) médico(a) sempre deve ter a palavra final no diagnóstico. O modelo não substitui a avaliação clínica profissional.

## 📄 Licença

Projeto acadêmico - FIAP POS TECH 2026
