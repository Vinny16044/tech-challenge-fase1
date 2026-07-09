import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Permite importar módulos da pasta src quando executado a partir da raiz do projeto.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from pipeline import run_wisconsin_pipeline
from models import create_models
from genetic.optimizer import GeneticAlgorithmConfig, GeneticHyperparameterOptimizer


# 1. Executa o pipeline da Fase 1 sem salvar figuras/modelos.
# Para testar com SEER, troque para run_seer_pipeline.
pipeline_result = run_wisconsin_pipeline(save_figures=False, save_models=False)

X_train = pipeline_result["X_train"]
y_train = pipeline_result["y_train"]

# 2. Usa os mesmos modelos definidos na Fase 1.
model_factories = {
    model_name: model.__class__
    for model_name, model in create_models().items()
}

# 3. Configuração do Algoritmo Genético.
config = GeneticAlgorithmConfig(
    population_size=20,
    generations=15,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism_size=2,
    tournament_size=3,
    selection_method="tournament",
    cv_splits=5,
    seed=42,
    early_stopping_rounds=5,
)

# 4. Escolha do modelo que será otimizado.
# Opções:
# - "Regressão Logística"
# - "Random Forest"
# - "KNN"
# - "Árvore de Decisão"
# - "XGBoost"
model_name = "Random Forest"

optimizer = GeneticHyperparameterOptimizer(
    model_name=model_name,
    model_factory=model_factories[model_name],
    config=config,
)

result = optimizer.optimize(X_train, y_train)

print("\nMelhor indivíduo encontrado:")
print(result.best_individual)

print("\nMelhor fitness:")
print(result.best_fitness)

history_df = pd.DataFrame(result.history)
print("\nHistórico:")
print(history_df[["generation", "best_fitness", "avg_fitness", "global_best_fitness"]])

plt.figure(figsize=(10, 5))
plt.plot(history_df["generation"], history_df["best_fitness"], marker="o", label="Melhor fitness da geração")
plt.plot(history_df["generation"], history_df["avg_fitness"], marker="o", label="Fitness médio")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title(f"Evolução do fitness - {model_name}")
plt.legend()
plt.grid(True)
plt.show()