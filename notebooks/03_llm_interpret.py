"""
Notebook de Demonstração — Interpretação via LLM
Responsável: Vinicius (P.O. e Desenvolvedor)

Sem GEMINI_API_KEY definido, o LLMInterpreter usa automaticamente o modo
offline determinístico. Para usar a LLM real (Gemini, tier gratuito):
    export GEMINI_API_KEY="sua-chave"   (Windows: set GEMINI_API_KEY=sua-chave)
Chave gratuita em: https://aistudio.google.com/apikey
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from llm.interpreter import LLMInterpreter, PatientCase, OptimizationCase


def main():
    interpreter = LLMInterpreter(model="gemini-flash-latest", temperature=0.2)
    modo = "LLM (Gemini)" if interpreter.is_llm_available() else "OFFLINE (fallback)"
    print("=" * 70)
    print(f"Modo de execução: {modo}")
    print("=" * 70)

    caso = PatientCase(
        model_name="Random Forest",
        dataset="wisconsin",
        prediction=1,
        probability=0.87,
        top_features=[
            ("worst concave points", 0.152),
            ("worst perimeter", 0.121),
            ("mean concave points", 0.098),
            ("worst radius", 0.081),
            ("worst area", 0.067),
        ],
        metrics={"accuracy": 0.9649, "precision": 0.9668,
                 "recall": 0.9649, "f1_score": 0.9645, "auc_roc": 0.9970},
        patient_id="paciente_007",
    )
    print("\n### 1. EXPLICAÇÃO DE PREDIÇÃO ###\n")
    print(interpreter.explain_prediction(caso))

    otim = OptimizationCase(
        model_name="Random Forest",
        baseline_metrics={"recall": 0.9649, "f1_score": 0.9645, "accuracy": 0.9649},
        optimized_metrics={"recall": 0.9825, "f1_score": 0.9788, "accuracy": 0.9781},
        best_params={"n_estimators": 300, "max_depth": 12, "min_samples_split": 4},
        ga_config={"population_size": 30, "generations": 25,
                   "mutation_rate": 0.15, "crossover_rate": 0.8},
    )
    print("\n\n### 2. EXPLICAÇÃO DA OTIMIZAÇÃO ###\n")
    print(interpreter.explain_optimization(otim))

    contexto = caso.to_dict()
    perguntas = [
        "Qual foi a probabilidade estimada?",
        "Quais features mais pesaram nesta predição?",
        "Qual o recall do modelo?",
        "O paciente tem histórico familiar?",
    ]
    print("\n\n### 3. PERGUNTAS E RESPOSTAS ###\n")
    for p in perguntas:
        print(f"P: {p}")
        print(f"R: {interpreter.answer_question(p, contexto)}\n")


if __name__ == "__main__":
    main()
