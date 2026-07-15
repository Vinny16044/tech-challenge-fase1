"""
Testes automatizados da camada de LLM (Vinicius).

Todos os testes rodam no modo OFFLINE (force_offline=True), garantindo que
sejam determinísticos e não dependam de chave de API ou de rede.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from llm import prompts
from llm.interpreter import LLMInterpreter, PatientCase, OptimizationCase


@pytest.fixture
def interpreter():
    return LLMInterpreter(force_offline=True)


@pytest.fixture
def caso():
    return PatientCase(
        model_name="Random Forest",
        dataset="wisconsin",
        prediction=1,
        probability=0.87,
        top_features=[("worst concave points", 0.152), ("worst perimeter", 0.121)],
        metrics={"recall": 0.9649, "f1_score": 0.9645, "accuracy": 0.9649},
        patient_id="p007",
    )


# ---------------------------------------------------------------- prompts

def test_prompt_predicao_contem_contexto_e_dados(caso):
    system, user = prompts.build_prediction_prompt(caso.to_dict())
    assert "apoio à decisão clínica" in system
    assert "Random Forest" in user
    assert "Tumor Maligno" in user  # rótulo mapeado do dataset wisconsin
    assert "worst concave points" in user


def test_format_probability():
    assert prompts.format_probability(0.87) == "87.0%"
    assert prompts.format_probability(None) == "não disponível"


# ---------------------------------------------------------------- predição

def test_explain_prediction_offline(interpreter, caso):
    texto = interpreter.explain_prediction(caso)
    assert interpreter.last_mode == "offline"
    assert "Tumor Maligno" in texto
    assert "87.0%" in texto
    assert "não substitui" in texto.lower()  # disclaimer presente


def test_explain_prediction_aceita_dict(interpreter, caso):
    # A API deve aceitar tanto dataclass quanto dict.
    texto = interpreter.explain_prediction(caso.to_dict())
    assert "Random Forest" in texto


# ---------------------------------------------------------------- otimização

def test_explain_optimization_detecta_ganho(interpreter):
    otim = OptimizationCase(
        model_name="Random Forest",
        baseline_metrics={"recall": 0.90, "f1_score": 0.90},
        optimized_metrics={"recall": 0.95, "f1_score": 0.94},
    )
    texto = interpreter.explain_optimization(otim)
    assert "+5.00 p.p." in texto
    assert "ganho" in texto.lower()
    assert "falsos negativos" in texto.lower()


def test_explain_optimization_sem_ganho(interpreter):
    otim = OptimizationCase(
        model_name="KNN",
        baseline_metrics={"recall": 0.90, "f1_score": 0.90},
        optimized_metrics={"recall": 0.88, "f1_score": 0.89},
    )
    texto = interpreter.explain_optimization(otim)
    assert "permanece adequado" in texto.lower() or "não houve ganho" in texto.lower()


# ---------------------------------------------------------------- Q&A

def test_qa_probabilidade(interpreter, caso):
    r = interpreter.answer_question("Qual a probabilidade estimada?", caso.to_dict())
    assert "87.0%" in r


def test_qa_recall(interpreter, caso):
    r = interpreter.answer_question("Qual o recall do modelo?", caso.to_dict())
    assert "96" in r


def test_qa_fora_do_contexto_recusa(interpreter, caso):
    r = interpreter.answer_question(
        "O paciente tem histórico familiar de câncer?", caso.to_dict()
    )
    assert "não está disponível" in r.lower()


# ---------------------------------------------------------------- disponibilidade

def test_is_llm_available_offline():
    interp = LLMInterpreter(force_offline=True, api_key="fake-key")
    assert interp.is_llm_available() is False


def test_sem_api_key_usa_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    interp = LLMInterpreter(api_key=None)
    interp.api_key = None  # ignora eventual .env presente no ambiente de dev
    assert interp.is_llm_available() is False
