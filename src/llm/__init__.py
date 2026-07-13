"""
Pacote de Integração com LLM
Responsável: Vinicius (P.O. e Desenvolvedor)

Camada de integração com uma LLM pré-treinada (Google Gemini) para traduzir as
saídas numéricas dos modelos (predição, probabilidade, features relevantes via
SHAP) e os resultados da otimização genética em explicações em linguagem natural,
acionáveis para profissionais de saúde. Inclui uma camada de perguntas e
respostas em linguagem natural sobre os diagnósticos.
"""

from .interpreter import LLMInterpreter, PatientCase, OptimizationCase

__all__ = ["LLMInterpreter", "PatientCase", "OptimizationCase"]
