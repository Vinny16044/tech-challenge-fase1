"""
Módulo de Interpretação via LLM
Responsável: Vinicius (P.O. e Desenvolvedor)

Camada que integra uma LLM pré-treinada (Google Gemini) para gerar explicações
em linguagem natural a partir das saídas dos modelos e da otimização genética,
além de uma camada de perguntas e respostas sobre as predições.

Arquitetura:
    - `LLMInterpreter` expõe uma interface simples e estável para o restante do
      projeto (explain_prediction, explain_optimization, answer_question).
    - A chamada à LLM é isolada em `_call_llm`. Se a biblioteca `google-genai`
      não estiver instalada, se não houver `GEMINI_API_KEY`, ou se a chamada
      falhar, o interpretador cai automaticamente num FALLBACK OFFLINE
      determinístico que gera o texto a partir dos próprios números.

O fallback garante que o sistema funcione (e seja testável) mesmo sem acesso à
internet ou sem chave de API — importante para reprodutibilidade e para a
demonstração/avaliação do projeto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import prompts


# ============================================================
# ESTRUTURAS DE ENTRADA (contratos)
# ============================================================

@dataclass
class PatientCase:
    """Representa o resultado de UMA predição individual a ser explicada."""
    model_name: str
    dataset: str  # "seer" ou "wisconsin"
    prediction: int
    probability: Optional[float] = None
    top_features: Optional[List[Tuple[str, float]]] = None
    metrics: Optional[Dict[str, float]] = None
    patient_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "prediction": self.prediction,
            "probability": self.probability,
            "top_features": self.top_features,
            "metrics": self.metrics,
            "patient_id": self.patient_id,
        }


@dataclass
class OptimizationCase:
    """Representa o comparativo original vs. otimizado de um modelo."""
    model_name: str
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    optimized_metrics: Dict[str, float] = field(default_factory=dict)
    best_params: Optional[Dict[str, Any]] = None
    ga_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "baseline_metrics": self.baseline_metrics,
            "optimized_metrics": self.optimized_metrics,
            "best_params": self.best_params,
            "ga_config": self.ga_config,
        }


# ============================================================
# INTERPRETADOR
# ============================================================

class LLMInterpreter:
    """
    Interpretador de resultados via LLM, com fallback offline.

    Args:
        provider: provedor da LLM. Atualmente "gemini" (Google Gemini).
        model: nome do modelo da LLM (ex.: "gemini-flash-latest", do tier gratuito).
        api_key: chave de API. Se None, lê de GEMINI_API_KEY (ou GOOGLE_API_KEY)
            no ambiente. NÃO deixe a chave hardcoded no código-fonte.
        temperature: criatividade da geração (baixa para contexto médico).
        force_offline: se True, ignora a API e usa sempre o fallback offline.
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-flash-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        force_offline: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.force_offline = force_offline
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        # Registra o modo usado na última chamada ("llm" ou "offline").
        self.last_mode: Optional[str] = None

    # ----------------------------------------------------------------
    # Disponibilidade da LLM
    # ----------------------------------------------------------------
    def is_llm_available(self) -> bool:
        """Verifica se é possível chamar a LLM (chave + biblioteca presentes)."""
        if self.force_offline or not self.api_key:
            return False
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return True

    # ----------------------------------------------------------------
    # Chamada de baixo nível à LLM (com fallback embutido)
    # ----------------------------------------------------------------
    def _call_llm(self, system: str, user: str, offline_fn) -> str:
        """
        Tenta chamar a LLM; em qualquer falha, usa `offline_fn()`.

        `offline_fn` é uma função sem argumentos que devolve o texto de fallback.
        """
        if not self.is_llm_available():
            self.last_mode = "offline"
            return offline_fn()

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.temperature,
                ),
            )
            self.last_mode = "llm"
            texto = (response.text or "").strip()
            # Garante que o disclaimer esteja presente.
            if "substitui" not in texto.lower():
                texto += "\n\n" + prompts.DISCLAIMER
            return texto
        except Exception as exc:  # noqa: BLE001 - degradação graciosa proposital
            self.last_mode = "offline"
            fallback = offline_fn()
            return (
                f"[Aviso: não foi possível usar a LLM ({exc}). "
                f"Explicação gerada pelo modo offline.]\n\n{fallback}"
            )

    # ----------------------------------------------------------------
    # API pública
    # ----------------------------------------------------------------
    def explain_prediction(self, case) -> str:
        """Gera explicação em linguagem natural de UMA predição."""
        data = case.to_dict() if isinstance(case, PatientCase) else dict(case)
        system, user = prompts.build_prediction_prompt(data)
        return self._call_llm(system, user, lambda: self._offline_prediction(data))

    def explain_optimization(self, case) -> str:
        """Gera explicação do ganho da otimização genética de um modelo."""
        data = case.to_dict() if isinstance(case, OptimizationCase) else dict(case)
        system, user = prompts.build_optimization_prompt(data)
        return self._call_llm(system, user, lambda: self._offline_optimization(data))

    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """Responde uma pergunta em linguagem natural com base no contexto."""
        system, user = prompts.build_qa_prompt(question, context)
        return self._call_llm(
            system, user, lambda: self._offline_qa(question, context)
        )

    # ----------------------------------------------------------------
    # FALLBACKS OFFLINE (determinísticos, baseados nos números)
    # ----------------------------------------------------------------
    def _offline_prediction(self, data: Dict[str, Any]) -> str:
        dataset = str(data.get("dataset", "")).lower()
        pred = data.get("prediction")
        significado = prompts.LABEL_MEANINGS.get(dataset, {}).get(
            pred, f"classe {pred}"
        )
        prob_txt = prompts.format_probability(data.get("probability"))
        ident = f" (caso {data['patient_id']})" if data.get("patient_id") else ""

        partes = [
            f"1. RESUMO{ident}: o modelo '{data.get('model_name', 'N/D')}' "
            f"indicou '{significado}' com probabilidade de {prob_txt} para a "
            f"classe positiva.",
        ]

        top = data.get("top_features")
        if top:
            partes.append("2. FATORES DE MAIOR PESO (SHAP):")
            partes.append(prompts.format_top_features(top))
            principal = top[0]
            direcao = "elevando" if principal[1] >= 0 else "reduzindo"
            partes.append(
                f"   A variável '{principal[0]}' foi a mais influente, "
                f"{direcao} a probabilidade da classe positiva."
            )
        else:
            partes.append(
                "2. FATORES DE MAIOR PESO: não foram fornecidas informações de "
                "importância de features (SHAP) para este caso."
            )

        metrics = data.get("metrics") or {}
        recall = metrics.get("recall")
        if isinstance(recall, (int, float)):
            partes.append(
                f"3. PRÓXIMOS PASSOS: o modelo apresenta Recall de "
                f"{recall * 100:.1f}%, métrica prioritária por minimizar falsos "
                f"negativos. Recomenda-se usar este resultado como apoio, "
                f"correlacionando-o ao quadro clínico e a exames complementares."
            )
        else:
            partes.append(
                "3. PRÓXIMOS PASSOS: utilize este resultado como apoio, "
                "correlacionando-o ao quadro clínico e a exames complementares."
            )

        partes.append(prompts.DISCLAIMER)
        return "\n".join(partes)

    def _offline_optimization(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline_metrics") or {}
        optimized = data.get("optimized_metrics") or {}
        nome = data.get("model_name", "N/D")

        linhas = [f"OTIMIZAÇÃO GENÉTICA — {nome}"]
        houve_ganho = False
        for chave, rotulo in [("recall", "Recall"), ("f1_score", "F1-Score"),
                              ("accuracy", "Acurácia")]:
            b, o = baseline.get(chave), optimized.get(chave)
            if isinstance(b, (int, float)) and isinstance(o, (int, float)):
                delta = (o - b) * 100
                if delta > 0:
                    houve_ganho = True
                linhas.append(
                    f"  - {rotulo}: {b * 100:.2f}% -> {o * 100:.2f}% "
                    f"({delta:+.2f} p.p.)"
                )

        rb, ro = baseline.get("recall"), optimized.get("recall")
        if isinstance(rb, (int, float)) and isinstance(ro, (int, float)):
            if ro > rb:
                linhas.append(
                    f"  O ganho no Recall (+{(ro - rb) * 100:.2f} p.p.) é "
                    f"especialmente relevante: reduz o risco de falsos negativos, "
                    f"o erro mais crítico em diagnóstico oncológico."
                )
            elif ro < rb:
                linhas.append(
                    "  Atenção: o Recall caiu após a otimização; avalie se o "
                    "ganho em outras métricas compensa a perda de sensibilidade."
                )

        params = data.get("best_params")
        if params:
            linhas.append("  Melhores hiperparâmetros:")
            for k, v in params.items():
                linhas.append(f"    - {k}: {v}")

        if houve_ganho:
            linhas.append(
                "  CONCLUSÃO: a otimização trouxe ganho de desempenho e o modelo "
                "otimizado tende a ser preferível ao original da Fase 1."
            )
        else:
            linhas.append(
                "  CONCLUSÃO: não houve ganho relevante; o modelo original da "
                "Fase 1 permanece adequado."
            )

        linhas.append(prompts.DISCLAIMER)
        return "\n".join(linhas)

    def _offline_qa(self, question: str, context: Dict[str, Any]) -> str:
        """
        Resposta offline simples: procura, por palavras-chave, o dado pedido
        dentro do contexto. Não é uma LLM — é um fallback determinístico.
        """
        q = question.lower()

        def achou(*termos: str) -> bool:
            return any(t in q for t in termos)

        if achou("probabilidade", "chance", "confiança"):
            prob = context.get("probability")
            if prob is not None:
                return (
                    f"A probabilidade estimada para a classe positiva é de "
                    f"{prompts.format_probability(prob)}.\n\n{prompts.DISCLAIMER}"
                )

        if achou("recall", "sensibilidade", "falso negativo"):
            metrics = context.get("metrics") or {}
            recall = metrics.get("recall")
            if isinstance(recall, (int, float)):
                return (
                    f"O Recall (sensibilidade) do modelo é de "
                    f"{recall * 100:.2f}%. Essa é a métrica prioritária por "
                    f"minimizar falsos negativos.\n\n{prompts.DISCLAIMER}"
                )

        if achou("feature", "fator", "variável", "variavel", "importante", "peso"):
            top = context.get("top_features")
            if top:
                return (
                    "As variáveis mais relevantes para esta predição foram:\n"
                    f"{prompts.format_top_features(top)}\n\n{prompts.DISCLAIMER}"
                )

        if achou("predição", "predicao", "diagnóstico", "diagnostico", "resultado"):
            dataset = str(context.get("dataset", "")).lower()
            pred = context.get("prediction")
            significado = prompts.LABEL_MEANINGS.get(dataset, {}).get(
                pred, f"classe {pred}"
            )
            if pred is not None:
                return (
                    f"A predição do modelo foi: {significado}.\n\n"
                    f"{prompts.DISCLAIMER}"
                )

        return (
            "A informação solicitada não está disponível no contexto de dados "
            "apresentado. Reformule a pergunta ou forneça os dados necessários.\n\n"
            f"{prompts.DISCLAIMER}"
        )


# ============================================================
# HELPERS DE INTEGRAÇÃO COM O PIPELINE
# ============================================================

def build_patient_case_from_pipeline(
    pipeline_result: Dict[str, Any],
    model_name: str,
    sample_index: int = 0,
    dataset: Optional[str] = None,
) -> PatientCase:
    """
    Conveniência: monta um PatientCase a partir do dicionário retornado pelos
    pipelines da Fase 1 (run_seer_pipeline / run_wisconsin_pipeline).
    """
    import numpy as np

    trained = pipeline_result["trained_models"]
    model = trained[model_name]
    X_test = pipeline_result["X_test"]
    results_df = pipeline_result.get("results")

    X_arr = np.asarray(X_test)
    X_sample = X_arr[sample_index:sample_index + 1]

    pred = int(model.predict(X_sample)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_sample)[0, 1])

    metrics = None
    if results_df is not None and model_name in getattr(results_df, "index", []):
        metrics = {
            col: float(results_df.loc[model_name, col])
            for col in results_df.columns
            if results_df.loc[model_name, col] is not None
        }

    if dataset is not None:
        ds = dataset
    elif "seer" in str(pipeline_result.get("dataset", "")).lower():
        ds = "seer"
    else:
        ds = "wisconsin"

    return PatientCase(
        model_name=model_name,
        dataset=ds,
        prediction=pred,
        probability=prob,
        metrics=metrics,
        patient_id=str(sample_index),
    )
