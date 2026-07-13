"""
Módulo de Prompt Engineering
Responsável: Vinicius (P.O. e Desenvolvedor)

Concentra todo o "prompt engineering" da integração com a LLM. A ideia é
manter os prompts separados da lógica de chamada (interpreter.py), facilitando
ajustes de tom, contexto e formato sem tocar no código de integração.

Princípios adotados (conforme orientações técnicas da Fase 2):
- Fixar o CONTEXTO MÉDICO e o PAPEL da LLM (assistente de apoio).
- Deixar EXPLÍCITO que o sistema é ferramenta de APOIO e NUNCA substitui o
  julgamento do profissional de saúde.
- Definir TOM (claro, técnico, sóbrio) e FORMATO da resposta.
- Transformar dados numéricos (predição, probabilidade, SHAP) em insights
  acionáveis, sem inventar informação que não esteja nos dados fornecidos.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# CONSTANTES DE CONTEXTO
# ============================================================

DISCLAIMER = (
    "⚠️ Este conteúdo é gerado por um sistema de apoio à decisão clínica e tem "
    "caráter exclusivamente informativo. NÃO substitui a avaliação, o diagnóstico "
    "ou a conduta de um profissional de saúde habilitado."
)

SYSTEM_MEDICAL = (
    "Você é um assistente de apoio à decisão clínica em oncologia mamária, "
    "especializado em traduzir saídas de modelos de Machine Learning para uma "
    "linguagem clara e acionável para profissionais de saúde.\n\n"
    "REGRAS OBRIGATÓRIAS:\n"
    "1. Você é uma FERRAMENTA DE APOIO. Nunca afirme diagnósticos como certeza "
    "absoluta nem substitua o julgamento clínico. Sempre reforce que a decisão "
    "final é do profissional de saúde.\n"
    "2. Baseie-se EXCLUSIVAMENTE nos dados numéricos fornecidos (predição, "
    "probabilidade, métricas e importância de features via SHAP). Não invente "
    "valores, exames ou informações que não estejam presentes.\n"
    "3. Em diagnóstico oncológico, o RECALL (sensibilidade) é a métrica mais "
    "crítica, pois um falso negativo é mais perigoso que um falso positivo. "
    "Contextualize os resultados sob essa ótica quando relevante.\n"
    "4. Use tom técnico, sóbrio e empático. Evite alarmismo e jargão "
    "desnecessário.\n"
    "5. Responda sempre em português do Brasil."
)

# Mapa de rótulos por dataset, para dar sentido clínico à predição binária.
LABEL_MEANINGS: Dict[str, Dict[int, str]] = {
    "seer": {1: "Sobrevivência esperada (Alive)", 0: "Óbito esperado (Dead)"},
    "wisconsin": {1: "Tumor Maligno", 0: "Tumor Benigno"},
}


# ============================================================
# HELPERS DE FORMATAÇÃO
# ============================================================

def format_probability(prob: Optional[float]) -> str:
    """Formata uma probabilidade (0-1) como percentual legível."""
    if prob is None:
        return "não disponível"
    return f"{prob * 100:.1f}%"


def format_top_features(top_features: Optional[List[Tuple[str, float]]],
                        max_items: int = 5) -> str:
    """
    Formata a lista de features mais relevantes (nome, contribuição SHAP)
    em texto tabular simples.
    """
    if not top_features:
        return "  (nenhuma informação de importância de features fornecida)"
    linhas = []
    for i, (nome, valor) in enumerate(top_features[:max_items], start=1):
        direcao = "aumenta" if valor >= 0 else "reduz"
        linhas.append(
            f"  {i}. {nome}: contribuição SHAP {valor:+.4f} "
            f"({direcao} a probabilidade da classe positiva)"
        )
    return "\n".join(linhas)


def format_metrics(metrics: Optional[Dict[str, float]]) -> str:
    """Formata um dicionário de métricas em texto legível."""
    if not metrics:
        return "  (métricas não fornecidas)"
    ordem = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    rotulos = {
        "accuracy": "Acurácia",
        "precision": "Precisão",
        "recall": "Recall (sensibilidade)",
        "f1_score": "F1-Score",
        "auc_roc": "AUC-ROC",
    }
    linhas = []
    for chave in ordem:
        if chave in metrics and metrics[chave] is not None:
            linhas.append(f"  - {rotulos[chave]}: {metrics[chave] * 100:.2f}%")
    # Inclui eventuais métricas extras não previstas na ordem.
    for chave, valor in metrics.items():
        if chave not in ordem and isinstance(valor, (int, float)):
            linhas.append(f"  - {chave}: {valor}")
    return "\n".join(linhas) if linhas else "  (métricas não fornecidas)"


# ============================================================
# BUILDERS DE PROMPT
# ============================================================

def build_prediction_prompt(case: Dict[str, Any]) -> Tuple[str, str]:
    """
    Monta o par (system, user) para explicar UMA predição individual.

    Espera um dicionário `case` com as chaves:
        - model_name (str): nome do modelo que gerou a predição.
        - dataset (str): "seer" ou "wisconsin" (usado para dar sentido ao rótulo).
        - prediction (int): classe prevista (0/1).
        - probability (float|None): probabilidade da classe positiva.
        - top_features (list[(str, float)]|None): features mais relevantes (SHAP).
        - metrics (dict|None): métricas do modelo, para contextualizar confiança.
        - patient_id (str|None): identificador opcional do caso.
    """
    dataset = str(case.get("dataset", "")).lower()
    pred = case.get("prediction")
    significado = LABEL_MEANINGS.get(dataset, {}).get(
        pred, f"classe {pred}"
    )
    patient_id = case.get("patient_id")
    ident = f" (caso {patient_id})" if patient_id else ""

    user = (
        f"Explique, para um profissional de saúde, o resultado abaixo gerado "
        f"pelo modelo de apoio ao diagnóstico{ident}.\n\n"
        f"MODELO: {case.get('model_name', 'não informado')}\n"
        f"DATASET: {case.get('dataset', 'não informado')}\n"
        f"PREDIÇÃO: {significado} (classe {pred})\n"
        f"PROBABILIDADE (classe positiva): {format_probability(case.get('probability'))}\n\n"
        f"FEATURES MAIS RELEVANTES PARA ESTA PREDIÇÃO (SHAP):\n"
        f"{format_top_features(case.get('top_features'))}\n\n"
        f"DESEMPENHO GERAL DO MODELO:\n"
        f"{format_metrics(case.get('metrics'))}\n\n"
        f"Produza uma explicação estruturada em três partes:\n"
        f"1. Resumo do resultado em uma frase.\n"
        f"2. Fatores que mais pesaram nesta predição (interprete o SHAP em "
        f"linguagem clínica).\n"
        f"3. Recomendação de próximos passos, deixando claro que é apoio à "
        f"decisão. Encerre com o aviso de que não substitui o médico."
    )
    return SYSTEM_MEDICAL, user


def build_optimization_prompt(case: Dict[str, Any]) -> Tuple[str, str]:
    """
    Monta o par (system, user) para explicar o GANHO da otimização genética
    de um modelo (Fase 1 original vs. otimizado por AG).

    Espera um dicionário `case` com as chaves:
        - model_name (str)
        - baseline_metrics (dict): métricas do modelo original da Fase 1.
        - optimized_metrics (dict): métricas do modelo otimizado por AG.
        - best_params (dict|None): melhores hiperparâmetros encontrados.
        - ga_config (dict|None): configuração do AG (população, gerações, taxas).
    """
    baseline = case.get("baseline_metrics") or {}
    optimized = case.get("optimized_metrics") or {}

    # Calcula deltas das principais métricas, se possível.
    deltas = []
    for chave, rotulo in [("recall", "Recall"), ("f1_score", "F1-Score"),
                          ("accuracy", "Acurácia")]:
        b, o = baseline.get(chave), optimized.get(chave)
        if isinstance(b, (int, float)) and isinstance(o, (int, float)):
            deltas.append(
                f"  - {rotulo}: {b * 100:.2f}% -> {o * 100:.2f}% "
                f"({(o - b) * 100:+.2f} p.p.)"
            )
    deltas_txt = "\n".join(deltas) if deltas else "  (deltas não calculáveis)"

    params = case.get("best_params")
    params_txt = (
        "\n".join(f"  - {k}: {v}" for k, v in params.items())
        if params else "  (não informados)"
    )

    ga = case.get("ga_config")
    ga_txt = (
        "\n".join(f"  - {k}: {v}" for k, v in ga.items())
        if ga else "  (não informada)"
    )

    user = (
        f"Explique, para um profissional de saúde e para a banca do projeto, o "
        f"resultado da otimização por Algoritmo Genético do modelo "
        f"'{case.get('model_name', 'não informado')}'.\n\n"
        f"MÉTRICAS ORIGINAIS (Fase 1):\n{format_metrics(baseline)}\n\n"
        f"MÉTRICAS OTIMIZADAS (Fase 2 - AG):\n{format_metrics(optimized)}\n\n"
        f"VARIAÇÃO POR MÉTRICA:\n{deltas_txt}\n\n"
        f"MELHORES HIPERPARÂMETROS ENCONTRADOS:\n{params_txt}\n\n"
        f"CONFIGURAÇÃO DO ALGORITMO GENÉTICO:\n{ga_txt}\n\n"
        f"Produza uma explicação que: (1) diga se houve ganho e onde ele foi "
        f"mais relevante; (2) destaque o impacto no Recall sob a ótica de "
        f"segurança do diagnóstico; (3) comente de forma sóbria se o ganho "
        f"justifica adotar o modelo otimizado."
    )
    return SYSTEM_MEDICAL, user


def build_qa_prompt(question: str, context: Dict[str, Any]) -> Tuple[str, str]:
    """
    Monta o par (system, user) para a camada de perguntas e respostas em
    linguagem natural sobre as predições.

    `context` é um dicionário livre com os dados disponíveis (predição,
    probabilidade, features, métricas, etc.). A LLM deve responder APENAS com
    base nesse contexto, sem extrapolar.
    """
    # Serializa o contexto de forma legível.
    linhas_ctx = []
    for chave, valor in context.items():
        if chave == "top_features":
            linhas_ctx.append("FEATURES RELEVANTES (SHAP):")
            linhas_ctx.append(format_top_features(valor))
        elif chave in ("metrics", "baseline_metrics", "optimized_metrics"):
            linhas_ctx.append(f"{chave.upper()}:")
            linhas_ctx.append(format_metrics(valor))
        else:
            linhas_ctx.append(f"{chave}: {valor}")
    contexto_txt = "\n".join(linhas_ctx) if linhas_ctx else "(sem contexto)"

    system = (
        SYSTEM_MEDICAL
        + "\n\nVocê está no modo PERGUNTAS E RESPOSTAS. Responda de forma "
        "objetiva e curta, usando SOMENTE o contexto fornecido. Se a resposta "
        "não estiver no contexto, diga explicitamente que a informação não está "
        "disponível nos dados apresentados — não invente."
    )
    user = (
        f"CONTEXTO DISPONÍVEL:\n{contexto_txt}\n\n"
        f"PERGUNTA DO PROFISSIONAL: {question}\n\n"
        f"Responda com base apenas no contexto acima."
    )
    return system, user
