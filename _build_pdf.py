# -*- coding: utf-8 -*-
"""Gera o Relatório Técnico da Fase 2 em PDF (estilo da Fase 1), com números reais."""
import json, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Table, TableStyle, Image, ListFlowable, ListItem)

OUT = "results/fase2"
resumo = json.load(open(f"{OUT}/resumo.json"))
wis = pd.read_csv(f"{OUT}/wisconsin_baseline_vs_otimizado.csv")
seer = pd.read_csv(f"{OUT}/seer_baseline_vs_otimizado.csv")
exps = pd.read_csv(f"{OUT}/experimentos_configs.csv")
ORDER = ["Regressão Logística", "Árvore de Decisão", "KNN", "Random Forest", "XGBoost"]
def order_df(df): return df.set_index("modelo").loc[[m for m in ORDER if m in df.modelo.values]].reset_index()
wis, seer = order_df(wis), order_df(seer)

NAVY = colors.HexColor("#0b3d66"); BLUE = colors.HexColor("#0d6efd")
LIGHT = colors.HexColor("#eaf2fb"); GREY = colors.HexColor("#6c757d")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=15, spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=BLUE, fontSize=12, spaceBefore=10, spaceAfter=5)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=10.2, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=ss["Normal"], fontSize=8.5, leading=11, textColor=GREY)
CELL = ParagraphStyle("CELL", parent=ss["Normal"], fontSize=8.6, leading=11)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")

story = []
def P(t, s=BODY): story.append(Paragraph(t, s))
def SP(h=6): story.append(Spacer(1, h))
def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=6) for i in items],
                              bulletType="bullet", start="•", leftIndent=12))
    SP(4)

def metrics_table(df, header):
    data = [[Paragraph(h, CELLB) for h in header]]
    for _, r in df.iterrows():
        drec = (r.recall_otim - r.recall_base) * 100
        cor = "#0a7d33" if drec > 0 else ("#b00020" if drec < 0 else "#000000")
        data.append([
            Paragraph(r.modelo, CELL),
            Paragraph(f"{r.recall_base*100:.2f}", CELL), Paragraph(f"{r.recall_otim*100:.2f}", CELLB),
            Paragraph(f'<font color="{cor}">{drec:+.2f}</font>', CELL),
            Paragraph(f"{r.f1_base*100:.2f}", CELL), Paragraph(f"{r.f1_otim*100:.2f}", CELL),
            Paragraph(f"{r.acc_base*100:.2f}", CELL), Paragraph(f"{r.acc_otim*100:.2f}", CELL),
        ])
    t = Table(data, colWidths=[3.2*cm,1.6*cm,1.7*cm,1.5*cm,1.5*cm,1.5*cm,1.5*cm,1.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c9d6e5")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story.append(t); SP(6)

def img(path, w=15.5*cm):
    from PIL import Image as PImage
    iw, ih = PImage.open(path).size
    story.append(Image(path, width=w, height=w*ih/iw)); SP(6)

# ===== CAPA =====
story.append(Spacer(1, 2.2*cm))
P("RELATÓRIO TÉCNICO", ParagraphStyle("cap", parent=H1, alignment=TA_CENTER, fontSize=24, textColor=NAVY))
P("Tech Challenge — Fase 2", ParagraphStyle("cap2", parent=H1, alignment=TA_CENTER, fontSize=17, textColor=BLUE, spaceBefore=2))
SP(6)
P("Otimização de Modelos de Diagnóstico com Algoritmos Genéticos e Integração com LLM",
  ParagraphStyle("sub", parent=BODY, alignment=TA_CENTER, fontSize=12, textColor=colors.black))
P("Sistema Inteligente de Suporte ao Diagnóstico em Saúde da Mulher",
  ParagraphStyle("sub2", parent=BODY, alignment=TA_CENTER, fontSize=11, textColor=GREY))
SP(18)
P("Pós Tech FIAP — Inteligência Artificial para Devs (IADT)",
  ParagraphStyle("c", parent=BODY, alignment=TA_CENTER, fontSize=11))
P("Julho / 2026", ParagraphStyle("c2", parent=BODY, alignment=TA_CENTER, fontSize=11, textColor=GREY))
SP(20)
P("Integrantes do Grupo", ParagraphStyle("ig", parent=H2, alignment=TA_CENTER))
for line in ["RM: 372717 — Vinicius Luz", "RM: 370400 — Natalia Cabrera",
             "RM: 371644 — Rodrigo Teles", "RM: 374077 — Paola Mendes Bernardes",
             "RM: 369257 — Thamy Lais Ferreira Portugal dos Santos"]:
    P(line, ParagraphStyle("m", parent=BODY, alignment=TA_CENTER, spaceAfter=2))
SP(10)
P('Repositório: <link href="https://github.com/Vinny16044/tech-challenge-fase1"><font color="#0d6efd">github.com/Vinny16044/tech-challenge-fase1</font></link>',
  ParagraphStyle("r", parent=BODY, alignment=TA_CENTER))
P('Vídeo de demonstração: <link href="https://www.youtube.com/watch?v=IJi-tI1lMKw"><font color="#0d6efd">youtube.com/watch?v=IJi-tI1lMKw</font></link>',
  ParagraphStyle("r2", parent=BODY, alignment=TA_CENTER))
story.append(PageBreak())

print("parte 1 (capa) montada")
# salva o objeto story parcial via marcador; continua em _build_pdf2 (concatenado abaixo)

# ===== 1. SUMÁRIO EXECUTIVO =====
P("1. Sumário Executivo", H1)
P("Este relatório documenta a Fase 2 do Tech Challenge da Pós Tech FIAP (IADT). Partindo do "
  "sistema de apoio ao diagnóstico em saúde da mulher desenvolvido na Fase 1, o grupo optou pelo "
  "<b>Projeto 1 — Otimização de Modelos de Diagnóstico</b>, que combina duas técnicas centrais: "
  "(1) <b>Algoritmos Genéticos (AG)</b> para otimizar automaticamente os hiperparâmetros dos cinco "
  "modelos de classificação já existentes, priorizando o Recall; e (2) <b>integração com uma LLM "
  "pré-treinada (Google Gemini)</b> para traduzir predições e resultados numéricos em explicações "
  "em linguagem natural, acionáveis para profissionais de saúde.")
P("A base de código da Fase 1 foi reorganizada em módulos, sobre os quais foram construídas as "
  "camadas de otimização genética (<font face='Courier'>src/genetic/</font>), integração com LLM "
  "(<font face='Courier'>src/llm/</font>), monitoramento e testes automatizados. Os modelos "
  "otimizados foram comparados diretamente com os modelos originais sobre os datasets SEER (4.024 "
  "registros) e Wisconsin (569 registros).")
P("Os ganhos mais expressivos ocorreram na <b>Regressão Logística</b>: no dataset SEER, o Recall "
  "subiu de 80,12% para 89,19% (+9,07 pontos percentuais) e, no Wisconsin, de 97,37% para 99,12%. "
  "Nem todos os modelos melhoraram — alguns, já bem calibrados na Fase 1, mantiveram-se estáveis "
  "ou apresentaram leve queda sob o orçamento de busca adotado, resultado discutido criticamente "
  "no capítulo 14. Mantém-se a premissa ética da Fase 1: o sistema é ferramenta de <b>apoio</b>, "
  "não substitui a avaliação médica.")

# ===== 2. CONTEXTO =====
P("2. Contexto do Desafio", H1)
P("A Fase 2 solicita a evolução da solução da Fase 1 com técnicas de otimização e de integração "
  "com modelos de linguagem. Dentro do Projeto 1, o objetivo é responder:")
P("<i>Como um algoritmo genético pode ajustar automaticamente os hiperparâmetros dos modelos de "
  "diagnóstico para maximizar a sensibilidade (Recall) e, ao mesmo tempo, como uma LLM pode tornar "
  "as predições interpretáveis e acionáveis para o profissional de saúde?</i>")
P("A solução atende aos requisitos do desafio: implementação de um AG com codificação de genes, "
  "operadores de seleção, cruzamento e mutação e função fitness; integração do AG aos cinco modelos "
  "sobre os dois datasets; execução de experimentos com configurações distintas; comparação entre "
  "modelos originais e otimizados; integração com LLM e prompt engineering; monitoramento/logging; "
  "e testes automatizados.")

# ===== 3. VISÃO GERAL =====
P("3. Visão Geral da Solução", H1)
P("O fluxo técnico da Fase 2 estende o pipeline da Fase 1: carregamento e pré-processamento dos "
  "dados → treinamento dos modelos base (Fase 1) → <b>otimização genética dos hiperparâmetros</b> "
  "→ reavaliação e comparação (original vs. otimizado) → <b>interpretação via LLM</b> das predições "
  "e da otimização → monitoramento e testes. Os notebooks documentam a trilha analítica e os módulos "
  "em <font face='Courier'>src/</font> consolidam a execução programática e reutilizável.")

# ===== 4. ESTRUTURA =====
P("4. Estrutura do Repositório (Fase 2)", H1)
code = ("tech-challenge-fase2/\n"
"|-- main.py                  # CLI (--dataset, --explain)\n"
"|-- run_fase2_experiments.py # Runner dos experimentos do AG\n"
"|-- pyproject.toml           # Dependencias (Poetry) + extra 'llm'\n"
"|-- src/\n"
"|   |-- preprocessing.py     # Pre-processamento (base Fase 1)\n"
"|   |-- models.py            # 5 modelos ML (base Fase 1)\n"
"|   |-- evaluation.py        # Metricas, SHAP\n"
"|   |-- genetic/\n"
"|   |   |-- encoding.py      # Codificacao dos genes (Thamy)\n"
"|   |   |-- operators.py     # Selecao, crossover, mutacao (Thamy)\n"
"|   |   |-- fitness.py       # Funcao fitness (Thamy)\n"
"|   |   |-- optimizer.py     # Loop do AG + experimentos (Paola)\n"
"|   |   \\-- experiments.py   # Orquestracao dos experimentos (Paola)\n"
"|   |-- llm/\n"
"|   |   |-- prompts.py       # Prompt engineering (Vinicius)\n"
"|   |   \\-- interpreter.py   # Explicacoes + Q&A via Gemini (Vinicius)\n"
"|   \\-- monitoring.py        # Logging e tracking (Paola)\n"
"|-- tests/                   # Testes automatizados - pytest (Paola)\n"
"\\-- results/fase2/           # Comparativos, graficos e logs")
story.append(Table([[Paragraph(code.replace("\n","<br/>").replace(" ","&nbsp;"),
    ParagraphStyle("code", parent=SMALL, fontName="Courier", fontSize=7.6, leading=10, textColor=colors.black))]],
    colWidths=[16*cm], style=TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f4f6f8")),
    ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#c9d6e5")),("LEFTPADDING",(0,0),(-1,-1),8),
    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)])))
SP(6)

# ===== 5. BASE HERDADA =====
P("5. Base Herdada da Fase 1", H1)
P("A Fase 1 entregou cinco modelos de classificação treinados sobre dois datasets de câncer de "
  "mama, que são o ponto de partida da otimização:")
bullets([
 "<b>SEER Breast Cancer</b> — base clínica/prognóstica, 4.024 registros e 15 variáveis; alvo "
 "Status (Alive=1 / Dead=0), com desbalanceamento tratado por <font face='Courier'>class_weight</font>.",
 "<b>Wisconsin Breast Cancer (Diagnostic)</b> — base tumoral, 569 registros e 30 variáveis "
 "morfológicas; alvo Diagnóstico (Maligno=1 / Benigno=0).",
 "Modelos: Regressão Logística, Árvore de Decisão, KNN, Random Forest e XGBoost.",
])
P("Reforça-se a decisão técnica central da Fase 1, mantida na função fitness do AG: priorizar o "
  "<b>Recall</b>, pois em diagnóstico oncológico um falso negativo é clinicamente mais grave que um "
  "falso positivo.")
story.append(PageBreak())

# ===== 6. ALGORITMO GENÉTICO =====
P("6. Algoritmo Genético — Fundamentos e Implementação", H1)
P("O núcleo do AG (módulo <font face='Courier'>src/genetic/</font>) foi implementado do zero, sem "
  "bibliotecas prontas de otimização, seguindo a arquitetura abaixo.")
P("6.1 Codificação dos genes", H2)
P("Cada indivíduo (cromossomo) é um dicionário em que cada gene representa um hiperparâmetro do "
  "modelo. O espaço de busca é definido por modelo em <font face='Courier'>encoding.py</font>. "
  "Exemplos: Random Forest — n_estimators, max_depth, min_samples_split, min_samples_leaf, "
  "max_features, class_weight; XGBoost — n_estimators, max_depth, learning_rate, subsample, "
  "colsample_bytree; Regressão Logística — C, penalty, solver, class_weight.")
P("6.2 Operadores genéticos", H2)
bullets([
 "<b>Seleção</b>: por torneio (padrão) ou roleta, sobre a população ordenada por fitness.",
 "<b>Cruzamento (crossover)</b>: uniforme — cada gene do filho é herdado aleatoriamente de um dos "
 "dois pais, com taxa de cruzamento configurável.",
 "<b>Mutação</b>: substitui genes por valores válidos do espaço de busca, com taxa configurável, "
 "preservando a validade do indivíduo.",
 "<b>Elitismo</b>: os melhores indivíduos são preservados entre gerações.",
 "<b>Critérios de parada</b>: número máximo de gerações e parada antecipada por estagnação do "
 "fitness; reprodutibilidade garantida por <font face='Courier'>seed</font>.",
])
P("6.3 Função fitness", H2)
P("A fitness combina três métricas obtidas por <b>validação cruzada estratificada</b> (evitando "
  "overfitting à partição de teste), com pesos que priorizam a sensibilidade clínica:")
P("<b>fitness = 0,60 · Recall + 0,30 · F1-Score + 0,10 · Acurácia</b>",
  ParagraphStyle("f", parent=BODY, alignment=TA_CENTER, textColor=NAVY, fontSize=11))
P("O peso majoritário no Recall reflete a prioridade médica de minimizar falsos negativos, "
  "mantendo o F1-Score como salvaguarda contra desequilíbrio em cenários de classes desbalanceadas.")

# ===== 7. CONFIGURAÇÃO DOS EXPERIMENTOS =====
P("7. Configuração dos Experimentos", H1)
P("A otimização principal (comparação original vs. otimizado nos cinco modelos) usou a seguinte "
  "configuração do AG: população = 12, gerações = 6, taxa de cruzamento = 0,8, taxa de mutação = "
  "0,15, elitismo = 2, seleção por torneio e validação cruzada com 3 folds. Para os modelos mais "
  "custosos do SEER (Random Forest e XGBoost, sobre ~3.200 amostras de treino), adotou-se um "
  "orçamento reduzido (população = 8, gerações = 4) por razões de tempo de processamento — decisão "
  "registrada por transparência e retomada na discussão crítica.")

# ===== 8. METODOLOGIA =====
P("8. Metodologia de Otimização e Comparação", H1)
bullets([
 "Divisão treino/teste estratificada (80/20, seed = 42), idêntica para modelo original e otimizado.",
 "Modelo <b>original</b>: hiperparâmetros padrão da Fase 1. Modelo <b>otimizado</b>: melhores "
 "hiperparâmetros encontrados pelo AG.",
 "Métricas reportadas no conjunto de teste: Recall, F1-Score, Acurácia e AUC-ROC (média ponderada).",
 "Toda a execução é registrada (logging) e coberta por testes automatizados dos operadores "
 "genéticos, da função fitness e do pipeline de otimização.",
])
story.append(PageBreak())

print("parte 2 (cap 1-8) montada")

# ===== 9. RESULTADOS WISCONSIN =====
P("9. Resultados — Dataset Wisconsin", H1)
P("A tabela apresenta o desempenho no conjunto de teste, comparando os modelos originais (Fase 1) "
  "e otimizados pelo AG. Valores em %; Δ é a variação de Recall em pontos percentuais.")
metrics_table(wis, ["Modelo","Rec. Orig.","Rec. Otim.","Δ Rec.","F1 Orig.","F1 Otim.","Acc Orig.","Acc Otim."])
P("Destaques: a <b>Regressão Logística</b> subiu de 97,37% para 99,12% de Recall e o <b>XGBoost</b> "
  "de 96,49% para 97,37%. O <b>Random Forest</b>, já muito bem ajustado na Fase 1, teve leve queda "
  "(97,37% → 96,49%), indicando que a configuração padrão já estava próxima do ótimo para esse "
  "modelo neste dataset.", BODY)
img(f"{OUT}/fig_wisconsin_recall.png")
P("A evolução do fitness ao longo das gerações (Random Forest) evidencia a convergência do AG e a "
  "atuação do elitismo, com o melhor fitness global crescendo de forma monotônica.", SMALL)
img(f"{OUT}/fig_wisconsin_fitness.png", 13*cm)
story.append(PageBreak())

# ===== 10. RESULTADOS SEER =====
P("10. Resultados — Dataset SEER", H1)
P("No dataset clínico SEER, mais desbalanceado e de maior dimensionalidade categórica, os ganhos "
  "da otimização foram mais expressivos para os modelos lineares e de árvore.")
metrics_table(seer, ["Modelo","Rec. Orig.","Rec. Otim.","Δ Rec.","F1 Orig.","F1 Otim.","Acc Orig.","Acc Otim."])
P("Destaque principal: a <b>Regressão Logística</b> saltou de 80,12% para <b>89,19%</b> de Recall "
  "(+9,07 p.p.), e a <b>Árvore de Decisão</b> de 79,13% para 83,48% (+4,35 p.p.). Random Forest e "
  "XGBoost, já robustos, mantiveram-se estáveis — coerente com o menor espaço de melhoria e o "
  "orçamento de busca reduzido aplicado a eles no SEER.", BODY)
img(f"{OUT}/fig_seer_recall.png")
img(f"{OUT}/fig_seer_fitness.png", 13*cm)
story.append(PageBreak())

# ===== 11. ANÁLISE DOS EXPERIMENTOS =====
P("11. Análise dos Experimentos (Configurações do AG)", H1)
P("Para avaliar o impacto das configurações do AG, foram executados três experimentos sobre um "
  "modelo representativo (Random Forest / Wisconsin), variando população, gerações e taxas de "
  "mutação e cruzamento.")
edata = [[Paragraph(h, CELLB) for h in ["Experimento","Pop.","Ger.","Mut.","Cross.","Best fitness","Recall otim."]]]
for _, r in exps.iterrows():
    edata.append([Paragraph(str(r.experimento), CELL), Paragraph(str(r.populacao), CELL),
        Paragraph(str(r.geracoes), CELL), Paragraph(f"{r.taxa_mutacao}", CELL),
        Paragraph(f"{r.taxa_crossover}", CELL), Paragraph(f"{r.best_fitness:.4f}", CELLB),
        Paragraph(f"{r.recall_otim*100:.2f}%", CELL)])
et = Table(edata, colWidths=[4.2*cm,1.3*cm,1.3*cm,1.4*cm,1.6*cm,2.6*cm,2.6*cm])
et.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c9d6e5")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
    ("ALIGN",(1,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
story.append(et); SP(6)
P("Observa-se que configurações mais robustas (Exp B e C, com maior população e taxas de mutação/"
  "cruzamento mais altas) alcançaram um best fitness superior ao experimento exploratório (Exp A), "
  "confirmando que o dimensionamento da busca influencia a qualidade da solução. A partir de certo "
  "ponto, contudo, o ganho marginal diminui — reforçando a importância de equilibrar qualidade da "
  "otimização e custo computacional.")

# ===== 12. INTEGRAÇÃO LLM =====
P("12. Integração com LLM (Google Gemini)", H1)
P("A camada de interpretação (módulo <font face='Courier'>src/llm/</font>) integra a LLM "
  "pré-treinada <b>Google Gemini</b> (modelo <font face='Courier'>gemini-flash-latest</font>, tier "
  "gratuito) para transformar as saídas numéricas dos modelos — predição, probabilidade e "
  "importância de features via SHAP — em explicações claras e acionáveis para o profissional de "
  "saúde. A arquitetura separa o <i>prompt engineering</i> da lógica de chamada:")
bullets([
 "<b>prompts.py</b> — fixa o contexto médico, o tom e o formato; deixa explícito que o sistema é "
 "ferramenta de apoio e prioriza a leitura clínica sob a ótica do Recall.",
 "<b>interpreter.py</b> — classe <font face='Courier'>LLMInterpreter</font> com três operações: "
 "explicar predição, explicar a otimização (original vs. otimizado) e responder perguntas em "
 "linguagem natural (Q&amp;A) sobre as predições.",
 "<b>Degradação graciosa</b> — sem chave de API, o interpretador usa um modo <i>offline</i> "
 "determinístico, garantindo reprodutibilidade e permitindo testes sem rede.",
])
P("A chave é lida da variável de ambiente <font face='Courier'>GEMINI_API_KEY</font> (nunca "
  "versionada no código), e o comando <font face='Courier'>python main.py --dataset wisconsin "
  "--explain</font> gera a explicação ao final do pipeline. O prompt reforça, em toda resposta, que "
  "o resultado não substitui o julgamento médico.")
story.append(PageBreak())

# ===== 13. TESTES E MONITORAMENTO =====
P("13. Testes Automatizados e Monitoramento", H1)
P("A qualidade da Fase 2 é sustentada por uma suíte de testes automatizados (pytest) que cobre os "
  "operadores genéticos (garantindo que crossover e mutação produzem indivíduos válidos), a função "
  "fitness, o pipeline de otimização, o monitoramento e toda a camada de LLM (validada em modo "
  "offline determinístico). A suíte completa do projeto executa com <b>25 testes aprovados</b>. "
  "O módulo <font face='Courier'>monitoring.py</font> registra logs das execuções em "
  "<font face='Courier'>results/fase2/logs/</font>, permitindo o tracking do desempenho.")

# ===== 14. DISCUSSÃO CRÍTICA =====
P("14. Discussão Crítica", H1)
P("14.1 Quando o AG agrega valor", H2)
P("Os maiores ganhos ocorreram em modelos com padrão de configuração distante do ótimo para o "
  "dado — notadamente a Regressão Logística no SEER (+9,07 p.p. de Recall), em que o ajuste de "
  "<font face='Courier'>C</font>, penalidade e <font face='Courier'>class_weight</font> teve "
  "impacto direto na sensibilidade. Modelos de árvore/ensemble já bem calibrados na Fase 1 "
  "(Random Forest, XGBoost) apresentaram ganho pequeno ou nulo, e em alguns casos leve queda.")
P("14.2 Limitações", H2)
bullets([
 "O orçamento de busca reduzido para RF/XGBoost no SEER pode ter limitado o ganho desses modelos; "
 "uma busca mais ampla tende a recuperar ou superar o baseline.",
 "A fitness usa métricas ponderadas (weighted) por compatibilidade com a Fase 1; para o SEER, "
 "medir o Recall especificamente na classe de risco (Dead) pode revelar ganhos ainda mais nítidos.",
 "A avaliação da qualidade das explicações da LLM é qualitativa; uma rubrica formal de clareza, "
 "correção factual e utilidade clínica fortaleceria a análise.",
 "Limitações de representatividade do SEER (população majoritariamente norte-americana) para a "
 "realidade brasileira permanecem, conforme discutido na Fase 1.",
])

# ===== 15. ÉTICA =====
P("15. Riscos e Cuidados Éticos", H1)
P("O sistema é uma ferramenta de <b>apoio à decisão clínica</b> e não substitui a avaliação de um "
  "profissional de saúde. A integração com LLM foi desenhada para reforçar esse limite: todo texto "
  "gerado inclui aviso explícito de que não substitui o médico, e o prompt proíbe afirmações "
  "diagnósticas categóricas e a invenção de dados não fornecidos. A priorização do Recall reduz o "
  "risco de falsos negativos, o erro clinicamente mais grave. Chaves de API e dados sensíveis não "
  "são versionados no repositório.")

# ===== 16. EXECUÇÃO =====
P("16. Instruções de Execução", H1)
P("16.1 Pré-requisitos e instalação", H2)
P("Python 3.10+ e Poetry. Instalação: <font face='Courier'>poetry install</font> e "
  "<font face='Courier'>poetry shell</font>. Para a integração com a LLM: "
  "<font face='Courier'>poetry install --extras llm</font>.")
P("16.2 Otimização e comparação (AG)", H2)
P("<font face='Courier'>python run_fase2_experiments.py</font> — executa a otimização genética e "
  "gera os comparativos e gráficos em <font face='Courier'>results/fase2/</font>.")
P("16.3 Interpretação via LLM", H2)
P("Defina a chave gratuita do Gemini (obtida em aistudio.google.com/apikey) e rode a explicação:")
P("&nbsp;&nbsp;<font face='Courier'>export GEMINI_API_KEY=\"sua-chave\"</font>  (Windows: "
  "<font face='Courier'>set GEMINI_API_KEY=sua-chave</font>)", SMALL)
P("&nbsp;&nbsp;<font face='Courier'>python main.py --dataset wisconsin --explain</font>", SMALL)
P("&nbsp;&nbsp;<font face='Courier'>python notebooks/03_llm_interpret.py</font>", SMALL)
P("Sem a chave, a interpretação roda em modo offline determinístico. Um arquivo "
  "<font face='Courier'>.env</font> (fora do controle de versão) também é suportado.")
SP(10)
story.append(Table([[Paragraph(
 '<b>Repositório:</b> <link href="https://github.com/Vinny16044/tech-challenge-fase1">'
 '<font color="#0d6efd">github.com/Vinny16044/tech-challenge-fase1</font></link>&nbsp;&nbsp;|&nbsp;&nbsp;'
 '<b>Vídeo:</b> <link href="https://www.youtube.com/watch?v=IJi-tI1lMKw">'
 '<font color="#0d6efd">youtube.com/watch?v=IJi-tI1lMKw</font></link>', BODY)]],
 colWidths=[16*cm], style=TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),
 ("BOX",(0,0),(-1,-1),0.5,BLUE),("LEFTPADDING",(0,0),(-1,-1),10),
 ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)])))

# ===== FOOTER / BUILD =====
def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#c9d6e5")); canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(GREY)
    canvas.drawString(2*cm, 1.0*cm, "Tech Challenge — Fase 2 · FIAP Pós Tech IADT")
    canvas.drawRightString(A4[0]-2*cm, 1.0*cm, f"Página {doc.page}")
    canvas.restoreState()

OUTPDF = "Tech Challenge - Fase 2.pdf"
doc = SimpleDocTemplate(OUTPDF, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                        leftMargin=2*cm, rightMargin=2*cm, title="Relatório Técnico — Tech Challenge Fase 2")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("PDF gerado:", OUTPDF)
