"""
Runner RESUMÍVEL dos experimentos da Fase 2 — Otimização por Algoritmo Genético.

Usa o código real do projeto (src/genetic/*) para otimizar hiperparâmetros dos 5
modelos sobre SEER (src/Clinico.csv) e Wisconsin (sklearn), comparando desempenho
ORIGINAL (Fase 1) vs. OTIMIZADO (AG). Também roda 3 configurações do AG sobre um
modelo representativo.

Cada tarefa (dataset+modelo) é salva incrementalmente; reexecutar retoma de onde
parou. Saídas em results/fase2/.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from models import create_models
from evaluation import evaluate_model
from genetic.optimizer import (
    GeneticAlgorithmConfig, optimize_hyperparameters, train_and_evaluate_optimized_model,
)

OUT = os.path.join("results", "fase2")
os.makedirs(OUT, exist_ok=True)
SEED = 42
MODELS = ["Regressão Logística", "Árvore de Decisão", "KNN", "Random Forest", "XGBoost"]
CFG = dict(population_size=12, generations=6, mutation_rate=0.15,
           crossover_rate=0.8, cv_splits=3, seed=SEED)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_seer():
    df = pd.read_csv("src/Clinico.csv"); df.columns = [c.strip() for c in df.columns]
    y = df["Status"].map({"Alive": 1, "Dead": 0}).astype(int).values
    X = df.drop(columns=["Status"]).copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.factorize(X[c])[0]
    return StandardScaler().fit_transform(X.values.astype(float)), y


def load_wisconsin():
    d = load_breast_cancer()
    return StandardScaler().fit_transform(d.data), 1 - d.target  # 1 = Maligno


LOADERS = {"wisconsin": load_wisconsin, "seer": load_seer}


def split(tag):
    X, y = LOADERS[tag]()
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def done_models(tag):
    p = os.path.join(OUT, f"{tag}_baseline_vs_otimizado.csv")
    if os.path.exists(p):
        return set(pd.read_csv(p)["modelo"].tolist())
    return set()


def append_row(tag, row):
    p = os.path.join(OUT, f"{tag}_baseline_vs_otimizado.csv")
    df = pd.DataFrame([row])
    df.to_csv(p, mode="a", header=not os.path.exists(p), index=False)


def run_one(tag, model_name, Xtr, Xte, ytr, yte):
    cfg = GeneticAlgorithmConfig(**CFG)
    base_m = create_models()[model_name]; base_m.fit(Xtr, ytr)
    base = evaluate_model(base_m, Xte, yte, model_name)
    res = optimize_hyperparameters(model_name, Xtr, ytr, config=cfg)
    opt = train_and_evaluate_optimized_model(model_name, res.best_individual,
                                             Xtr, ytr, Xte, yte)["metrics"]
    append_row(tag, {
        "modelo": model_name,
        "recall_base": base["recall"], "recall_otim": opt["recall"],
        "f1_base": base["f1_score"], "f1_otim": opt["f1_score"],
        "acc_base": base["accuracy"], "acc_otim": opt["accuracy"],
        "auc_base": base["auc_roc"], "auc_otim": opt["auc_roc"],
        "best_fitness": res.best_fitness,
        "best_params": json.dumps(res.best_individual, ensure_ascii=False),
    })
    hp = os.path.join(OUT, f"_hist_{tag}.json")
    hist = json.load(open(hp)) if os.path.exists(hp) else {}
    hist[model_name] = [{k: v for k, v in d.items() if k != "best_individual"}
                        for d in res.history]
    json.dump(hist, open(hp, "w"))
    log(f"[{tag}] OK {model_name}: recall {base['recall']:.4f}->{opt['recall']:.4f}")


def driver(budget_s=38):
    t0 = time.time()
    with joblib.parallel_backend("threading", n_jobs=4):
        for tag in ("wisconsin", "seer"):
            feito = done_models(tag)
            pend = [m for m in MODELS if m not in feito]
            if not pend:
                continue
            Xtr, Xte, ytr, yte = split(tag)
            for m in pend:
                if time.time() - t0 > budget_s:
                    log("orçamento atingido; reexecute para continuar."); return False
                run_one(tag, m, Xtr, Xte, ytr, yte)
    log("TODOS os datasets concluídos.")
    return True


if __name__ == "__main__":
    driver()
