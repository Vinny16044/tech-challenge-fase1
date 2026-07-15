import sys,warnings,os,json; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import joblib,pandas as pd, run_fase2_experiments as R
from genetic.optimizer import GeneticAlgorithmConfig, optimize_hyperparameters, train_and_evaluate_optimized_model
CONFIGS={
 "Exp A - Exploratório":dict(population_size=10,generations=6,mutation_rate=0.10,crossover_rate=0.7,cv_splits=3,seed=42),
 "Exp B - Balanceado":dict(population_size=12,generations=6,mutation_rate=0.15,crossover_rate=0.8,cv_splits=2,seed=42),
 "Exp C - Intensivo":dict(population_size=14,generations=6,mutation_rate=0.25,crossover_rate=0.9,cv_splits=2,seed=42),
}
P="results/fase2/experimentos_configs.csv"
done=set()
if os.path.exists(P): done=set(pd.read_csv(P)["experimento"])
which=sys.argv[1]
if which in done: print("já feito",which); sys.exit()
Xtr,Xte,ytr,yte=R.split("wisconsin")
cfg=GeneticAlgorithmConfig(**CONFIGS[which])
with joblib.parallel_backend("threading",n_jobs=2):
    res=optimize_hyperparameters("Random Forest",Xtr,ytr,config=cfg)
    opt=train_and_evaluate_optimized_model("Random Forest",res.best_individual,Xtr,ytr,Xte,yte)["metrics"]
row={"experimento":which,"populacao":cfg.population_size,"geracoes":cfg.generations,
 "taxa_mutacao":cfg.mutation_rate,"taxa_crossover":cfg.crossover_rate,
 "best_fitness":round(res.best_fitness,4),"recall_otim":round(opt["recall"],4),
 "f1_otim":round(opt["f1_score"],4),"acc_otim":round(opt["accuracy"],4),
 "geracoes_exec":len(res.history)}
pd.DataFrame([row]).to_csv(P,mode="a",header=not os.path.exists(P),index=False)
print("OK",which,"fit=%.4f recall=%.4f"%(res.best_fitness,opt["recall"]))
