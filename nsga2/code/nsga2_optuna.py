"""Optuna hyperparameter search for NSGA-II (all features, 500k, chronological 70/15/15 split).
Objective = 3D hypervolume of the final front measured on VALIDATION data (test data is never used to tune).
Population size is FIXED (default 200) because hypervolume grows with archive size; tuning it would just reward big fronts.
Usage: python nsga2_optuna.py [csv_path] [n_trials] [pop] [out_dir] [workers]
"""
import sys, os, time, random, copy, json
import numpy as np, pandas as pd, optuna
from multiprocessing import Pool
from deap import base, creator, tools
from pymoo.indicators.hv import HV

# ignore the "-f kernel.json" arguments Jupyter/Colab passes when this code is pasted into a notebook cell
_A = [a for a in sys.argv[1:] if a != "-f" and not a.endswith(".json")]
CSV = _A[0] if len(_A) > 0 else "synthetic_fraud_500k.csv"
N_TRIALS = int(_A[1]) if len(_A) > 1 else 24
POP = int(_A[2]) if len(_A) > 2 else 200
OUT = _A[3] if len(_A) > 3 else "optuna_out"
WORKERS = int(_A[4]) if len(_A) > 4 else 2
os.makedirs(OUT, exist_ok=True)
LABEL, TIME = "Fraud_Label", "Time"
CAP, MAX_PATH_LEN, MAX_PATHS, MIN_TP = 160, 5, 30, 5
Q = np.linspace(0.05, 0.95, 19)

df = pd.read_csv(CSV).sort_values(TIME).reset_index(drop=True)
CAT = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
NUM = [c for c in df.columns if c not in CAT + [LABEL, TIME]]
n = len(df); i1, i2 = int(.70 * n), int(.85 * n)
parts = {"train": df.iloc[:i1], "val": df.iloc[i1:i2], "test": df.iloc[i2:]}
y = {k: v[LABEL].values.astype(bool) for k, v in parts.items()}
conds = []
for f in NUM:
    for t in np.unique(np.quantile(parts["train"][f].values, Q)):
        conds.append((f, "<=", float(t))); conds.append((f, ">", float(t)))
for f in CAT:
    for v in sorted(parts["train"][f].unique()): conds.append((f, "=", v))
def _mask(v, c):
    f, op, t = c; col = v[f].values
    return (col <= t) if op == "<=" else (col > t) if op == ">" else (col == t)
masks = {k: np.stack([_mask(v, c) for c in conds]) for k, v in parts.items()}
NC = len(conds)

def metrics(rs, split):
    M = masks[split]; pred = np.zeros(M.shape[1], bool)
    for path in rs:
        p = M[path[0]].copy()
        for c in path[1:]: p &= M[c]
        pred |= p
    yy = y[split]; tp = int((pred & yy).sum()); fp = int((pred & ~yy).sum())
    return tp, (tp / (tp + fp) if tp + fp else 0.0), tp / yy.sum()
tc = lambda rs: sum(len(p) for p in rs)
def rand_path(): return [random.randrange(NC) for _ in range(random.randint(1, 3))]
def rand_ruleset(): return [rand_path() for _ in range(random.randint(1, 6))]
def repair(rs):
    rs = [list(dict.fromkeys(p))[:MAX_PATH_LEN] for p in rs if p][:MAX_PATHS]
    while tc(rs) > CAP and rs:
        i = max(range(len(rs)), key=lambda j: len(rs[j]))
        if len(rs[i]) > 1: rs[i].pop(random.randrange(len(rs[i])))
        else: rs.pop(i)
    return rs or [rand_path()]
def mutate(rs):
    r = random.random()
    if r < 0.25 and len(rs) < MAX_PATHS: rs.append(rand_path())
    elif r < 0.40 and len(rs) > 1: rs.pop(random.randrange(len(rs)))
    else:
        p = random.choice(rs); r2 = random.random()
        if r2 < 0.4: p[random.randrange(len(p))] = random.randrange(NC)
        elif r2 < 0.7 and len(p) < MAX_PATH_LEN: p.append(random.randrange(NC))
        elif len(p) > 1: p.pop(random.randrange(len(p)))
        else: p[0] = random.randrange(NC)
def cx(a, b):
    pool = [list(p) for p in a + b]; random.shuffle(pool)
    k = random.randint(1, max(1, len(pool) - 1)); a[:] = pool[:k]; b[:] = pool[k:] or [rand_path()]
creator.create("Fit3", base.Fitness, weights=(1.0, 1.0, -1.0)); creator.create("RS", list, fitness=creator.Fit3)
def evaluate(rs):
    tp, p, r = metrics(rs, "train")
    return (0.0, 0.0, float(CAP)) if tp < MIN_TP else (p, r, float(tc(rs)))
hv_ind = HV(ref_point=np.array([1., 1., 1.]))

def run_nsga2(pop_n, gens, cxpb, mutpb, seed):
    random.seed(seed); np.random.seed(seed)
    pop = [creator.RS(rand_ruleset()) for _ in range(pop_n)]
    for i in pop: i[:] = repair(i); i.fitness.values = evaluate(i)
    pop = tools.selNSGA2(pop, pop_n)
    for g in range(gens):
        off = [copy.deepcopy(i) for i in tools.selTournamentDCD(pop, pop_n)]
        for a, b in zip(off[::2], off[1::2]):
            if random.random() < cxpb: cx(a, b); del a.fitness.values, b.fitness.values
        for i in off:
            if random.random() < mutpb: mutate(i); del i.fitness.values
        for i in off:
            i[:] = repair(i)
            if not i.fitness.valid: i.fitness.values = evaluate(i)
        pop = tools.selNSGA2(pop + off, pop_n)
    front = tools.sortNondominated(pop, pop_n, first_front_only=True)[0]
    def hv_on(split):
        pts = []
        for i in front:
            _, p, r = metrics(i, split); pts.append([1 - p, 1 - r, min(tc(i) / CAP, 1.0)])
        return float(hv_ind(np.array(pts)))
    f1s = []
    for i in front:
        _, p, r = metrics(i, "val"); f1s.append(2 * p * r / (p + r) if p + r else 0)
    return {"hv_val": hv_on("val"), "hv_test": hv_on("test"), "best_val_f1": max(f1s), "front": len(front)}

DB = f"sqlite:///{OUT}/optuna_study.db"
STUDY = f"nsga2_pop{POP}"      # one study per population size, so runs with different pop never mix
def objective(trial):
    gens = trial.suggest_int("generations", 100, 300, step=50)
    cxpb = trial.suggest_float("crossover_rate", 0.5, 0.95)
    mutpb = trial.suggest_float("mutation_rate", 0.1, 0.9)
    t0 = time.time(); r = run_nsga2(POP, gens, cxpb, mutpb, seed=trial.number)
    for k, v in r.items(): trial.set_user_attr(k, v)
    trial.set_user_attr("secs", round(time.time() - t0, 1))
    print(f"  trial {trial.number}: gens={gens} cx={cxpb:.2f} mut={mutpb:.2f} -> val HV {r['hv_val']:.4f}, "
          f"best val F1 {r['best_val_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return r["hv_val"]

def reseed(j): return run_nsga2(POP, j[1], j[2], j[3], j[4])

def worker(job):
    k, n = job
    if n <= 0: return
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # the sampler is NOT stored in the database, so it must be given again here (else Optuna uses an unseeded default)
    study = optuna.load_study(study_name=STUDY, storage=DB,
                              sampler=optuna.samplers.TPESampler(seed=k, n_startup_trials=8))
    study.optimize(objective, n_trials=n)

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=STUDY, storage=DB, direction="maximize", load_if_exists=True)
    COMPLETE = optuna.trial.TrialState.COMPLETE
    n_done = len(study.get_trials(states=(COMPLETE,)))
    remaining = max(0, N_TRIALS - n_done)                  # a re-run tops up to N_TRIALS instead of adding N_TRIALS more
    if n_done: print(f"continuing study '{STUDY}' in {OUT}: {n_done} trials already complete", flush=True)
    print(f"running {remaining} trial(s) on {WORKERS} worker(s) (pop {POP})", flush=True)
    split = [remaining // WORKERS + (1 if k < remaining % WORKERS else 0) for k in range(WORKERS)]
    if remaining:
        with Pool(WORKERS) as p: p.map(worker, list(enumerate(split)))
    study = optuna.load_study(study_name=STUDY, storage=DB)
    tr = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    tr.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in tr.columns]
    tr = tr[tr["state"] == "COMPLETE"]                     # drop failed / interrupted trials
    tr.sort_values("value", ascending=False).to_csv(f"{OUT}/optuna_trials.csv", index=False)
    if tr.empty: sys.exit("No completed trials - check the error messages above.")
    top = tr.sort_values("value", ascending=False).head(3)
    print("\nTop 3 trials (validation HV):"); print(top[["number", "generations", "crossover_rate", "mutation_rate", "value", "best_val_f1", "front"]].round(4).to_string(index=False))
    # re-check the best settings over several seeds (search used 1 seed per trial)
    jobs = [(int(t.number), int(t.generations), float(t.crossover_rate), float(t.mutation_rate), s) for _, t in top.iterrows() for s in (101, 102, 103)]
    with Pool(WORKERS) as p: res = p.map(reseed, jobs)
    rows = [{"trial": j[0], "generations": j[1], "crossover_rate": j[2], "mutation_rate": j[3], "seed": j[4], **r} for j, r in zip(jobs, res)]
    rc = pd.DataFrame(rows); rc.to_csv(f"{OUT}/optuna_top3_reseeded.csv", index=False)
    summ = rc.groupby(["trial", "generations", "crossover_rate", "mutation_rate"])[["hv_val", "hv_test", "best_val_f1"]].agg(["mean", "std"]).round(4)
    print("\nTop 3 re-run with 3 new seeds:"); print(summ.to_string())
    best = summ["hv_val"]["mean"].idxmax()
    json.dump({"pop": POP, "generations": int(best[1]), "crossover_rate": float(best[2]), "mutation_rate": float(best[3])}, open(f"{OUT}/best_params.json", "w"), indent=1)
    print("\nBest (after reseeding):", best)
