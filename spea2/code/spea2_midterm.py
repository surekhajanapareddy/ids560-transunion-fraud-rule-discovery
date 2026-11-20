"""SPEA2 (precision, recall, complexity) on synthetic_fraud_500k with the team's 6 shared features.
Usage: python spea2_midterm.py [csv_path] [pop] [gens] [seed]
"""
import sys, time, random, copy, json
import numpy as np, pandas as pd
from deap import base, creator, tools
from pymoo.indicators.hv import HV

# ignore the "-f kernel.json" arguments Jupyter/Colab passes when this code is run in a notebook cell
_A = [a for a in sys.argv[1:] if a != "-f" and not a.endswith(".json")]
CSV = _A[0] if len(_A) > 0 else "synthetic_fraud_500k.csv"
POP = int(_A[1]) if len(_A) > 1 else 100
GENS = int(_A[2]) if len(_A) > 2 else 40
SEED = int(_A[3]) if len(_A) > 3 else 42
OUT = "."

FEATURES = ["Failed_Login_Count", "Session_Duration", "Transactions_Last_24h",
            "Transactions_Last_7d", "Credit_Limit_Utilization", "Card_Usage_Frequency"]
LABEL, TIME = "Fraud_Label", "Time"
MAX_TOTAL_CONDITIONS = 160
MAX_PATH_LEN = 5
MAX_PATHS = 30
MIN_TP = 5
Q = np.linspace(0.05, 0.95, 19)

random.seed(SEED); np.random.seed(SEED)

# ---- data: time-ordered 70/15/15 split ---------------------------------------
df = pd.read_csv(CSV, usecols=FEATURES + [LABEL, TIME]).sort_values(TIME).reset_index(drop=True)
n = len(df); i1, i2 = int(.70 * n), int(.85 * n)
parts = {"train": df.iloc[:i1], "val": df.iloc[i1:i2], "test": df.iloc[i2:]}
y = {k: v[LABEL].values.astype(bool) for k, v in parts.items()}
print({k: (len(v), int(y[k].sum())) for k, v in parts.items()})

# ---- condition bank: every (feature, op, quantile threshold) as a precomputed mask
conds = []  # (feature, op, threshold)
for f in FEATURES:
    for t in np.unique(np.quantile(parts["train"][f].values, Q)):
        conds.append((f, "<=", float(t))); conds.append((f, ">", float(t)))
masks = {k: np.stack([(v[f].values <= t) if op == "<=" else (v[f].values > t)
                      for f, op, t in conds]) for k, v in parts.items()}
NC = len(conds)
print("condition bank:", NC)

def predict(rs, split):
    M = masks[split]; pred = np.zeros(M.shape[1], bool)
    for path in rs:
        p = M[path[0]].copy()
        for c in path[1:]: p &= M[c]
        pred |= p
    return pred

def metrics(rs, split):
    pred = predict(rs, split); yy = y[split]
    tp = int((pred & yy).sum()); fp = int((pred & ~yy).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / yy.sum()
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, prec, rec, f1

def total_conditions(rs): return sum(len(p) for p in rs)

def rand_path(): return [random.randrange(NC) for _ in range(random.randint(1, 3))]
def rand_ruleset(): return [rand_path() for _ in range(random.randint(1, 6))]

def repair(rs):
    rs = [list(dict.fromkeys(p))[:MAX_PATH_LEN] for p in rs if p][:MAX_PATHS]
    while total_conditions(rs) > MAX_TOTAL_CONDITIONS and rs:
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
    k = random.randint(1, max(1, len(pool) - 1))
    a[:] = pool[:k]; b[:] = pool[k:] or [rand_path()]

creator.create("Fit3", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("RS", list, fitness=creator.Fit3)

def evaluate(rs):
    tp, prec, rec, _ = metrics(rs, "train")
    if tp < MIN_TP: return (0.0, 0.0, float(MAX_TOTAL_CONDITIONS))
    return (prec, rec, float(total_conditions(rs)))

tb = base.Toolbox()
tb.register("individual", tools.initIterate, creator.RS, rand_ruleset)
tb.register("population", tools.initRepeat, list, tb.individual)

hv_ind = HV(ref_point=np.array([1.0, 1.0, 1.0]))
def hv3d(front):
    pts = [[1 - i.fitness.values[0], 1 - i.fitness.values[1],
            min(i.fitness.values[2] / MAX_TOTAL_CONDITIONS, 1.0)] for i in front]
    return float(hv_ind(np.array(pts))) if pts else 0.0

def sel_parents(pop, k):
    """Binary tournament by Pareto dominance (standard SPEA2 mating selection).
    FIX: DEAP's selTournament compares the 3 objectives in order (precision first, then recall),
    which biased SPEA2 toward high precision."""
    out = []
    for _ in range(k):
        a, b = random.sample(pop, 2)
        if a.fitness.dominates(b.fitness): out.append(a)
        elif b.fitness.dominates(a.fitness): out.append(b)
        else: out.append(random.choice((a, b)))
    return out

def run():
    t0 = time.time()
    pop = tb.population(n=POP)
    for ind in pop:
        ind[:] = repair(ind); ind.fitness.values = evaluate(ind)
    pop = tools.selSPEA2(pop, POP); hist = []
    for g in range(GENS):
        off = [copy.deepcopy(i) for i in sel_parents(pop, POP)]
        for a, b in zip(off[::2], off[1::2]):
            if random.random() < 0.7: cx(a, b); del a.fitness.values, b.fitness.values
        for i in off:
            if random.random() < 0.5: mutate(i); del i.fitness.values
        for i in off:
            i[:] = repair(i)
            if not i.fitness.valid: i.fitness.values = evaluate(i)
        pop = tools.selSPEA2(pop + off, POP)
        f0 = tools.sortNondominated(pop, POP, first_front_only=True)[0]
        hv = hv3d(f0)
        hist.append({"gen": g, "front_size": len(f0), "hv3d": hv,
                     "best_precision": max(i.fitness.values[0] for i in f0),
                     "best_recall": max(i.fitness.values[1] for i in f0),
                     "min_cond": min(i.fitness.values[2] for i in f0),
                     "max_cond": max(i.fitness.values[2] for i in f0),
                     "elapsed_s": round(time.time() - t0, 1)})
        if g % 5 == 0 or g == GENS - 1:
            h = hist[-1]; print(f"gen {g:3d} front={h['front_size']:3d} HV3D={hv:.4f} "
                                f"bestP={h['best_precision']:.3f} bestR={h['best_recall']:.3f} "
                                f"cond[{h['min_cond']:.0f}-{h['max_cond']:.0f}] {h['elapsed_s']}s", flush=True)
    return tools.sortNondominated(pop, POP, first_front_only=True)[0], pd.DataFrame(hist)

def pretty(rs):
    return "  OR  ".join("(" + " AND ".join(f"{conds[c][0]} {conds[c][1]} {conds[c][2]:.4g}" for c in p) + ")" for p in rs)

front, hist = run()
rows, seen = [], set()
for ind in front:
    key = tuple(round(v, 6) for v in ind.fitness.values)
    if key in seen: continue
    seen.add(key)
    _, vp, vr, vf = metrics(ind, "val"); _, tp_, tr, tf = metrics(ind, "test")
    _, trp, trr, trf = metrics(ind, "train")
    feats = {conds[c][0] for p in ind for c in p}
    rows.append({"train_precision": trp, "train_recall": trr, "val_precision": vp, "val_recall": vr, "val_f1": vf,
                 "test_precision": tp_, "test_recall": tr, "test_f1": tf,
                 "total_conditions": total_conditions(ind), "n_rules": len(ind),
                 "distinct_features": len(feats), "within_160_cap": total_conditions(ind) <= MAX_TOTAL_CONDITIONS,
                 "rule_set": pretty(ind)})
fdf = pd.DataFrame(rows).sort_values("total_conditions")
fdf.to_csv(f"{OUT}/spea2_3obj_pareto_front.csv", index=False)
hist.to_csv(f"{OUT}/spea2_3obj_history.csv", index=False)
print(f"\nPareto front: {len(fdf)} unique solutions | all within cap: {fdf.within_160_cap.all()}")
print(f"Final 3D HV (train objectives): {hist.hv3d.iloc[-1]:.4f}")
pd.set_option("display.width", 200)
print(fdf[["train_precision", "train_recall", "val_precision", "val_recall", "test_precision", "test_recall",
           "total_conditions", "n_rules", "distinct_features"]].round(3).to_string(index=False))
