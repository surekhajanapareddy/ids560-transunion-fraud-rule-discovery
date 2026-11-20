"""MOEA/D fraud rule discovery, aligned to the supplied NSGA-II setup.

Training: chronological 70/15/15; train-only threshold bank and objectives.
Search: validate on the final population's nondominated front (not the historical archive).
This keeps MOEA/D's original Tchebycheff/neighborhood/replacement algorithm intact.
"""
from __future__ import annotations
import time
import copy
import random
import hashlib
import json
import os
import numpy as np
import pandas as pd

LABEL, TIME = "Fraud_Label", "Time"
MAX_PATH_LEN, MAX_PATHS, MIN_TP = 5, 30, 5
HV_NORM = 160.0  # comparable to NSGA-II even when condition_cap is 50
Q = np.linspace(0.05, 0.95, 19)
SIX_FEATURES = ["Failed_Login_Count", "Session_Duration", "Transactions_Last_24h",
                "Transactions_Last_7d", "Credit_Limit_Utilization", "Card_Usage_Frequency"]
_DATA_MANIFEST = None
_DATA_PATH = None
_CONDITIONS = None
_MASKS = None
_Y = None


def prepare_data(csv_path, feature_mode="all", strict_feature_count=True):
    """Reproduce the actual NSGA-II preprocessing, down to feature/condition order.

    all: matches nsga2_allfeatures.py; six: matches nsga2_midterm.py.
    This does NOT certify that a teammate ran the same bytes: compare csv_sha256
    manifests across their environments to verify that independently.
    """
    global _DATA_PATH, _CONDITIONS, _MASKS, _Y, _DATA_MANIFEST
    if feature_mode not in ("all", "six"):
        raise ValueError('feature_mode must be "all" or "six"')
    if feature_mode == "six":
        df = pd.read_csv(csv_path, usecols=SIX_FEATURES + [LABEL, TIME]).sort_values(TIME).reset_index(drop=True)
        cat, num = [], SIX_FEATURES[:]
    else:
        df = pd.read_csv(csv_path).sort_values(TIME).reset_index(drop=True)
        # The teammate's NSGA-II source puts *all* object/string columns in CAT,
        # including TIME/LABEL if either is a string. Refuse this pathological
        # case rather than silently using target/time as a rule feature.
        unsafe = [c for c in (LABEL, TIME) if df[c].dtype == object or str(df[c].dtype).startswith("str")]
        if unsafe:
            raise ValueError(f"NSGA-II source would leak target/time into CAT: {unsafe}. "
                             "Fix NSGA-II preprocessing too before claiming strict parity.")
        cat = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
        num = [c for c in df.columns if c not in cat + [LABEL, TIME]]
    features = num + cat
    if strict_feature_count and len(features) != (40 if feature_mode == "all" else 6):
        raise ValueError(f"Expected {40 if feature_mode == 'all' else 6} features in {feature_mode} mode, "
                         f"found {len(features)}: {features}")
    if not pd.api.types.is_numeric_dtype(df[LABEL]):
        raise ValueError("Fraud_Label must be numeric for strict parity")
    n = len(df); i1, i2 = int(0.70 * n), int(0.85 * n)
    parts = {"train": df.iloc[:i1], "val": df.iloc[i1:i2], "test": df.iloc[i2:]}
    y = {k: v[LABEL].values.astype(bool) for k, v in parts.items()}
    conditions = []
    for feat in num:
        if parts["train"][feat].isna().any():
            raise ValueError(f"Numeric feature {feat} has missing train values: teammate's np.quantile is not NaN-safe")
        for t in np.unique(np.quantile(parts["train"][feat].values, Q)):
            conditions.append((feat, "<=", float(t))); conditions.append((feat, ">", float(t)))
    for feat in cat:
        # NO dropna: exact teammate's categorical bank construction.
        try:
            unique_values = sorted(parts["train"][feat].unique())
        except TypeError as exc:
            raise ValueError(f"Categorical {feat} has un-sortable mixed categories") from exc
        for val in unique_values:
            conditions.append((feat, "=", val))
    if not conditions:
        raise ValueError("No candidate rule conditions found")

    def mask(part, c):
        feat, op, val = c; col = part[feat].values
        if op == "<=": return col <= val
        if op == ">": return col > val
        return col == val

    masks = {k: np.stack([mask(v, c) for c in conditions]) for k, v in parts.items()}
    with open(csv_path, "rb") as fh:
        sha = hashlib.file_digest(fh, "sha256").hexdigest()
    digest = hashlib.sha256("\n".join(repr((feat, op, val)) for feat, op, val in conditions).encode()).hexdigest()
    sorted_digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
    _DATA_PATH, _CONDITIONS, _MASKS, _Y = csv_path, conditions, masks, y
    _DATA_MANIFEST = {
        "csv_basename": os.path.basename(csv_path), "csv_sha256": sha,
        "n_transactions": n, "feature_mode": feature_mode,
        "sorted_rows_sha256": sorted_digest, "n_tied_time_values": int(df[TIME].duplicated().sum()),
        "num_features": num, "cat_features": cat, "n_features": len(features),
        "split_sizes": {k: len(v) for k, v in parts.items()},
        "split_fraud_counts": {k: int(y[k].sum()) for k in parts},
        "n_candidate_conditions": len(conditions), "condition_bank_sha256": digest,
        "split": "Time ascending, reset_index, floor(70%), floor(85%)",
        "quantiles": "np.linspace(0.05, 0.95, 19) from train only",
        "label": LABEL, "time_column": TIME,
    }
    print(f"Prepared {n} transactions; {len(num)} numeric + {len(cat)} categorical = {len(features)} features; "
          f"{len(conditions)} candidate conditions", flush=True)
    print({k: (len(v), int(y[k].sum())) for k,v in parts.items()}, flush=True)
    print(f"CSV SHA-256: {sha}; conditions SHA-256: {digest}", flush=True)
    return _DATA_MANIFEST.copy()


def get_data_manifest():
    if _DATA_MANIFEST is None:
        raise RuntimeError("Call prepare_data() first")
    return _DATA_MANIFEST.copy()


def metrics(rs, split):
    mask_bank = _MASKS[split]
    pred = np.zeros(mask_bank.shape[1], dtype=bool)
    for path in rs:
        if not path: continue
        p = mask_bank[path[0]].copy()
        for c in path[1:]: p &= mask_bank[c]
        pred |= p
    yy = _Y[split]
    tp = int((pred & yy).sum()); fp = int((pred & ~yy).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / int(yy.sum()) if yy.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, precision, recall, f1


def total_conditions(rs):
    return sum(map(len, rs))


def minimization_obj(obj):
    return np.array([1. - obj[0], 1. - obj[1], obj[2] / HV_NORM], float)


def nondominated_indices(vals):
    vals = np.asarray(vals, float)
    return [i for i,v in enumerate(vals) if not np.any(
        np.all(vals <= v, axis=1) & np.any(vals < v, axis=1)
    )]


def hypervolume_from_points(points):
    """Exact HV for minimization of 3 dimensions, reference point [1,1,1].

    Same 3D sweep implementation as in the user's original MOEA/D notebook.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.all((pts >= 0) & (pts < 1), axis=1)]
    if not len(pts): return 0.0
    xs = np.unique(np.r_[pts[:, 0], 1.0])
    volume = 0.0
    for left,right in zip(xs[:-1], xs[1:]):
        active = pts[pts[:, 0] <= left, 1:]
        order = np.argsort(active[:, 0], kind="stable")
        ys, zs = active[order, 0], active[order, 1]
        uniq, ix = np.unique(ys, return_index=True)
        minz = np.minimum.reduceat(zs, ix)
        running_min = np.minimum.accumulate(minz)
        volume += (right - left) * np.sum(np.diff(np.r_[uniq, 1.0]) * (1.0 - running_min))
    return float(volume)


def front_hv(front, split):
    if not front: return 0.0
    points = []
    for rs, _ in front:
        _, p, r, _ = metrics(rs, split)
        points.append([1.0-p, 1.0-r, min(total_conditions(rs)/HV_NORM, 1.0)])
    return hypervolume_from_points(points)


def make_weights(n, seed):
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet([1., 1., 1.], size=n)
    if n >= 3: weights[:3] = [[.98,.01,.01],[.01,.98,.01],[.01,.01,.98]]
    return weights


def pretty(rs):
    return "  OR  ".join("(" + " AND ".join(
        f"{_CONDITIONS[c][0]} {_CONDITIONS[c][1]} " +
        (f"{_CONDITIONS[c][2]:.4g}" if _CONDITIONS[c][1] != "=" else repr(_CONDITIONS[c][2]))
        for c in path) + ")" for path in rs)


def front_frame(front, condition_cap=160):
    # NSGA-II main script deduplicates on the *train fitness vector*, not on
    # reported train metrics. These differ for paths rejected for train TP<5.
    rows = []
    seen = set()
    for rs, obj in front:
        key = tuple(round(float(v), 6) for v in obj)
        if key in seen: continue
        seen.add(key)
        _, vp, vr, vf = metrics(rs, "val")
        _, tp, tr, tf = metrics(rs, "test")
        _, trp, trr, _ = metrics(rs, "train")
        feats = {_CONDITIONS[c][0] for path in rs for c in path}
        rows.append({"train_precision":trp,"train_recall":trr,
                     "val_precision":vp,"val_recall":vr,"val_f1":vf,
                     "test_precision":tp,"test_recall":tr,"test_f1":tf,
                     "total_conditions":total_conditions(rs),"n_rules":len(rs),
                     "distinct_features":len(feats),"within_160_cap":total_conditions(rs)<=160,
                     "within_run_cap":total_conditions(rs)<=condition_cap,
                     "rule_set":pretty(rs)})
    cols=["train_precision","train_recall","val_precision","val_recall","val_f1", "test_precision", "test_recall","test_f1", "total_conditions","n_rules","distinct_features","within_160_cap","within_run_cap","rule_set"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty: return df
    return df.sort_values("total_conditions").reset_index(drop=True)


def run_moead(pop_n=200, gens=250, cxpb=.9, mutpb=.8, seed=42,
              condition_cap=160, track_archive=False, track_history=False,
              neighborhood_size=20, max_replacements=2, mating_prob_neighbor=.9,
              force_variation=True):
    """Return (scalar summary, final population Pareto front, archive, generation history).

    For tuning, track_archive=False speeds up runs without changing optimization.
    force_variation=True (FIX): a child that got neither crossover nor mutation is mutated, so exact
    copies of a parent can no longer replace neighbours (40% of children in Exp 3 were unchanged copies).
    Use force_variation=False only to reproduce the original saved results.
    The external archive is *not* used for MOEA/D population selection/replacement.
    """
    if _MASKS is None: raise RuntimeError("Call prepare_data(csv_path) before run_moead")
    if pop_n < 2 or gens < 1: raise ValueError("pop_n >=2 and gens >=1 are required")
    if not 0 <= cxpb <= 1 or not 0 <= mutpb <= 1: raise ValueError("probabilities must be in [0,1]")
    random.seed(seed); np.random.seed(seed)
    NC = len(_CONDITIONS)
    n_evaluations=0
    t0 = time.time()

    def rand_path(): return [random.randrange(NC) for _ in range(random.randint(1,3))]
    def rand_ruleset(): return [rand_path() for _ in range(random.randint(1,6))]
    def repair(rs):
        rs = [list(dict.fromkeys(path))[:MAX_PATH_LEN] for path in rs if path][:MAX_PATHS]
        while total_conditions(rs)>condition_cap and rs:
            i = max(range(len(rs)),key=lambda j:len(rs[j]))
            if len(rs[i])>1: rs[i].pop(random.randrange(len(rs[i])))
            else: rs.pop(i)
        return rs or [rand_path()]
    def mutate(rs):
        r=random.random()
        if r<.25 and len(rs)<MAX_PATHS: rs.append(rand_path())
        elif r<.40 and len(rs)>1: rs.pop(random.randrange(len(rs)))
        else:
            path=random.choice(rs); r2=random.random()
            if r2<.4: path[random.randrange(len(path))]=random.randrange(NC)
            elif r2<.7 and len(path)<MAX_PATH_LEN: path.append(random.randrange(NC))
            elif len(path)>1: path.pop(random.randrange(len(path)))
            else: path[0]=random.randrange(NC)
    def crossover(a,b):
        pool=[list(path) for path in a+b]; random.shuffle(pool)
        k=random.randint(1,max(1,len(pool)-1))
        a[:]=pool[:k]; b[:]=pool[k:] or [rand_path()]
    def evaluate(rs):
        tp,p,r,_ = metrics(rs,"train")
        return (0.,0.,float(condition_cap)) if tp<MIN_TP else (p,r,float(total_conditions(rs)))

    weights = make_weights(pop_n,seed)
    dists = np.linalg.norm(weights[:,None,:]-weights[None,:,:],axis=2)
    neighbors=np.argsort(dists,axis=1)[:,:min(neighborhood_size,pop_n)]
    pop=[repair(rand_ruleset()) for _ in range(pop_n)]
    objs=[evaluate(rs) for rs in pop]
    n_evaluations+=len(pop)
    vals=np.array([minimization_obj(obj) for obj in objs])
    ideal=np.min(vals,axis=0).copy()
    archive={}; archive_vectors=np.empty((0,3),float)

    def update_archive(ind,obj):
        nonlocal archive_vectors
        key=tuple(tuple(path) for path in ind)
        f=minimization_obj(obj)
        if key in archive: return
        if len(archive_vectors):
            less_or_equal=np.all(archive_vectors<=f,axis=1)
            strictly_less=np.any(archive_vectors<f,axis=1)
            if np.any(less_or_equal & strictly_less):return
            dominates=np.all(f<=archive_vectors,axis=1)&np.any(f<archive_vectors,axis=1)
            if np.any(dominates):
                old_keys=list(archive)
                for j in np.flatnonzero(dominates): del archive[old_keys[int(j)]]
                archive_vectors=archive_vectors[~dominates]
        archive[key]=(copy.deepcopy(ind),obj)
        archive_vectors=np.vstack((archive_vectors,f))

    if track_archive:
        for ind,obj in zip(pop,objs):update_archive(ind,obj)
    history=[]
    for gen in range(gens):
        for i in np.random.permutation(pop_n):
            use_neighborhood=random.random()<mating_prob_neighbor
            pool=neighbors[i] if use_neighborhood else np.arange(pop_n)
            if len(pool)<2:pool=np.arange(pop_n)
            a,b=random.sample(list(map(int,pool)),2)
            child=copy.deepcopy(pop[a])
            crossed=random.random()<cxpb
            if crossed:
                other=copy.deepcopy(pop[b]);crossover(child,other)
            if random.random()<mutpb or (force_variation and not crossed):mutate(child)
            child=repair(child)
            obj=evaluate(child)
            n_evaluations+=1
            f=minimization_obj(obj)
            ideal=np.minimum(ideal,f)
            if track_archive:update_archive(child,obj)
            target=list(map(int,pool));random.shuffle(target)
            replaced=0
            for j in target:
                if np.max(weights[j]*np.abs(f-ideal))<=np.max(weights[j]*np.abs(vals[j]-ideal)):
                    pop[j]=copy.deepcopy(child);objs[j]=obj;vals[j]=f
                    replaced+=1
                    if replaced>=max_replacements:break
        if track_history:
            inds=nondominated_indices(vals)
            pop_front=[(pop[i],objs[i]) for i in inds]
            train_pop_hv=hypervolume_from_points([vals[i] for i in inds])
            # Match NSGA-II's *history* semantics: current population front,
            # 1 row per generation, HV normalized by 160, ref [1,1,1].
            arch=list(archive.values()) if track_archive else []
            train_arch_hv=(hypervolume_from_points([minimization_obj(obj) for _,obj in arch])
                           if arch and (gen%5==0 or gen==gens-1) else float("nan"))
            history.append({"gen":gen,"front_size":len(pop_front),"hv3d":train_pop_hv,
                            "best_precision":max(obj[0] for _,obj in pop_front),
                            "best_recall":max(obj[1] for _,obj in pop_front),
                            "min_cond":min(obj[2] for _,obj in pop_front),
                            "max_cond":max(obj[2] for _,obj in pop_front),
                            "elapsed_s":round(time.time()-t0,1),
                            "archive_size":len(arch) if track_archive else None,
                            "archive_hv3d":train_arch_hv})
            if gen%5==0 or gen==gens-1:
                print(f"gen {gen:3d} population_front={len(pop_front):3d} HVpop={train_pop_hv:.4f}" +
                      (f" archive={len(arch)} HVarchive={train_arch_hv:.4f}" if arch else ""),flush=True)

    pop_front=[(pop[i],objs[i]) for i in nondominated_indices(vals)]
    # The same objects, evaluated on validation, like nsga2_optuna.py.
    valscores=[metrics(rs,"val")[3] for rs,_ in pop_front]
    best_idx=int(np.argmax(valscores))
    best_rs=pop_front[best_idx][0]
    summary={"hv_val":front_hv(pop_front,"val"), "hv_test":front_hv(pop_front,"test"),
             "best_val_f1":valscores[best_idx],
             "selected_test_f1":metrics(best_rs,"test")[3],
             "front":len(pop_front),
             "n_evaluations":n_evaluations,
             "force_variation":force_variation,
             "hv_train_pop":hypervolume_from_points([minimization_obj(obj) for _,obj in pop_front])}
    if track_archive:
        summary["hv_train_archive"]=hypervolume_from_points([minimization_obj(obj) for _,obj in archive.values()])
        summary["archive_size"]=len(archive)
    return summary,pop_front,list(archive.values()),pd.DataFrame(history)
