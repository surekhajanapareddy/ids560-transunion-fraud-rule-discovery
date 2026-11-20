"""Random Forest benchmark + feature importance, same data/split as NSGA-II (500k, all features, chronological 70/15/15).
Usage: python rf_benchmark.py [csv_path]"""
import sys, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
CSV = sys.argv[1] if len(sys.argv) > 1 else "synthetic_fraud_500k.csv"
LABEL, TIME = "Fraud_Label", "Time"
df = pd.read_csv(CSV).sort_values(TIME).reset_index(drop=True)
CAT = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
NUM = [c for c in df.columns if c not in CAT + [LABEL, TIME]]
n = len(df); i1, i2 = int(.70*n), int(.85*n)
tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(tr[CAT])
def X(d):
    x = d[NUM].copy(); x[CAT] = enc.transform(d[CAT]); return x
Xtr, Xva, Xte = X(tr), X(va), X(te); ytr, yva, yte = tr[LABEL].values, va[LABEL].values, te[LABEL].values
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42).fit(Xtr, ytr)
print(f"trained in {time.time()-t0:.0f}s")
pv, pt = rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xte)[:, 1]
# threshold chosen on VALIDATION (best F1), then applied to test
p, r, th = precision_recall_curve(yva, pv); f = 2*p*r/(p+r+1e-12); thr = th[np.argmax(f[:-1])]
rows = []
for name, t in [("default 0.5", 0.5), (f"validation-best threshold {thr:.2f}", thr)]:
    for split, pr, yy in [("val", pv, yva), ("test", pt, yte)]:
        pred = pr >= t
        rows.append({"threshold": name, "split": split, "precision": precision_score(yy, pred), "recall": recall_score(yy, pred), "f1": f1_score(yy, pred)})
res = pd.DataFrame(rows).round(3); print(res.to_string(index=False)); res.to_csv("rf_benchmark_results.csv", index=False)
imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
imp.round(4).to_csv("rf_feature_importance.csv", header=["importance"]); print("\nTop 12 features:"); print(imp.head(12).round(4).to_string())
