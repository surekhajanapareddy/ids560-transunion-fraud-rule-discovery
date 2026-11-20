"""Re-run a fixed SPEA2 setting with the same seeds/scoring as the Optuna re-check (for a fair comparison).
Usage: python optuna_baseline.py [csv_path] [pop] [gens] [crossover] [mutation] [out_csv]"""
import sys
A = [a for a in sys.argv[1:] if a != "-f" and not a.endswith(".json")] + [None] * 6
CSV = A[0] or "synthetic_fraud_500k.csv"; POP = int(A[1] or 200); GENS = int(A[2] or 250)
CX = float(A[3] or 0.9); MUT = float(A[4] or 0.8); OUTCSV = A[5] or "baseline_same_seeds.csv"
import tempfile; sys.argv = ["spea2_optuna.py", CSV, "1", str(POP), tempfile.mkdtemp(), "1"]
import spea2_optuna as m, pandas as pd
from multiprocessing import Pool
def f(seed): return m.run_spea2(POP, GENS, CX, MUT, seed)
if __name__ == "__main__":
    with Pool(2) as p: r = p.map(f, [101, 102, 103])
    d = pd.DataFrame(r); d["seed"] = [101, 102, 103]; d.to_csv(OUTCSV, index=False)
    print(d[["hv_val", "hv_test", "best_val_f1"]].agg(["mean", "std"]).round(4).to_string())
