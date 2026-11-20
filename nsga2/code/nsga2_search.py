"""Parameter search for NSGA-II (all features, 500k, chronological 70/15/15) using param_search.py.
Usage:
  python nsga2_search.py --csv synthetic_fraud_500k.csv --n_iter 12 --seeds 3 --pop 200 --out nsga2_search
  (--n_iter 0 = full grid; --gens / --cx / --mut take comma-separated value lists)
Population is fixed (default 200): hypervolume grows with archive size, so tuning it would just reward big fronts."""
import argparse, sys
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="synthetic_fraud_500k.csv"); ap.add_argument("--n_iter", type=int, default=12)
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--pop", type=int, default=200)
ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--out", default="nsga2_search")
ap.add_argument("--gens", default="100,150,200,250,300"); ap.add_argument("--cx", default="0.5,0.6,0.7,0.8,0.9")
ap.add_argument("--mut", default="0.1,0.2,0.4,0.5,0.6,0.8")
A, _ = ap.parse_known_args()   # tolerate the extra "-f kernel.json" args when run inside Jupyter/Colab
import tempfile
sys.argv = ["nsga2_optuna.py", A.csv, "1", str(A.pop), tempfile.mkdtemp(), "1"]   # lets the NSGA-II module load the data
import nsga2_optuna as m
from param_search import param_search

def nsga2_run(params, seed):      # adapter: (params, seed) -> metrics dict
    return m.run_nsga2(A.pop, params["generations"], params["crossover_rate"], params["mutation_rate"], seed)

if __name__ == "__main__":
    space = {"generations": [int(x) for x in A.gens.split(",")],
             "crossover_rate": [float(x) for x in A.cx.split(",")],
             "mutation_rate": [float(x) for x in A.mut.split(",")]}
    runs, summary = param_search(nsga2_run, space, n_iter=A.n_iter or None, seeds=range(A.seeds), n_jobs=A.jobs, out_prefix=A.out)
    import pandas as pd; pd.set_option("display.width", 200)
    print("\nRanked by mean validation hypervolume:")
    print(summary[["config_id", "generations", "crossover_rate", "mutation_rate", "hv_val_mean", "hv_val_std", "hv_test_mean", "best_val_f1_mean"]].round(4).to_string(index=False))
