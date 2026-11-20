"""Algorithm-agnostic hyperparameter search (scikit-learn style), shared by NSGA-II / SPEA2 / MOEA/D.

    runs, summary = param_search(run_fn, param_space, n_iter=12, seeds=(0, 1, 2))

run_fn(params: dict, seed: int) -> dict of metrics, must include the `scoring` key (default "hv_val",
validation hypervolume) and may include e.g. hv_test, best_val_f1, front. Other algorithms only need to
write a small adapter with that signature (see moead_search.py).

  * n_iter=None  -> exhaustive grid (sklearn ParameterGrid)
  * n_iter=int   -> random search (sklearn ParameterSampler)
Every configuration is run with several seeds; configurations are ranked by the MEAN score.
Tune on validation data only: test metrics are reported but never used for ranking.
"""
import time, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.model_selection import ParameterGrid, ParameterSampler

def _call(args):
    run_fn, cid, params, seed = args
    t0 = time.time(); out = run_fn(params, seed)
    return {"config_id": cid, **params, "seed": seed, **out, "secs": round(time.time() - t0, 1)}

def param_search(run_fn, param_space, n_iter=None, seeds=(0, 1, 2), n_jobs=2, scoring="hv_val",
                 random_state=0, out_prefix=None):
    configs = list(ParameterGrid(param_space)) if n_iter is None else list(ParameterSampler(param_space, n_iter, random_state=random_state))
    jobs = [(run_fn, i, c, s) for i, c in enumerate(configs) for s in seeds]
    keys = sorted(param_space.keys())
    pkey = lambda d: tuple((k, d[k]) for k in keys)          # identifies a configuration by its VALUES
    done_rows = []
    import os
    part = f"{out_prefix}_runs_partial.csv" if out_prefix else None
    if part and os.path.exists(part):                      # resume after a crash/restart
        old = pd.read_csv(part).to_dict("records")
        cid_of = {pkey(c): i for i, c in enumerate(configs)}
        # reuse a saved run only if its parameter values AND seed are part of THIS search
        # (matching on config_id alone silently reused runs from a search with different settings)
        done_rows = [r for r in old if all(k in r for k in keys) and pkey(r) in cid_of and r["seed"] in set(seeds)]
        for r in done_rows: r["config_id"] = cid_of[pkey(r)]
        done = {(pkey(r), r["seed"]) for r in done_rows}
        jobs = [j for j in jobs if (pkey(j[2]), j[3]) not in done]
        if len(old) > len(done_rows):
            os.replace(part, f"{out_prefix}_runs_partial_OLD_{int(time.time())}.csv")
            print(f"ignored {len(old) - len(done_rows)} saved runs from a search with different settings "
                  f"(moved to {out_prefix}_runs_partial_OLD_*.csv)", flush=True)
        print(f"resuming: {len(done)} matching runs already done", flush=True)
    print(f"{len(configs)} configurations x {len(seeds)} seeds = {len(jobs)} runs on {n_jobs} workers", flush=True)
    with Pool(n_jobs) as p:
        rows = list(done_rows)
        for r in p.imap_unordered(_call, jobs):
            rows.append(r)
            if out_prefix: pd.DataFrame(rows).to_csv(f"{out_prefix}_runs_partial.csv", index=False)   # survives a crash/restart
            print(f"  done cfg {r['config_id']} seed {r['seed']}: {scoring}={r[scoring]:.4f} ({r['secs']}s)", flush=True)
    runs = pd.DataFrame(rows).sort_values(["config_id", "seed"])
    keys = list(param_space.keys()); metrics = [c for c in runs.columns if c not in keys + ["config_id", "seed", "secs"]]
    summary = runs.groupby(["config_id"] + keys)[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index().sort_values(f"{scoring}_mean", ascending=False).reset_index(drop=True)
    if out_prefix:
        runs.to_csv(f"{out_prefix}_runs.csv", index=False); summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    return runs, summary
