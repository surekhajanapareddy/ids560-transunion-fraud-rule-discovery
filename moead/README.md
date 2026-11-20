# MOEA/D — IDS 560 TransUnion Capstone

## Final submission: code, experiment results, and tuning results

**Status:** Six main experiments, scikit-learn-style parameter search (12 configurations × 3 seeds = 36 runs), Optuna (24 completed trials), Optuna top-three retest (3 × 3 seeds = 9 runs), Exp 4 same-seed baseline (3 runs), and the Random Forest reference benchmark have all been completed. The results included here are saved outputs of actual runs, not placeholders.

**Dataset notice:** The source `synthetic_fraud_500k.csv` is intentionally excluded from this shareable ZIP to keep it small, as with the NSGA-II teammate's results bundle. To *rerun* the code, obtain the shared 500K CSV and place it next to `moead_core.py`. The SHA-256 hash of the file used for these MOEA/D results is:

`442f264eed7648bf0d227a53bb171130dda7639adcf572c8376c68cda1d1909d`

The provided NSGA-II archive does not contain its original CSV, so byte-for-byte equality of the **actual datasets** is not independently verified. For a strictly controlled comparison, confirm the NSGA-II input SHA-256 and time ordering as described in `data_parity_report.json`.

## Experiment protocols

| Item | MOEA/D setting |
|---|---|
| Data split | Chronological by `Time`; 70% train / 15% validation / 15% test |
| Candidate features | Exp 1: 6; Exp 2–6: all 40 |
| Numeric thresholds | Unique training-set quantiles from 5% to 95% in 5% steps (`<=` and `>`) |
| Categorical thresholds | Categories observed in training data (`=`) |
| Rules | AND of conditions within each rule; OR across rules |
| Rule limits | Maximum 5 conditions/rule, 30 rules/ruleset; cap 160 total conditions (Exp 6 uses cap 50) |
| Optimization objectives | Maximize training precision and recall; minimize condition count |
| Invalid rules | Train true positives fewer than 5 |
| Reported Pareto front | Nondominated solutions from **final population**, using training objectives; historical archive stored separately |
| Hypervolume | 3 objectives `[1-precision, 1-recall, min(conditions/160, 1)]`, minimization reference `[1,1,1]` |
| Single rule selection | Highest *validation* F1 among final-front solutions; evaluate once on test |
| Main experiments | Seed 42; same six configurations used in NSGA-II |

The evolutionary optimizer remains genuine **MOEA/D** (weighted Tchebycheff decomposition, neighborhood search and bounded replacement); its internal algorithm is intentionally different from NSGA-II. Matching the hyperparameter **protocol** does not require identical optimal parameter values or equal computation time.

## Completed main experiments

| Experiment | Features | Population | Generations | Crossover | Mutation | Condition cap |
|---|---:|---:|---:|---:|---:|---:|
| Exp 1 | 6 | 100 | 150 | 0.7 | 0.5 | 160 |
| Exp 2 | 40 | 100 | 150 | 0.7 | 0.5 | 160 |
| Exp 3 | 40 | 60 | 100 | 0.5 | 0.2 | 160 |
| Exp 4 | 40 | 200 | 250 | 0.9 | 0.8 | 160 |
| Exp 5 | 40 | 200 | 250 | 0.9 | 0.2 | 160 |
| Exp 6 | 40 | 200 | 250 | 0.9 | 0.8 | 50 |

Each `results/expX_moead/` directory includes `summary.json`, `data_manifest.json`, final-population Pareto front CSV, HV history CSV, and plots. The all-time Pareto archive is included as a *separate diagnostic*, not as the primary Pareto front.

## Completed automatic hyperparameter searches

**Scikit-learn-style search** (`param_search.py` + `moead_search.py`): `ParameterSampler` with random state 0; 12 sampled configurations, 3 seeds per configuration (0, 1, 2); population 200; ranking by **mean validation hypervolume**; same parameter grid as NSGA-II.

**Optuna search** (`moead_optuna.py`): TPE seed 0, 8 startup trials, 24 completed trials, 2 workers, population 200; generations 100–300 in steps of 50; crossover 0.5–0.95; mutation 0.1–0.9. Top three trial configurations retested on seeds 101–103, selected by mean validation hypervolume; Exp 4 baseline also tested on seeds 101–103. Due to adaptive TPE and differing algorithm evaluations, individual suggestions and results can differ from NSGA-II.

The shared Random Forest benchmark uses the same RF source script as NSGA-II. Differences in benchmark outputs should be resolved by checking raw CSV identity and environment before describing it as an exactly identical baseline.

## Fixes (Nov 20) - code only, saved results unchanged
The saved results in `results/` were produced by the original code and are still valid. These fixes change future runs:
- **No unchanged parent copies (`moead_core.py`):** a child that got neither crossover nor mutation was an exact copy of its parent
  but could still replace neighbours, filling the population with duplicates (about 2% of children in Exp 4/6, 8% in Exp 5,
  15% in Exp 1/2 and 40% in Exp 3). Such a child is now always mutated (`force_variation=True`).
  To reproduce the original saved results exactly, add `--no-force-variation` to `moead_run.py` / `run_experiments.py`.
  `--skip-existing` only skips outputs made with the same setting (recorded in `summary.json`).
- **`moead_optuna.py`:** the seeded TPE sampler was not passed to the worker processes, so they used Optuna's unseeded default.
  Now seeded; one study per population size (`moead_popN`) so runs with different `--pop` never mix; one line printed per trial.
  It also works on Windows/Mac now (the database path and data were only set up in the main process).
- **`param_search.py`:** re-running with different settings but the same `--out` silently reused old runs (matched on config number only).
  It now reuses a saved run only when its parameter values and seed match; other saved runs are moved to `*_OLD_*.csv`.
  `moead_search.py` also loads the data inside worker processes (Windows/Mac).
- **All scripts** now tolerate the extra `-f kernel.json` argument when run from a Colab/Jupyter cell.
- Not changed (design choice, noted for the report): weight vectors are drawn at random (Dirichlet) plus the three corners,
  instead of an evenly spread simplex-lattice.

## Directory map

- `moead_core.py`: MOEA/D algorithm, shared feature preprocessing, rule representation, fitness metrics.
- `run_experiments.py`, `moead_run.py`: the six main experiments.
- `param_search.py`, `moead_search.py`: 36-run parameter search.
- `moead_optuna.py`, `moead_optuna_baseline.py`: Optuna trials, retesting, same-seed baseline.
- `rf_benchmark.py`: Random Forest reference benchmark.
- `plot_optuna_search.py`, `make_moead_plots.py`: visualization.
- `compare_preprocessing.py`, `verify_completion.py`, `verify_tuning_outputs.py`: consistency and completeness checks.
- `results/exp1_moead/` through `results/exp6_moead/`: six main experiments.
- `results/parameter_search/`: 36 runs and a 12-configuration summary.
- `results/optuna/`: trial table, best parameters, 9 reseeded runs, 3 baseline runs, saved SQLite study and plot.
- `results/rf_benchmark/`: RF benchmark and feature importances.
- `MOEAD_Strict_Aligned_Colab.ipynb`, `MOEAD_Tuning_Only_Colab.ipynb`: notebook runners.

## Run / reproduce

```bash
pip install -r requirements.txt
# Put the matching synthetic_fraud_500k.csv next to moead_core.py first
python compare_preprocessing.py --csv synthetic_fraud_500k.csv --out data_parity_report.json
python run_experiments.py --exp all --skip-existing
python moead_search.py --n_iter 12 --seeds 3 --pop 200 --jobs 2
python moead_optuna.py --trials 24 --pop 200 --workers 2
python moead_optuna_baseline.py --pop 200 --gens 250 --cx .9 --mut .8
python make_moead_plots.py
python plot_optuna_search.py
python verify_completion.py --csv synthetic_fraud_500k.csv
python verify_tuning_outputs.py results
```

Running the full set of experiments and tuning again is computationally expensive. The existing results are provided so reviewers need not rerun them. When using Colab in a fresh runtime, supply the matching raw dataset separately.

**Comparison caveats:** NSGA-II's original dataset bytes have not yet been verified against the MOEA/D SHA-256 above. Likewise, matching population and generations does not guarantee equal numbers of objective evaluations across different algorithms. These are study limitations, not uncompleted MOEA/D runs.
