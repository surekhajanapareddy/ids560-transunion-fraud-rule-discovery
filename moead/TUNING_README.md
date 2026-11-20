# Automated tuning: final completed deliverables

Both automated searches are completed and their actual result files are contained in this ZIP.

## scikit-learn-style parameter search

- Shared `param_search.py` with NSGA-II; `ParameterSampler` random state 0.
- Population 200; 12 parameter configurations, seeds 0, 1, 2 (36 runs).
- Rank configurations by mean validation hypervolume.
- Outputs: `results/parameter_search/moead_search_runs.csv` (36), `moead_search_summary.csv` (12), and partial/restart file.

## Optuna search

- TPE seed 0; 8 startup trials; 24 completed trials; 2 workers; population 200.
- Generations: 100, 150, 200, 250, 300; crossover 0.5–0.95; mutation 0.1–0.9.
- Top three trials rerun with seeds 101, 102, 103 (9 runs total), ranked by mean validation HV.
- Exp 4 parameters retested on the same three seeds (3 runs).
- Outputs under `results/optuna/`: `optuna_trials.csv`, `optuna_top3_reseeded.csv`, `exp4_setting_same_seeds.csv`, `best_params.json`, `optuna_search.png`, `moead_optuna.db`.

The Random Forest benchmark is a shared model-independent baseline. Main experiment results Exp 1–6 are also included in `results/`.

## Reproduction and validation

This ZIP excludes `synthetic_fraud_500k.csv`. Use the matching team CSV (SHA-256 in `README.md`) to rerun. From the project folder:

```bash
python verify_tuning_outputs.py results
python verify_completion.py --csv synthetic_fraud_500k.csv
```

In a fresh Colab session the notebooks require the matching CSV separately. The included results can be reviewed without rerunning any code.

**Input parity caveat:** NSGA-II's original CSV is not in its results bundle, so byte-for-byte input parity has not been established by the ZIP comparison alone.
