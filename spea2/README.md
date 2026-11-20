# SPEA2 results: TransUnion adaptive fraud-rule discovery (IDS 560, Group 1)

Everything SPEA2 has run on the 500,000-transaction synthetic dataset: code, results, graphs. Commands are in `RUN_COMMANDS.txt`.
The dataset file (`synthetic_fraud_500k.csv`) is not included.

## What SPEA2 does here
It searches for fraud **rule sets**. A *rule* is a set of conditions joined by AND; a *rule set* is several rules joined by OR.
It balances three goals at once: high **precision**, high **recall** and **few conditions** (hard cap: 160 conditions in the whole rule set).
The result is a menu of options (a Pareto front), not one answer. SPEA2 selects survivors by strength and nearest-neighbour density (DEAP `selSPEA2`).

## Setup used in every experiment (identical to the NSGA-II and MOEA/D packages)
- 500,000 transactions, about 2,500 frauds (0.5%); chronological (time-ordered) 70/15/15 train / validation / test split, no resampling.
- Rules are learned on **train** data only; **test** scores are held-out.
- Max 5 conditions per rule, max 30 rules; a rule set needs at least 5 true frauds on train to count.
- Numeric conditions use 19 quantile thresholds (`<=` / `>`); categorical conditions use `=`. Time is used only for the split, never as a rule feature.
- Hypervolume: 3 objectives `[1-precision, 1-recall, conditions/160]`, reference point `[1,1,1]`.
- Single seed (42) for Exp 1-6. "Best rule set" below = highest **validation** F1 in that experiment's front, scored on test.
- The only change from `nsga2_allfeatures.py` is the selection step (`selSPEA2` instead of `selNSGA2` / `selTournamentDCD`; parents by tournament of size 2).

## Folder map
- `code/` : `spea2_midterm.py` (Exp 1, 6 features), `spea2_allfeatures.py` (Exp 2-6; crossover, mutation, population and condition cap are arguments),
  `spea2_optuna.py` + `spea2_optuna_baseline.py` (Optuna), `param_search.py` + `spea2_search.py` (scikit-learn-style search),
  `rf_benchmark.py` (Random Forest, same script for all algorithms), `make_plots.py` (redraws all graphs)
- `results/exp1_6features` ... `results/exp6_cap50`: `spea2_3obj_pareto_front.csv` (every rule set with train/val/test precision, recall, F1, total_conditions, n_rules and rule text),
  `spea2_3obj_history.csv` (hypervolume per generation), `spea2_pareto_plot.png`
- `results/optuna/`, `results/parameter_search/`, `results/rf_benchmark/`: tuning and benchmark outputs
- `results/compare_*.png`: side-by-side graphs. The Pareto front CSVs load directly into the Streamlit dashboard.

## Experiments, why we ran them, and what we found
| Exp | Settings | Why | Best rule set (test) | Front |
|---|---|---|---|---|
| 1 | 6 features, pop 100, 150 gens, cx 0.7, mut 0.5 | Sponsor-specified features; controlled first comparison | 109 conditions, 29 rules; P 0.54 / R 0.59 / F1 0.56 | 100 sets, HV 0.523 |
| 2 | All 40 features, same settings | Sponsors: do not limit features manually | 45 conditions, 12 rules; P 0.65 / R 0.74 / F1 0.69 | 100 sets, HV 0.639 |
| 3 | All features, pop 60, 100 gens, cx 0.5, mut 0.2 | Small-search setting for the team's comparison | 19 conditions, 5 rules; P 0.59 / R 0.61 / F1 0.60 | 60 sets, HV 0.528 |
| 4 | All features, pop 200, 250 gens, cx 0.9, mut 0.8 | Large-search setting | 89 conditions, 21 rules; P 0.91 / R 0.70 / F1 0.79 | 200 sets, HV 0.761 |
| 5 | Exp 4 with mutation 0.2 | Effect of a lower mutation rate | 116 conditions, 24 rules; P 0.89 / R 0.72 / F1 0.80 | 200 sets, HV 0.735 |
| 6 | Exp 4 with a 50-condition cap | Are short, readable rule sets enough? | 45 conditions, 12 rules; P 0.88 / R 0.77 / F1 0.82 | 200 sets, HV 0.808 |
| Param search | 12 settings x 3 seeds (scikit-learn ParameterSampler), pop 200 | Reusable search function for all three algorithms | Best: 150 gens, cx 0.7, mut 0.8 (validation HV 0.769) | Worst: 150 gens, cx 0.5, mut 0.2 (0.707) |
| Optuna | 24 trials, pop 200 fixed; tunes generations, crossover, mutation | Systematic search instead of guessing | Best single trial: 200 gens, cx 0.71, mut 0.89, validation HV 0.783. After re-testing the top 3 on 3 new seeds: best is 150 gens, cx 0.57, mut 0.63 (validation HV 0.781) vs 0.756 for the Exp 4 setting on the same seeds. | |
| Random Forest | 300 trees, same split | Sponsors asked for a benchmark | P 0.98 / R 0.99 / F1 0.985 | Accuracy ceiling; a black box |

## Main findings
1. All features beat 6 features (Exp 1 to 2).
2. A bigger search helps (Exp 3 to 4). Unlike NSGA-II and MOEA/D, SPEA2's F1 does not drop with low mutation (Exp 5), and its capped Exp 6 run is its best.
3. SPEA2's tuning prefers fewer generations (150-250) with high mutation (0.6-0.9).
4. A 50-condition cap costs SPEA2 no accuracy in this run (single seed).

## Example rule sets (see the `rule_set` column for all)
- Exp 4 (29 conditions, 7 rules; P 0.96 / R 0.53): `Transactions_Last_7d > 15 AND Time_Since_Last_Transaction <= 4.21 AND Account_Age_Days <= 194 AND Num_Unique_Merchants_7d > 7` OR `Velocity_Transactions_24h > 6 AND Transactions_Last_7d > 15 AND Days_Since_Last_Address_Change <= 400 AND Email_Domain_Age_Days <= 1540` OR ...
- Exp 6 (45 conditions, 12 rules; P 0.88 / R 0.77): `Transactions_Last_7d > 14 AND Account_Age_Days <= 194 AND Days_Since_Last_Address_Change <= 400 AND Transaction_Hour <= 6` OR `Account_Age_Days <= 194 AND Velocity_Transactions_24h > 6 AND Transaction_Hour <= 6 AND Credit_Limit_Utilization > 0.525` OR ...

## Fixes (Nov 20) - code only, saved results unchanged
The saved results in `results/` were produced by the original code and are still valid. These fixes change future runs:
- **Parent selection (all SPEA2 scripts):** `tools.selTournament` compares the three objectives in order (precision first,
  then recall), so parents were picked mostly by precision. This likely pushed SPEA2 toward its high precision (91% in Exp 4).
  Replaced with `sel_parents`, a binary tournament by Pareto dominance (if one parent dominates the other it wins, otherwise a random pick),
  which is the standard SPEA2 mating selection. Re-running will give somewhat different numbers.
- **`spea2_optuna.py`:** the seeded TPE sampler was not passed to the worker processes, so they used Optuna's unseeded default;
  re-running into the same folder added another full set of trials; it printed nothing until the end. Now: seeded workers,
  one study per population size (`spea2_popN`), re-runs only top up to the requested number of trials, one line printed per trial,
  and only completed trials are kept.
- **`param_search.py`:** re-running with different settings but the same `--out` silently reused old runs (matched on config number only).
  It now reuses a saved run only when its parameter values and seed match; other saved runs are moved to `*_OLD_*.csv`.
- **All scripts** now also work when run from a Colab/Jupyter cell (they used to fail on the extra `-f kernel.json` argument).

## Caveats
- Single seed for Exp 1-6, so small differences are within noise; the tuning runs used 3 seeds (search) and 1 seed per trial (Optuna).
- SPEA2 keeps its whole archive nondominated, so its fronts have up to 200 sets, versus 160-196 for the others. Hypervolume grows with front size, so compare it with care.
- This package runs SPEA2 through the same code as NSGA-II. The team's separate SPEA2 notebook (Ananya) evolves rules on the validation split, allows 10 conditions per rule and 16 rules, and uses a different hypervolume reference point; its numbers are not directly comparable.
- Synthetic data: results show how the methods behave, not production performance.
