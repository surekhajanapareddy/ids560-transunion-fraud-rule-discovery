# NSGA-II results: TransUnion adaptive fraud-rule discovery (IDS 560, Group 1)

Everything NSGA-II (Surekha) has run on the 500,000-transaction synthetic dataset: code, results, graphs.
Commands are in `RUN_COMMANDS.txt`. The dataset file (`synthetic_fraud_500k.csv`) is not included.

## What NSGA-II does here
It searches for fraud **rule sets**. A *rule* is a set of conditions joined by AND (e.g. `Account_Age_Days <= 194 AND Velocity_Transactions_24h > 4`).
A *rule set* is several rules joined by OR. It balances three goals at once: high **precision**, high **recall**, and **few conditions**
(hard cap: 160 conditions in the whole rule set). The result is a menu of options (a Pareto front), not one answer.

## Setup used in every experiment
- 500,000 transactions, about 2,500 frauds (0.5%); chronological (time-ordered) 70/15/15 train / validation / test split, no resampling.
- Rules are learned on **train** data only; **test** scores are held-out.
- Max 5 conditions per rule, max 30 rules; a rule set needs at least 5 true frauds on train to count.
- Single seed (42) for Exp 1-6. Time is used only for the split, never as a rule feature.
- "Best rule set" below = highest **validation** F1 in that experiment's front, scored on test.

## Folder map
- `code/` - all scripts (see `RUN_COMMANDS.txt`)
  - `nsga2_midterm.py` Exp 1 (6 features) | `nsga2_allfeatures.py` Exp 2-6 (all features; crossover, mutation, population, condition cap are arguments)
  - `nsga2_optuna.py`, `optuna_baseline.py` Optuna search | `param_search.py` + `nsga2_search.py` scikit-learn-style parameter search
  - `rf_benchmark.py` Random Forest benchmark + feature importance | `make_plots.py` redraws all graphs
- `results/exp1_6features` ... `results/exp6_cap50` - one folder per experiment:
  `nsga2_3obj_pareto_front.csv` (every rule set: train/val/test precision, recall, F1, total_conditions, n_rules, and the rule text in `rule_set`),
  `nsga2_3obj_history.csv` (hypervolume per generation), `nsga2_pareto_plot.png`
- `results/optuna/`, `results/parameter_search/`, `results/rf_benchmark/` - tuning and benchmark outputs
- `results/compare_*.png` - side-by-side graphs (Exp 1 vs 2, Exp 3 vs 4, Exp 4/5/6)

## Experiments, why we ran them, and what we found
| Exp | Settings | Why | Best rule set (test) | Finding |
|---|---|---|---|---|
| 1 | 6 features, pop 100, 150 gens, cx 0.7, mut 0.5 | Sponsor-specified features; controlled first comparison | 21 conditions, 5 rules; P 0.65 / R 0.45 / F1 0.53 | Limited features cap accuracy |
| 2 | All 40 features, same settings | Sponsors: do not limit features manually | 34 conditions, 8 rules; P 0.64 / R 0.70 / F1 0.67 | Big jump over Exp 1 |
| 3 | All features, pop 60, 100 gens, cx 0.5, mut 0.2 | Small-search setting for the team's comparison | 24 conditions, 6 rules; P 0.69 / R 0.61 / F1 0.65 | Too little search is worse |
| 4 | All features, pop 200, 250 gens, cx 0.9, mut 0.8 | Large-search setting | 122 conditions, 29 rules; P 0.85 / R 0.77 / F1 0.81 | Much better; compact set (20 cond.) reaches F1 0.74 |
| 5 | Exp 4 with mutation 0.2 | Effect of a lower mutation rate | 78 conditions, 19 rules; P 0.80 / R 0.69 / F1 0.74 | Worse than Exp 4 |
| 6 | Exp 4 with a 50-condition cap | Are short, readable rule sets enough? | 29 conditions, 7 rules; P 0.86 / R 0.69 / F1 0.76 | Almost no loss at 29 conditions |
| Optuna | 24 trials, pop 200 fixed; tunes generations, crossover, mutation | Systematic search instead of guessing | Best: 300 gens, cx 0.54, mut 0.55; validation HV 0.796 vs 0.784 for Exp 4 setting (3 seeds) | Small gain |
| Param search | 12 settings x 3 seeds (scikit-learn ParameterSampler) | Reusable search function for all three algorithms | Best: 200 gens, cx 0.7, mut 0.6 (HV 0.786) | Moderate crossover/mutation, 200-250 gens work best; 100 gens and mutation 0.1-0.2 worst |
| Random Forest | 300 trees, same split | Sponsors asked for a benchmark | P 0.98 / R 0.99 / F1 0.985 | Accuracy ceiling; a black box, not a rule system |

Hypervolume (HV) = one number for the whole front (higher is better; reference point [1,1,1]; complexity = conditions/160). It grows with the number of solutions, so compare algorithms at a similar front size.

## Main findings
1. Using all features beats 6 features by a wide margin (F1 0.53 -> 0.67).
2. A bigger search helps (Exp 3 -> 4), and low mutation hurts (Exp 5). Moderate-to-high crossover and mutation with about 200-300 generations work best.
3. Accuracy costs readability: the best sets are long (100+ conditions), but compact sets of 20-30 conditions reach F1 0.74-0.76, and a 50-condition cap loses almost nothing.
4. Tuning (Optuna / parameter search) gives only modest gains compared with using all features.
5. Interpretable rules reach roughly 80% of a Random Forest's F1.
6. Rules keep coming back to the same signals: new account (Account_Age_Days <= ~194), recent address change (<= ~400 days), very young email domain (<= ~142 days),
   high Behavioral_Anomaly_Score / IP_Risk_Score, and bursts of transactions. Most important features in the Random Forest: Email_Domain_Age_Days,
   Days_Since_Last_Address_Change, Account_Age_Days, Transactions_Last_7d, Time_Since_Last_Transaction.

## Example rule sets (from the Pareto fronts; see the `rule_set` column for all of them)
- Exp 4, compact (20 conditions, 5 rules; P 0.89 / R 0.64): e.g. `Account_Age_Days <= 194 AND Velocity_Transactions_24h > 4 AND Failed_Login_Count > 2`
  OR `Behavioral_Anomaly_Score > 31.3 AND IP_Risk_Score > 37.8 AND Days_Since_Last_Address_Change <= 400 AND Distance_From_Home_Location > 2.81` OR ...
- Exp 6 (29 conditions, 7 rules; P 0.86 / R 0.69): e.g. `Email_Domain_Age_Days <= 142 AND Transactions_Last_7d > 20 AND Days_Since_Last_Address_Change <= 400 AND Session_Duration <= 135.1` OR ...
- Exp 1 (6 features, 21 conditions): e.g. `Session_Duration <= 60.1 AND Transactions_Last_7d > 20 AND Credit_Limit_Utilization > 0.525` OR ...
Some rules contain repeated thresholds on the same feature (an artifact of the search); they can be simplified for a final version.

## For SPEA2 and MOEA/D: how to use the shared parameter search
`code/param_search.py` works for any algorithm. Write a small function `run(params, seed)` that returns a dict with at least `hv_val`
(validation hypervolume) and optionally `hv_test`, `best_val_f1`, `front`; then call
`param_search(run, {"generations": [...], "crossover_rate": [...], ...}, n_iter=12, seeds=range(3))`. See `nsga2_search.py` for a complete example.
Please use the same split, population size and condition cap so the tuning is comparable across algorithms.

## Fixes (Nov 19) - re-running the Optuna and parameter-search scripts
The saved results in `results/optuna` and `results/parameter_search` are valid: all 24 Optuna trials and all 36 search runs completed.
The bugs below only affected running the scripts **again**, which is why a re-run could look like it "did not run" or gave odd results:
- `param_search.py`: re-running with **different settings** but the same `--out` silently reused the old runs (matched only on config number)
  and printed the old results again. It now reuses a saved run only when its parameter values and seed match; other runs are moved to `*_OLD_*.csv`.
- `nsga2_optuna.py`: re-running into the same folder **added** another full set of trials to the old study (also when the population differed).
  It now keeps one study per population size and only tops up to the requested number of trials.
- `nsga2_optuna.py`: the seeded TPE sampler was not passed to the worker processes, so they used Optuna's unseeded default. Fixed.
- `nsga2_optuna.py`: printed nothing until the end (~35 min); it now prints a line after every trial.
- Both scripts now also work when run from a Colab/Jupyter cell (they used to fail on the extra `-f kernel.json` argument).

## Caveats
- Single seed for Exp 1-6, so small differences are within noise; the tuning runs used 3 seeds.
- Team splits are not fully aligned yet (SPEA2 used a stratified split; NSGA-II is chronological).
- Number of solutions differs between algorithms, which affects hypervolume.
- Synthetic data: results show how the methods behave, not production performance.
