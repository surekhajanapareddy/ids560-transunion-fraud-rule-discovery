# Adaptive Rule Discovery for Fraud Detection

**IDS 560 Capstone, University of Illinois Chicago · Group 1 · Sponsor: TransUnion**

Team: Wenyao Mo, Ananya Gannamaneni, Surekha Janapareddy, Caroline Edmunds-Diez

We use multi-objective evolutionary algorithms to discover short, readable fraud detection rules. Each algorithm balances three goals at once:

- **Precision ↑**: of the transactions flagged, the share that are fraud
- **Recall ↑**: of all fraud, the share that gets flagged
- **Conditions ↓**: total conditions in the rule set, with TransUnion's hard limit of 160

A *rule* is a set of conditions joined by AND (e.g. `Account_Age_Days <= 194 AND Velocity_Transactions_24h > 4`). A *rule set* is several rules joined by OR. Each algorithm returns a Pareto front: a menu of rule sets with different trade-offs instead of one answer.

## Algorithms

| Folder | Algorithm | Selection |
|---|---|---|
| [`nsga2/`](nsga2) | NSGA-II | Non-dominated sorting + crowding distance (DEAP `selNSGA2`, `selTournamentDCD`) |
| [`spea2/`](spea2) | SPEA2 | Strength + nearest-neighbour density (DEAP `selSPEA2`) |
| [`moead/`](moead) | MOEA/D | Weighted Tchebycheff decomposition with neighbourhoods |

All three share the same data split, rule representation, crossover, mutation, repair, condition cap and hypervolume, so only the search strategy differs.

## Setup (identical for all three)

- `synthetic_fraud_500k.csv`: 500,000 transactions, 2,500 fraud (0.5%), 40 candidate features
- Chronological 70 / 15 / 15 train / validation / test split (test = 75,000 newer transactions, 394 fraud)
- Rules are learned on **train**; the best rule set is picked by **validation** F1; **test** is used once for reporting
- Max 5 conditions per rule, 30 rules per set; a rule set must catch at least 5 frauds on train
- Hypervolume on `[1 - precision, 1 - recall, conditions / 160]` with reference point `[1, 1, 1]` (pymoo `HV`)

## Results (test set, best rule set by validation F1)

**Up to 160 conditions (Experiment 4: population 200, 250 generations)**

| Algorithm | Precision | Recall | F1 | Rules | Conditions |
|---|---|---|---|---|---|
| NSGA-II | 85% | 77% | 81% | 29 | 122 |
| MOEA/D | 82% | **82%** | **82%** | 29 | 119 |
| SPEA2 | **91%** | 70% | 79% | **21** | **89** |

**Same search with a 50 condition cap (Experiment 6)**

| Algorithm | Precision | Recall | F1 | Rules | Conditions |
|---|---|---|---|---|---|
| NSGA-II | 86% | 69% | 76% | **7** | **29** |
| MOEA/D | 80% | **78%** | 79% | 13 | 50 |
| SPEA2 | **88%** | 77% | **82%** | 12 | 45 |

**Test F1 (%) by experiment**

| Experiment | Settings | NSGA-II | MOEA/D | SPEA2 |
|---|---|---|---|---|
| 1. 6 features | 6 sponsor features, pop 100, 150 gens, cx 0.7, mut 0.5 | 53 | 60 | 56 |
| 2. All 40 features | same settings, all features | 67 | 75 | 69 |
| 3. Small search | pop 60, 100 gens, cx 0.5, mut 0.2 | 65 | 54 | 60 |
| 4. Large search | pop 200, 250 gens, cx 0.9, mut 0.8 | 81 | 82 | 79 |
| 5. Less random change | Exp 4 with mutation 0.2 | 74 | 72 | 80 |
| 6. 50 condition cap | Exp 4 with a 50 condition cap | 76 | 79 | 82 |

### Key findings

- Using all 40 features instead of 6 added **13 to 15** F1 points.
- A larger search added **16 to 28** F1 points. Optuna (24 trials) and a scikit-learn parameter search (12 settings × 3 seeds) confirmed these settings; tuning gained at most 0.025 in validation hypervolume.
- A 50 condition cap kept F1 at **76 to 82%**, so short, readable rule sets lose very little.
- **Pareto tie:** SPEA2 led on precision and MOEA/D on recall; no algorithm won on both.
- **Conclusion:** our rule sets catch 69 to 82% of fraud at 80 to 91% precision, well under the 160 condition limit. More features and a larger search mattered more than the choice of algorithm.
- A Random Forest benchmark (300 trees, same split) reaches F1 ≈ 0.985, which is the accuracy ceiling, but it is a black box rather than a rule system.

## Repository layout

```
nsga2/       code/, results/ (Exp 1-6, Optuna, parameter search, RF benchmark), README, RUN_COMMANDS
spea2/       same structure as nsga2/
moead/       moead_core.py and runner scripts, results/, README, RUN_INSTRUCTIONS, Colab notebooks
midterm/     midterm notebooks (2K-row sample) for NSGA-II, SPEA2 and MOEA/D
presentation/  midterm and final slides
```

Each algorithm folder has its own README with the full protocol, results and a **"Fixes (Nov 20)"** section (Nov 19 for NSGA-II) describing code corrections made after the saved results were produced. The saved results are unchanged and valid; the fixes affect future re-runs.

## How to run

```bash
pip install numpy pandas matplotlib deap pymoo scikit-learn optuna
# place synthetic_fraud_500k.csv in the algorithm's code folder, then see its RUN_COMMANDS / RUN_INSTRUCTIONS
python nsga2/code/nsga2_allfeatures.py synthetic_fraud_500k.csv 200 250 42 0.9 0.8 out_exp4 160
```

The dataset is **not** included in this repository (it is 119 MB and sponsor project data). MOEA/D results were produced from a file with SHA-256 `442f264eed7648bf0d227a53bb171130dda7639adcf572c8376c68cda1d1909d`.

## Limitations

- Experiments 1 to 6 use a single seed (42), so differences of 1 to 2 points are within noise; tuning runs used 3 seeds.
- Front sizes differ between algorithms, which affects hypervolume.
- Synthetic data: results show how the methods behave, not production performance.
