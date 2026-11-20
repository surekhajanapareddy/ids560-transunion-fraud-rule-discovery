"""Regenerates every graph from the CSVs in ../results.  Usage: python make_plots.py   (run from the code folder)"""
import os, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
T = {"exp1_6features": "Exp #1: 6 features (pop 100, 150 gens, cx 0.7, mut 0.5)", "exp2_allfeatures": "Exp #2: all features (pop 100, 150 gens, cx 0.7, mut 0.5)",
     "exp3_allfeatures": "Exp #3: all features (pop 60, 100 gens, cx 0.5, mut 0.2)", "exp4_allfeatures": "Exp #4: all features (pop 200, 250 gens, cx 0.9, mut 0.8)",
     "exp5_lower_mutation": "Exp #5: all features (pop 200, 250 gens, cx 0.9, mut 0.2)", "exp6_cap50": "Exp #6: all features, 50-condition cap (pop 200, 250 gens, cx 0.9, mut 0.8)"}
def load(d): return pd.read_csv(f"{R}/{d}/nsga2_3obj_pareto_front.csv"), pd.read_csv(f"{R}/{d}/nsga2_3obj_history.csv")
for d, title in T.items():
    f, h = load(d); fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    s = ax[0].scatter(f.test_recall, f.test_precision, c=f.total_conditions, cmap="viridis", vmin=1, vmax=f.total_conditions.max(), s=38)
    ax[0].set(xlabel="Recall (test)", ylabel="Precision (test)", title="Pareto front: precision vs recall, colored by complexity"); fig.colorbar(s, ax=ax[0], label="Total conditions")
    ax[1].plot(h.gen, h.hv3d, color="darkorange"); ax[1].set(xlabel="Generation", ylabel="3D hypervolume", title="Hypervolume over generations")
    fig.suptitle(f"NSGA-II - {title}", fontsize=11); fig.tight_layout(); fig.savefig(f"{R}/{d}/nsga2_pareto_plot.png", dpi=150); plt.close(fig)
def compare(dirs, labels, colors, name, title):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for d, lab, c in zip(dirs, labels, colors):
        f, _ = load(d); lab = f"{lab} (n={len(f)})"
        ax[0].scatter(f.test_recall, f.test_precision, s=14 + f.total_conditions / 2.5, alpha=.45, color=c, label=lab)
        g = f.sort_values("total_conditions"); ax[1].scatter(g.total_conditions, g.test_f1, s=9, alpha=.2, color=c)
        ax[1].step(g.total_conditions, g.test_f1.cummax(), where="post", lw=2.2, color=c, label=lab)
    ax[0].set(xlabel="Recall (test)", ylabel="Precision (test)", title="Precision-recall trade-off (bubble size = conditions)"); ax[0].legend(loc="lower left")
    ax[1].set(xlabel="Total conditions", ylabel="Best F1 (test) at or below this many conditions", title="Complexity vs best F1"); ax[1].legend(loc="lower right")
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(f"{R}/{name}", dpi=150); plt.close(fig)
compare(["exp1_6features", "exp2_allfeatures"], ["Exp #1: 6 features", "Exp #2: all features"], ["tab:purple", "tab:red"], "compare_exp1_vs_exp2.png", "NSGA-II: 6 features vs all features")
compare(["exp3_allfeatures", "exp4_allfeatures"], ["Exp #3", "Exp #4"], ["tab:blue", "tab:green"], "compare_exp3_vs_exp4.png", "NSGA-II, all features: Experiment #3 vs #4")
compare(["exp4_allfeatures", "exp5_lower_mutation", "exp6_cap50"], ["#4 base (mut 0.8)", "#5 (mut 0.2)", "#6 (cap 50)"], ["tab:blue", "tab:orange", "tab:green"], "compare_exp4_5_6.png", "NSGA-II, all features: Experiments #4, #5, #6")
t = pd.read_csv(f"{R}/optuna/optuna_trials.csv").sort_values("number")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(t.number, t.value, "o", color="gray", label="trial"); ax[0].plot(t.number, t.value.cummax(), color="tab:red", label="best so far")
ax[0].set(xlabel="Trial", ylabel="Validation hypervolume", title="Optuna search progress"); ax[0].legend()
s = ax[1].scatter(t.crossover_rate, t.mutation_rate, c=t.value, s=60 + 6 * (t.generations - 100) / 50 * 5, cmap="viridis"); fig.colorbar(s, ax=ax[1], label="Validation HV")
ax[1].set(xlabel="Crossover rate", ylabel="Mutation rate", title="Settings tried (bigger dot = more generations)")
fig.suptitle("NSGA-II Optuna search (all features, pop 200)", fontsize=11); fig.tight_layout(); fig.savefig(f"{R}/optuna/optuna_search.png", dpi=150)
print("plots written to", os.path.abspath(R))
