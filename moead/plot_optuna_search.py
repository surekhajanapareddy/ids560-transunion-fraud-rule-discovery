"""Create NSGA-II-format two-panel Optuna plot from actual trial results."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
inp=Path('results/optuna/optuna_trials.csv')
if not inp.is_file(): raise SystemExit('Missing optuna_trials.csv; Optuna must finish first')
df=pd.read_csv(inp)
df=df[df.state.eq('COMPLETE')].sort_values('number')
if df.empty: raise SystemExit('No completed Optuna trials')
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
ax[0].plot(df.number,df.value,'o',color='gray',label='trial')
ax[0].plot(df.number,df.value.cummax(),color='tab:red',label='best so far')
ax[0].set(xlabel='Trial',ylabel='Validation hypervolume',title='Optuna search progress')
ax[0].legend()
s=ax[1].scatter(df.crossover_rate,df.mutation_rate,c=df.value,s=60+6*(df.generations-100)/50*5,cmap='viridis')
fig.colorbar(s,ax=ax[1],label='Validation HV')
ax[1].set(xlabel='Crossover rate',ylabel='Mutation rate',title='Settings tried (larger dot = more generations)')
fig.suptitle('MOEA/D Optuna search (all features, pop 200)')
fig.tight_layout()
out=Path('results/optuna/optuna_search.png')
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=150)
plt.close(fig)
print('Saved',out)
