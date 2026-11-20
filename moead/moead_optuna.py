"""Optuna TPE search, matching the teammate's NSGA-II search protocol.
Search: 24 trials; pop=200; generations=100..300 step 50; crossover .5..95;
mutation .1..9; TPE seed=0; 8 startup trials; trial.seed=trial.number;
re-check top 3 using seeds 101,102,103, select by mean validation HV.

python moead_optuna.py --csv synthetic_fraud_500k.csv --trials 24 --pop 200 --workers 2 --out results/optuna
"""
import argparse, json, os, time
from multiprocessing import Pool
import optuna
import pandas as pd
from moead_core import prepare_data, get_data_manifest, run_moead
p=argparse.ArgumentParser()
p.add_argument('--csv',default='synthetic_fraud_500k.csv')
p.add_argument('--trials',type=int,default=24)
p.add_argument('--pop',type=int,default=200)
p.add_argument('--workers',type=int,default=2,
    help='2 follows teammate; 1 reduces Colab memory but may change TPE search trajectory')
p.add_argument('--out',default='results/optuna')
p.add_argument('--allow-feature-count-mismatch',action='store_true')
a,_=p.parse_known_args()   # tolerate Jupyter/Colab '-f kernel.json' args
DB=f'sqlite:///{os.path.abspath(a.out)}/moead_optuna.db'   # set at import so worker processes see it on every OS
STUDY=f'moead_pop{a.pop}'   # one study per population size, so runs with different pop never mix
import moead_core

def _ensure_data():
    # worker processes on Windows/Mac start empty; on Linux/Colab the data is already there
    if moead_core._MASKS is None: prepare_data(a.csv,strict_feature_count=not a.allow_feature_count_mismatch)

def run(gens,cx,mut,seed):
    _ensure_data()
    d,_,_,_=run_moead(pop_n=a.pop,gens=gens,cxpb=cx,mutpb=mut,seed=seed,
                      condition_cap=160,track_archive=False,track_history=False)
    return d

def objective(trial):
    gens=trial.suggest_int('generations',100,300,step=50)
    cx=trial.suggest_float('crossover_rate',.5,.95)
    mut=trial.suggest_float('mutation_rate',.1,.9)
    start=time.time()
    d=run(gens,cx,mut,seed=trial.number)
    for k,v in d.items():trial.set_user_attr(k,v)
    trial.set_user_attr('secs',round(time.time()-start,1))
    print(f"  trial {trial.number}: gens={gens} cx={cx:.2f} mut={mut:.2f} -> val HV {d['hv_val']:.4f}, "
          f"best val F1 {d['best_val_f1']:.4f} ({time.time()-start:.0f}s)",flush=True)
    return d['hv_val']

def worker(job):
    k,n=job
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # FIX: the sampler is NOT stored in the database, so it must be given again here
    # (before, workers silently used Optuna's unseeded default sampler)
    study=optuna.load_study(study_name=STUDY,storage=DB,
        sampler=optuna.samplers.TPESampler(seed=k,n_startup_trials=8))
    study.optimize(objective,n_trials=n)

def reseed(job):
    number,gens,cx,mut,seed=job
    return run(gens,cx,mut,seed)

if __name__=='__main__':
    if a.workers<1:raise ValueError('workers must be >=1')
    os.makedirs(a.out,exist_ok=True)
    prepare_data(a.csv,strict_feature_count=not a.allow_feature_count_mismatch)
    with open(f'{a.out}/data_manifest.json','w') as f: json.dump(get_data_manifest(),f,indent=2)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(study_name=STUDY,storage=DB,direction='maximize',load_if_exists=True)
    completed=len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    remaining=max(0,a.trials-completed)   # a re-run tops up to --trials instead of adding more
    print(f'study {STUDY}: {completed} trials already complete, running {remaining} more on {a.workers} worker(s)',flush=True)
    if remaining:
        counts=[remaining//a.workers+(i<remaining%a.workers) for i in range(a.workers)]
        jobs=[(k,c) for k,c in enumerate(counts) if c]
        if len(jobs)==1:
            worker(jobs[0])
        else:
            with Pool(len(jobs)) as pool: pool.map(worker,jobs)
    study=optuna.load_study(study_name=STUDY,storage=DB)
    tr=study.trials_dataframe(attrs=('number','value','params','user_attrs','state'))
    tr.columns=[c.replace('params_','').replace('user_attrs_','') for c in tr.columns]
    tr.sort_values('value',ascending=False).to_csv(f'{a.out}/optuna_trials.csv',index=False)
    top=tr[tr.state.eq('COMPLETE')].sort_values('value',ascending=False).head(3)
    if top.empty:raise RuntimeError('No complete trials')
    jobs=[(int(t.number),int(t.generations),float(t.crossover_rate),float(t.mutation_rate),seed)
          for _,t in top.iterrows() for seed in (101,102,103)]
    if a.workers>1:
        with Pool(a.workers) as pool: res=pool.map(reseed,jobs)
    else:
        res=list(map(reseed,jobs))
    rows=[{'trial':j[0],'generations':j[1],'crossover_rate':j[2],'mutation_rate':j[3],
           'seed':j[4],**r} for j,r in zip(jobs,res)]
    retest=pd.DataFrame(rows)
    retest.to_csv(f'{a.out}/optuna_top3_reseeded.csv',index=False)
    summary=retest.groupby(['trial','generations','crossover_rate','mutation_rate'])[['hv_val','hv_test','best_val_f1']].agg(['mean','std'])
    summary=summary.sort_values(('hv_val','mean'),ascending=False)
    print('Top 3 trials retested on seeds 101, 102, 103:')
    print(summary.to_string())
    best=summary.index[0]
    with open(f'{a.out}/best_params.json','w') as f:
        json.dump({'pop':a.pop,'generations':int(best[1]),'crossover_rate':float(best[2]),'mutation_rate':float(best[3])},f,indent=2)
