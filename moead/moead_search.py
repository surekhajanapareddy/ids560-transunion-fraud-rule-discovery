"""Reuse teammate's identical param_search.py on final-population validation HV.
Example: python moead_search.py --csv synthetic_fraud_500k.csv --n_iter 12 --seeds 3 --pop 200 --out results/parameter_search/moead_search
"""
import argparse,os,json
p=argparse.ArgumentParser()
p.add_argument('--csv',default='synthetic_fraud_500k.csv')
p.add_argument('--n_iter',type=int,default=12)
p.add_argument('--seeds',type=int,default=3)
p.add_argument('--pop',type=int,default=200)
p.add_argument('--jobs',type=int,default=2)
p.add_argument('--out',default='results/parameter_search/moead_search')
p.add_argument('--gens',default='100,150,200,250,300')
p.add_argument('--cx',default='0.5,0.6,0.7,0.8,0.9')
p.add_argument('--mut',default='0.1,0.2,0.4,0.5,0.6,0.8')
a,_=p.parse_known_args()   # tolerate Jupyter/Colab '-f kernel.json' args
import moead_core
from moead_core import prepare_data,get_data_manifest,run_moead
from param_search import param_search

def moead_run(params, seed):
    if moead_core._MASKS is None: prepare_data(a.csv)   # worker processes on Windows/Mac start empty
    summary, _, _, _=run_moead(pop_n=a.pop,gens=params['generations'],cxpb=params['crossover_rate'],
        mutpb=params['mutation_rate'],seed=seed,condition_cap=160,
        track_archive=False,track_history=False)
    return summary  # hv_val is the ranking metric, as in teammate's NSGA-II search

if __name__=='__main__':
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
    prepare_data(a.csv)
    with open(a.out+"_data_manifest.json","w") as f:json.dump(get_data_manifest(),f,indent=2)
    space={'generations':[int(x) for x in a.gens.split(',')],
           'crossover_rate':[float(x) for x in a.cx.split(',')],
           'mutation_rate':[float(x) for x in a.mut.split(',')]}
    _,summary=param_search(moead_run,space,n_iter=a.n_iter or None,
        seeds=range(a.seeds),n_jobs=a.jobs,out_prefix=a.out)
    print('\nRanked by mean validation HV (population Pareto front):')
    print(summary[['generations','crossover_rate','mutation_rate','hv_val_mean','hv_val_std','best_val_f1_mean']].to_string(index=False))
