"""MOEA/D fixed-setting comparison vs NSGA-II experiments (same data/rules/HV semantics).

python moead_run.py --csv synthetic_fraud_500k.csv --pop 200 --gens 250 --cx .9 --mut .8 --seed 42 --cap 160 --out results/exp4
"""
import argparse, json, os
from moead_core import prepare_data, get_data_manifest, run_moead, front_frame
p=argparse.ArgumentParser()
p.add_argument('--csv',default='synthetic_fraud_500k.csv')
p.add_argument('--pop',type=int,default=200)
p.add_argument('--gens',type=int,default=250)
p.add_argument('--cx',type=float,default=.9)
p.add_argument('--mut',type=float,default=.8)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--cap',type=int,default=160)
p.add_argument('--features',choices=['all','six'],default='all')
p.add_argument('--out',default='results/exp4_moead')
p.add_argument('--no-force-variation',action='store_true',help='reproduce the original saved results (allows unchanged parent copies)')
p.add_argument('--allow-feature-count-mismatch',action='store_true',help='Only for local toy tests, not strict 500k benchmark')
a,_=p.parse_known_args()   # tolerate Jupyter/Colab '-f kernel.json' args
if __name__=='__main__':
    os.makedirs(a.out,exist_ok=True)
    prepare_data(a.csv, feature_mode=a.features, strict_feature_count=not a.allow_feature_count_mismatch)
    with open(f'{a.out}/data_manifest.json','w') as f:json.dump(get_data_manifest(),f,indent=2)
    summary,pop_front,archive,hist=run_moead(pop_n=a.pop,gens=a.gens,cxpb=a.cx,mutpb=a.mut,
        seed=a.seed,condition_cap=a.cap,track_archive=True,track_history=True,
        force_variation=not a.no_force_variation)
    # PRIMARY result = final population nondominated front, identical semantics to NSGA-II CSV.
    pf=front_frame(pop_front,condition_cap=a.cap)
    pf.to_csv(f'{a.out}/moead_3obj_pareto_front.csv',index=False)
    # ARCHIVE is a separate analysis output; do not plot it against NSGA-II final front.
    front_frame(archive,condition_cap=a.cap).to_csv(f'{a.out}/moead_historical_archive.csv',index=False)
    hist.to_csv(f'{a.out}/moead_3obj_history.csv',index=False)
    selected=pf.loc[pf.val_f1.idxmax()] if len(pf) else None
    summary.update({'algorithm':'MOEA/D','pop':a.pop,'gens':a.gens,'seed':a.seed,
                    'crossover_rate':a.cx,'mutation_rate':a.mut,
                    'condition_cap':a.cap,'features':a.features,
                    'final_unique_front_size':len(pf)})
    if selected is not None:
        summary['selected_rule_by_val_f1']={k:(bool(v) if isinstance(v,bool) else v.item() if hasattr(v,'item') else v)
           for k,v in selected[['val_precision','val_recall','val_f1',
                                 'test_precision','test_recall','test_f1',
                                 'total_conditions','n_rules']].to_dict().items()}
    with open(f'{a.out}/summary.json','w') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
