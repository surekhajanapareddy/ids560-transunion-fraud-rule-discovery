"""Re-evaluate fixed params on seeds 101,102,103 against Optuna selected params."""
import argparse,os
import pandas as pd
from moead_core import prepare_data,run_moead
p=argparse.ArgumentParser()
p.add_argument('--csv',default='synthetic_fraud_500k.csv')
p.add_argument('--pop',type=int,default=200)
p.add_argument('--gens',type=int,default=250)
p.add_argument('--cx',type=float,default=.9)
p.add_argument('--mut',type=float,default=.8)
p.add_argument('--out',default='results/optuna/exp4_setting_same_seeds.csv')
a,_=p.parse_known_args()
if __name__=='__main__':
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
    prepare_data(a.csv)
    rows=[]
    for s in (101,102,103):
        d,_,_,_=run_moead(pop_n=a.pop,gens=a.gens,cxpb=a.cx,mutpb=a.mut,seed=s,
                          condition_cap=160,track_archive=False,track_history=False)
        rows.append({'setting':f'Exp #4 ({a.gens} gens, cx {a.cx:g}, mut {a.mut:g})',
                     'hv_val':d['hv_val'],'hv_test':d['hv_test'],
                     'best_val_f1':d['best_val_f1'],'front':d['front'],
                     'seed':s,'generations':a.gens,
                     'crossover_rate':a.cx,'mutation_rate':a.mut,
                     'selected_test_f1':d.get('selected_test_f1'),
                     'n_evaluations':d.get('n_evaluations')})
    df=pd.DataFrame(rows)
    df.to_csv(a.out,index=False)
    print(df.to_string(index=False))
    print('Mean val HV:',df.hv_val.mean())
