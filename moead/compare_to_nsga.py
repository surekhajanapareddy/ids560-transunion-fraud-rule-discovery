"""Compare a finished NSGA-II experiment directory with a matched MOEA/D directory.

python compare_to_nsga.py --nsga ../NSGA2_all_results/results/exp4_allfeatures --moead results/exp4_moead
Check that the CSVs came from the exact same input bytes before interpreting numbers.
"""
import argparse,os,json,pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--nsga',required=True)
p.add_argument('--moead',required=True)
p.add_argument('--out',default='comparison.csv')
a,_=p.parse_known_args()
rows=[]
for name,d,basename in [('NSGA-II',a.nsga,'nsga2'),('MOEA/D',a.moead,'moead')]:
    f=pd.read_csv(os.path.join(d,basename+'_3obj_pareto_front.csv'))
    h=pd.read_csv(os.path.join(d,basename+'_3obj_history.csv'))
    assert len(f) and len(h)
    best=f.sort_values(['val_f1','total_conditions'],ascending=[False,True]).iloc[0]
    rows.append({'algorithm':name,'unique_final_front_size':len(f),
        'last_train_hv':float(h.hv3d.iloc[-1]),
        'best_val_f1':float(best.val_f1),
        'selected_test_precision':float(best.test_precision),
        'selected_test_recall':float(best.test_recall),
        'selected_test_f1':float(best.test_f1),
        'selected_conditions':int(best.total_conditions),
        'selected_n_rules':int(best.n_rules),
        'generations':len(h)})
report=pd.DataFrame(rows)
report.to_csv(a.out,index=False)
print(report.to_string(index=False))
print('\nIMPORTANT: this program compares output metrics, but NSGA-II original '
      'files do not contain the original dataset SHA-256. Verify same input file manually.')
