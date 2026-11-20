"""Report deliverable completeness; does not claim that missing results were run.
python verify_completion.py --csv synthetic_fraud_500k.csv
"""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--root',default='results')
p.add_argument('--csv',default='synthetic_fraud_500k.csv')
a,_=p.parse_known_args()
root=Path(a.root)
if not root.is_absolute():root=Path(__file__).resolve().parent/root
csv=Path(a.csv)
if not csv.is_absolute():csv=Path.cwd()/csv
actual_hash=None
if csv.is_file():
    with csv.open('rb') as f:actual_hash=hashlib.file_digest(f,'sha256').hexdigest()
print('DATA SHA256:',actual_hash or 'CSV not found')
complete=0
for i, gens in ((1,150),(2,150),(3,100),(4,250),(5,250),(6,250)):
    folder=root/f'exp{i}_moead'
    names=['summary.json','data_manifest.json','moead_3obj_history.csv','moead_3obj_pareto_front.csv']
    miss=[n for n in names if not (folder/n).is_file()]
    flags=[]
    if not miss:
        try:
            s=json.loads((folder/'summary.json').read_text())
            m=json.loads((folder/'data_manifest.json').read_text())
            h=pd.read_csv(folder/'moead_3obj_history.csv')
            f=pd.read_csv(folder/'moead_3obj_pareto_front.csv')
            if len(h)!=gens:flags.append(f'history rows {len(h)} != {gens}')
            if len(f)==0:flags.append('empty Pareto CSV')
            if actual_hash and m.get('csv_sha256')!=actual_hash:flags.append('CSV SHA MISMATCH')
            if s.get('gens')!=gens:flags.append('summary gens mismatch')
        except Exception as e:flags.append(f'cannot validate files: {e}')
    okay=not miss and not flags
    complete+=int(okay)
    print(f'Exp {i}: {"COMPLETE" if okay else "INCOMPLETE"}' +
          (f'; missing: {", ".join(miss)}' if miss else '')+
          (f'; issues: {", ".join(flags)}' if flags else ''))
required={
  'Scikit-learn parameter search': ['parameter_search/moead_search_runs.csv','parameter_search/moead_search_summary.csv'],
  'Optuna': ['optuna/optuna_trials.csv','optuna/optuna_top3_reseeded.csv','optuna/best_params.json'],
  'Optuna baseline (same seeds)': ['optuna/exp4_setting_same_seeds.csv'],
}
for label,fs in required.items():
    missing=[x for x in fs if not (root/x).is_file()]
    print(f'{label}: {"COMPLETE" if not missing else "MISSING"}'+(' - '+', '.join(missing) if missing else ''))
print(f'Completed Exp 1-6: {complete}/6')
