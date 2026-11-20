"""NSGA-II-controlled Exp 1-6 runner. Run from any directory.
Examples:
    python run_experiments.py --exp 4
    python run_experiments.py --exp all --skip-existing
"""
import argparse, subprocess, sys, json, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = {
 1: dict(features='six',pop=100,gens=150,cx=.7,mut=.5,cap=160),
 2: dict(features='all',pop=100,gens=150,cx=.7,mut=.5,cap=160),
 3: dict(features='all',pop=60,gens=100,cx=.5,mut=.2,cap=160),
 4: dict(features='all',pop=200,gens=250,cx=.9,mut=.8,cap=160),
 5: dict(features='all',pop=200,gens=250,cx=.9,mut=.2,cap=160),
 6: dict(features='all',pop=200,gens=250,cx=.9,mut=.8,cap=50),
}

def matches_existing(out, config, csv_sha, force_variation=True):
    """Only reuse outputs with complete primary CSVs and matching settings."""
    req = ('summary.json', 'data_manifest.json','moead_3obj_pareto_front.csv','moead_3obj_history.csv')
    if not all((out/f).is_file() and (out/f).stat().st_size for f in req): return False
    try:
        s=json.loads((out/'summary.json').read_text())
        m=json.loads((out/'data_manifest.json').read_text())
        return (s.get('pop')==config['pop'] and s.get('gens')==config['gens'] and
                s.get('seed')==42 and s.get('condition_cap')==config['cap'] and
                s.get('features')==config['features'] and
                s.get('crossover_rate')==config['cx'] and
                s.get('mutation_rate')==config['mut'] and
                s.get('force_variation',False)==force_variation and
                m.get('n_transactions')==500000 and m.get('csv_sha256')==csv_sha and
                m.get('n_features')==(6 if config['features']=='six' else 40))
    except (ValueError,KeyError,TypeError,OSError):return False

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='synthetic_fraud_500k.csv')
    p.add_argument('--exp',required=True,choices=[str(i) for i in range(1,7)]+['all'])
    p.add_argument('--out',default='results')
    p.add_argument('--no-force-variation',action='store_true',help='reproduce the original saved results')
    p.add_argument('--skip-existing',action='store_true',help='Skip complete outputs with matching settings; audit dataset manifest separately')
    a,_=p.parse_known_args()
    csv=Path(a.csv).expanduser()
    if not csv.is_absolute(): csv=Path.cwd()/csv
    if not csv.is_file(): raise FileNotFoundError(csv)
    with csv.open('rb') as f: csv_sha=hashlib.file_digest(f,'sha256').hexdigest()
    results=Path(a.out).expanduser()
    if not results.is_absolute(): results=HERE/results
    for idx in (list(CONFIGS) if a.exp=='all' else [int(a.exp)]):
        c=CONFIGS[idx]
        out=results/f'exp{idx}_moead'
        if a.skip_existing and matches_existing(out,c,csv_sha,not a.no_force_variation):
            print(f'SKIP Exp {idx} (existing files, settings match): {out}',flush=True)
            continue
        cmd=[sys.executable,str(HERE/'moead_run.py'),'--csv',str(csv),
             '--features',c['features'],'--pop',str(c['pop']),'--gens',str(c['gens']),
             '--cx',str(c['cx']),'--mut',str(c['mut']),'--cap',str(c['cap']),
             '--seed','42','--out',str(out)]+(['--no-force-variation'] if a.no_force_variation else [])
        print('RUN Exp',idx,':',' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True,cwd=str(HERE))
    print('Finished requested experiments. Run verify_completion.py to audit outputs.')
