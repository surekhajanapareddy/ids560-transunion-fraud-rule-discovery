"""Strict alignment audit of MOEA/D search deliverables against the NSGA-II search protocol.
This only validates outputs; it does not generate or fabricate any runs.
"""
from pathlib import Path
import pandas as pd
import json, sys
root=Path(sys.argv[1] if len(sys.argv)>1 else 'results')
issues=[]
def bad(msg):
    print('INCOMPLETE:',msg)
    issues.append(msg)

def csv_file(rel):
    p=root/rel
    if not p.is_file():
        bad('missing '+str(p))
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        bad('unreadable '+str(p)+': '+str(e))
        return None

runs=csv_file('parameter_search/moead_search_runs.csv')
summary=csv_file('parameter_search/moead_search_summary.csv')
if runs is not None:
    expected_seeds={0,1,2}
    if len(runs)!=36:bad(f'scikit-learn runs: expected 36, found {len(runs)}')
    if 'config_id' not in runs or 'seed' not in runs:bad('scikit-learn runs missing config_id/seed')
    else:
        if runs['config_id'].nunique()!=12:bad(f'scikit-learn configs: expected 12, found {runs["config_id"].nunique()}')
        for config,g in runs.groupby('config_id'):
            if len(g)!=3 or set(g.seed)!=expected_seeds:bad(f'config {config}: expected exactly seeds 0,1,2')
        if runs.duplicated(['config_id','seed']).any():bad('duplicate scikit-learn config_id/seed')
    if 'hv_val' not in runs or runs.hv_val.isna().any():bad('missing scikit-learn validation HV')
    if 'generations' in runs and 'crossover_rate' in runs and 'mutation_rate' in runs:
        allowed_gens={100,150,200,250,300}
        allowed_cx={0.5,0.6,0.7,0.8,0.9}
        allowed_mut={0.1,0.2,0.4,0.5,0.6,0.8}
        if not set(runs.generations).issubset(allowed_gens):bad('scikit-learn generation grid differs')
        if not set(runs.crossover_rate.round(6)).issubset(allowed_cx):bad('scikit-learn crossover grid differs')
        if not set(runs.mutation_rate.round(6)).issubset(allowed_mut):bad('scikit-learn mutation grid differs')
if summary is not None:
    if len(summary)!=12:bad(f'scikit-learn summary: expected 12 configs, found {len(summary)}')
    if 'hv_val_mean' not in summary:bad('scikit-learn summary missing hv_val_mean')

trials=csv_file('optuna/optuna_trials.csv')
if trials is not None:
    completed=trials[trials['state'].eq('COMPLETE')] if 'state' in trials else None
    if completed is None:bad('Optuna trials missing state')
    elif len(completed)!=24:bad(f'Optuna completed trials: expected 24, found {len(completed)}')
    if 'number' not in trials or trials.number.nunique()!=len(trials):bad('Optuna trial numbers not unique')
    for field in ('generations','crossover_rate','mutation_rate','value'):
        if field not in trials:bad('Optuna trials missing '+field)
        elif trials[field].isna().any():bad('Optuna trials contain null '+field)
    if completed is not None and len(completed):
        if 'generations' in completed and not set(completed.generations).issubset({100,150,200,250,300}):bad('Optuna generations grid differs')
        if 'crossover_rate' in completed and not completed.crossover_rate.between(.5,.95).all():bad('Optuna crossover range differs')
        if 'mutation_rate' in completed and not completed.mutation_rate.between(.1,.9).all():bad('Optuna mutation range differs')
reseed=csv_file('optuna/optuna_top3_reseeded.csv')
if reseed is not None:
    if len(reseed)!=9:bad(f'Optuna top-3 reseeding: expected 9 runs, found {len(reseed)}')
    if 'trial' not in reseed or 'seed' not in reseed:bad('reseed file missing trial/seed')
    else:
        if reseed.trial.nunique()!=3:bad('Optuna reseed has not exactly 3 trials')
        for t,g in reseed.groupby('trial'):
            if len(g)!=3 or set(g.seed)!={101,102,103}:bad(f'Optuna trial {t} reseeds not 101,102,103')
        if trials is not None and 'number' in trials and 'value' in trials and 'state' in trials:
            top3=set(trials[trials.state.eq('COMPLETE')].nlargest(3,'value').number)
            if set(reseed.trial)!=top3:bad('reseed trial IDs do not match top 3 Optuna results')
    if 'hv_val' not in reseed or reseed.hv_val.isna().any():bad('Optuna retest missing hv_val')

baseline=csv_file('optuna/exp4_setting_same_seeds.csv')
if baseline is not None:
    if len(baseline)!=3:bad(f'Exp4 baseline: expected 3 seeds, found {len(baseline)}')
    if 'seed' not in baseline or set(baseline.seed)!={101,102,103}:bad('Exp4 baseline seeds not 101,102,103')
    for field,v in [('generations',250),('crossover_rate',.9),('mutation_rate',.8)]:
        if field not in baseline or not baseline[field].eq(v).all():bad('Exp4 baseline mismatch '+field)
    if 'hv_val' not in baseline or baseline.hv_val.isna().any():bad('Exp4 baseline missing hv_val')

best=root/'optuna/best_params.json'
if not best.is_file():bad('missing '+str(best))
else:
    try:
        params=json.loads(best.read_text())
        for k in ('pop','generations','crossover_rate','mutation_rate'):
            if k not in params:bad('best_params missing '+k)
        if params.get('pop')!=200:bad('best_params pop != 200')
        if reseed is not None and len(reseed)>0 and all(k in reseed for k in ('trial','hv_val','generations','crossover_rate','mutation_rate')):
            winner=reseed.groupby('trial').hv_val.mean().idxmax()
            rec=reseed[reseed.trial.eq(winner)].iloc[0]
            for field in ('generations','crossover_rate','mutation_rate'):
                if abs(params.get(field,0)-float(rec[field]))>1e-9:bad('best_params differs from retest winner '+field)
    except Exception as e:bad('best_params parse error: '+str(e))
# The NSGA-II package also provides a shared Random Forest benchmark.
rf=csv_file('rf_benchmark/rf_benchmark_results.csv')
if rf is not None:
    if len(rf)!=4:bad(f'RF result rows: expected 4, found {len(rf)}')
    if not all(col in rf for col in ('threshold','split','precision','recall','f1')):bad('RF results missing expected columns')
fi=csv_file('rf_benchmark/rf_feature_importance.csv')
if fi is not None and len(fi)!=40:bad(f'RF feature importances: expected 40, found {len(fi)}')
plot=root/'optuna/optuna_search.png'
if not plot.is_file():bad('missing Optuna search plot '+str(plot))
print('\n'+'='*56)
print('TUNING AUDIT:', 'PASS' if not issues else 'FAIL', '| issues:',len(issues))
print('Required: 12 x 3 scikit-learn runs, 24 Optuna trials, 3 x 3 reseeds, 3 baseline seeds, RF benchmark, best_params and Optuna plot')
print('='*56)
if issues:sys.exit(1)
