"""Verify that MOEA/D's feature/condition bank is bit-for-bit consistent with
    the uploaded NSGA-II source's preprocessing definitions on a GIVEN csv.
    NOTE: cannot independently verify teammate's real run without her exact CSV/hash.
"""
import argparse, json, hashlib
import numpy as np, pandas as pd
from moead_core import prepare_data, get_data_manifest, SIX_FEATURES, Q, LABEL, TIME
p=argparse.ArgumentParser()
p.add_argument('--csv',default='synthetic_fraud_500k.csv')
p.add_argument('--features',choices=['all','six'],default='all')
p.add_argument('--out',default='data_parity_report.json')
p.add_argument('--allow-feature-count-mismatch',action='store_true')
a,_=p.parse_known_args()
if __name__=='__main__':
    m=prepare_data(a.csv,feature_mode=a.features,strict_feature_count=not a.allow_feature_count_mismatch)
    import moead_core as mc
    # Independent direct transcription of nsga2_allfeatures.py / nsga2_midterm.py.
    if a.features=='all':
        df=pd.read_csv(a.csv).sort_values(TIME).reset_index(drop=True)
        cats=[c for c in df.columns if df[c].dtype==object or str(df[c].dtype).startswith('str')]
        nums=[c for c in df.columns if c not in cats+[LABEL,TIME]]
    else:
        df=pd.read_csv(a.csv,usecols=SIX_FEATURES+[LABEL,TIME]).sort_values(TIME).reset_index(drop=True)
        cats=[];nums=SIX_FEATURES
    n=len(df); train=df.iloc[:int(.70*n)]
    conds=[]
    for f in nums:
        for t in np.unique(np.quantile(train[f].values,Q)):
            conds.append((f,'<=',float(t)));conds.append((f,'>',float(t)))
    for f in cats:
        for v in sorted(train[f].unique()):conds.append((f,'=',v))
    # Check exact tuple order, float thresholds, type and categorical values,
    # not only feature count and printed split sizes.
    assert len(conds)==len(mc._CONDITIONS), 'Condition lengths differ'
    for i,(a1,b1) in enumerate(zip(conds,mc._CONDITIONS)):
        if a1!=b1:raise AssertionError(f'Condition #{i} differs: {a1!r} vs {b1!r}')
    assert list(nums)==m['num_features'] and list(cats)==m['cat_features'], 'Feature order differs'
    report={'preprocessing_matches_supplied_nsga_source':True,
            'verified_against':'source definitions, not teammate actual output',**m}
    with open(a.out,'w') as f:json.dump(report,f,indent=2)
    print(f"PASS: exact features and all {len(conds)} ordered conditions match the teammate's NSGA-II source.\n"
          f"Compare CSV SHA-256 {m['csv_sha256']} against the value from her environment.")
