"""Generate MOEA/D experiment figures from existing results, no model rerun.
python make_moead_plots.py --root results
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',default='results');a,_=p.parse_known_args()
root=Path(a.root)
if not root.is_absolute(): root=Path(__file__).resolve().parent/root
records=[]
for i in range(1,7):
    folder=root/f'exp{i}_moead'
    pf=folder/'moead_3obj_pareto_front.csv'
    history=folder/'moead_3obj_history.csv'
    if not (pf.exists() and history.exists()):
        print('No complete result for Exp',i,'— skip')
        continue
    df=pd.read_csv(pf);h=pd.read_csv(history)
    if len(df):
        fig,ax=plt.subplots(figsize=(8,5))
        sc=ax.scatter(df.total_conditions,df.test_recall,c=df.test_precision,s=22,cmap='viridis',vmin=0,vmax=1)
        fig.colorbar(sc,ax=ax,label='Test precision')
        ax.set(xlabel='Total conditions',ylabel='Test recall',title=f'MOEA/D Exp {i}: Final-population Pareto solutions')
        fig.tight_layout();fig.savefig(folder/'moead_pareto_plot.png',dpi=150);plt.close(fig)
        best=df.sort_values(['val_f1','total_conditions'],ascending=[False,True]).iloc[0]
        records.append({'Exp':f'Exp {i}','Test F1':best.test_f1,'Train HV':h.hv3d.iloc[-1]})
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(h.gen,h.hv3d)
    ax.set(xlabel='Generation',ylabel='Train HV (final population)',title=f'MOEA/D Exp {i}: Hypervolume by generation')
    fig.tight_layout();fig.savefig(folder/'moead_hv_history.png',dpi=150);plt.close(fig)
if records:
    df=pd.DataFrame(records);df.to_csv(root/'moead_experiment_comparison.csv',index=False)
    for metric, filename in [('Test F1','moead_exp_test_f1.png'),('Train HV','moead_exp_train_hv.png')]:
        fig,ax=plt.subplots(figsize=(8,5));ax.bar(df.Exp,df[metric]);ax.set(ylabel=metric,title='MOEA/D: Experiment comparison')
        fig.tight_layout();fig.savefig(root/filename,dpi=150);plt.close(fig)
print('Plots generated for available experiments only.')

# Same three pairwise comparison deliverables as teammate NSGA-II, using MOEA/D fronts.
def pair_compare(expids,labels,colors,name,title):
    data=[]
    for i in expids:
        p=root/f'exp{i}_moead/moead_3obj_pareto_front.csv'
        if not p.is_file():
            print('Skipping comparison:',name,'— missing',p);return
        data.append(pd.read_csv(p))
    fig,ax=plt.subplots(1,2,figsize=(13,4.6))
    for f,lab,col in zip(data,labels,colors):
        lab=f'{lab} (n={len(f)})'
        ax[0].scatter(f.test_recall,f.test_precision,s=14+f.total_conditions/2.5,alpha=.45,color=col,label=lab)
        g=f.sort_values('total_conditions')
        ax[1].scatter(g.total_conditions,g.test_f1,s=9,alpha=.2,color=col)
        ax[1].step(g.total_conditions,g.test_f1.cummax(),where='post',lw=2.2,color=col,label=lab)
    ax[0].set(xlabel='Recall (test)',ylabel='Precision (test)',title='Precision-recall trade-off (bubble size = conditions)');ax[0].legend(loc='lower left')
    ax[1].set(xlabel='Total conditions',ylabel='Best F1 (test) at or below this many conditions',title='Complexity vs best F1');ax[1].legend(loc='lower right')
    fig.suptitle(title,fontsize=11);fig.tight_layout();fig.savefig(root/name,dpi=150);plt.close(fig)
pair_compare([1,2],['Exp #1: 6 features','Exp #2: all features'],['tab:purple','tab:red'],'compare_exp1_vs_exp2.png','MOEA/D: 6 features vs all features')
pair_compare([3,4],['Exp #3','Exp #4'],['tab:blue','tab:green'],'compare_exp3_vs_exp4.png','MOEA/D, all features: Experiment #3 vs #4')
pair_compare([4,5,6],['#4 base (mut 0.8)','#5 (mut 0.2)','#6 (cap 50)'],['tab:blue','tab:orange','tab:green'],'compare_exp4_5_6.png','MOEA/D, all features: Experiments #4, #5, #6')
