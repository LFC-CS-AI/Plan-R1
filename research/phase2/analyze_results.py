"""Make supplementary diagnostics, figures and tables from saved data only."""
import csv
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MPLCONFIGDIR','/tmp/phase2-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import dump_csv

LABELS={'deterministic':'deterministic','mc':'MC','ensemble':'ensemble',
 'binomial_union':'binomial union','paired_difference':'paired difference',
 'residual_calibration':'global residual','marginal_cp':'pooled CP',
 'simultaneous_cp':'simultaneous CP','worst_member':'worst member',
 'action_residual':'action residual (post-hoc control)', 'actionwise_cp':'actionwise CP (post-hoc control)'}


def read(path):return list(csv.DictReader(path.open()))


def main():
    p=Path(__file__).resolve().parent;out=p/'results';figdir=p/'figures';figdir.mkdir(exist_ok=True)
    rows=read(out/'decisions.csv')+read(out/'extension_decisions.csv')
    refs=read(out/'reference.csv')
    summary=read(out/'summary.csv')+read(out/'extension_summary.csv')
    risk={}
    for r in refs:
        key=(r['seed'],r['strength'],r['domain'],r['scene'])
        risk.setdefault(key,{})[int(r['action'])]=r
    stats={}; failures={}; example=None
    for r in rows:
        # Extension files normalize numeric fields to floats; compare normalized keys.
        key=(str(int(float(r['seed']))),r['strength'],r['domain'],str(int(float(r['scene']))))
        ref=risk[key]
        pred=np.array([float(r['pred_risk_'+str(a)]) for a in range(5)])
        total,ties,inversions=0,0,0
        for a in range(1,5):
            for b in range(a+1,5):
                ra,rb=ref[a],ref[b]
                true_diff=float(ra['risk'])-float(rb['risk'])
                definite=(float(ra['risk_lo'])>float(rb['risk_hi'])+.05 or
                          float(rb['risk_lo'])>float(ra['risk_hi'])+.05)
                if definite:
                    total+=1; d=pred[a]-pred[b]
                    ties+=int(abs(d)<1e-12);inversions+=int(d*true_diff<0)
        skey=(r['domain'],float(r['strength']),int(float(r['budget'])),r['method'])
        acc=stats.setdefault(skey,[0,0,0]);acc[0]+=total;acc[1]+=ties;acc[2]+=inversions
        f=failures.setdefault(skey,[0,0,0]);chosen=int(float(r['chosen']))
        false_safe=int(float(r['false_safe']))
        safe=[a for a in range(1,5) if float(ref[a]['risk'])<=.1]
        implicated=[a for a in safe if pred[chosen]<pred[a] and
                    float(ref[chosen]['risk_lo'])>float(ref[a]['risk_hi'])+.05]
        if false_safe:
            f[0]+=1;f[1]+=int(bool(safe));f[2]+=int(bool(implicated))
        if example is None and r['method']=='deterministic' and r['domain']=='id' and float(r['strength'])==1.3 and int(r['budget'])==1280 and false_safe and implicated:
            example={'rule':'first ID strength=1.3 B=1280 deterministic false-safe with a reference-safe nonfallback alternative and strict selected-action risk inversion',
                     'safer_alternatives_with_inversion':implicated,'decision':r,'reference':[ref[a] for a in range(5)]}
    split=[]
    for key,v in sorted(stats.items()):
        den,ties,inv=v
        split.append(dict(domain=key[0],strength=key[1],budget=key[2],method=key[3],pairs=den,
            ties=ties,inversions=inv,tie_fraction=ties/den if den else np.nan,
            strict_inversion_fraction=inv/den if den else np.nan,
            tie_aware_error=(inv+.5*ties)/den if den else np.nan))
    dump_csv(out/'ranking_decomposition.csv',split)
    dump_csv(out/'selection_failure_modes.csv',[
        dict(domain=k[0],strength=k[1],budget=k[2],method=k[3],false_acceptances=v[0],
             with_safe_nonfallback_alternative=v[1],selected_inversion_against_safe_alternative=v[2],
             no_safe_nonfallback_alternative=v[0]-v[1]) for k,v in sorted(failures.items())])
    (out/'example.json').write_text(json.dumps(example,indent=2))
    slookup={(r['domain'],float(r['strength']),int(float(r['budget'])),r['method']):r for r in summary}
    strengths=[0.,.65,1.3];budgets=[80,320,1280]
    display=['deterministic','mc','ensemble','binomial_union','paired_difference','residual_calibration','simultaneous_cp','action_residual']
    colors={m:plt.get_cmap('tab10')(i) for i,m in enumerate(display)}
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row,domain in enumerate(['id','driver_shift']):
        for col,strength in enumerate(strengths):
            ax=axes[row,col]
            for m in ['deterministic','mc','ensemble','action_residual']:
                rr=[slookup[(domain,strength,b,m)] for b in budgets]
                ax.plot(budgets,[float(x['ranking_error']) for x in rr],marker='o',color=colors[m],label=LABELS[m])
            ax.set(xscale='log',xlabel='Candidate prediction budget',ylabel='Tie-aware pair ranking error',
                   title=f'{domain}, reaction={strength}',ylim=(-.02,.55))
    axes[0,0].legend(fontsize=7)
    fig.savefig(figdir/'ranking_by_budget.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row,domain in enumerate(['id','driver_shift']):
        for col,strength in enumerate(strengths):
            ax=axes[row,col]
            for m in display:
                r=slookup[(domain,strength,1280,m)]
                ax.scatter(float(r['progress_relative_to_oracle']),float(r['expected_collision']),color=colors[m],label=LABELS[m],s=45)
            ax.set(xlabel='Expected progress / risk-constrained oracle',ylabel='Expected collision probability',
                title=f'{domain}, reaction={strength}',ylim=(-.008,.34))
            ax.axhline(.1,color='gray',ls=':',lw=1);ax.axvline(.8,color='gray',ls=':',lw=1)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.savefig(figdir/'safety_progress.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    keys=['false_safe_conditional','progress_relative_to_oracle','rejection_rate']
    for row,domain in enumerate(['id','driver_shift']):
        for col,key in enumerate(keys):
            ax=axes[row,col]
            for m in display:
                rr=[slookup[(domain,1.3,b,m)] for b in budgets]
                val=np.array([float(x[key]) for x in rr])
                ax.plot(budgets,val,marker='o',color=colors[m],label=LABELS[m])
            ax.set(xscale='log',xlabel='Candidate prediction budget',ylabel=key,title=domain+', reaction=1.3')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.savefig(figdir/'selection_reliability.png',dpi=160);plt.close(fig)
    if example:
        rr=example['reference'];d=example['decision'];x=np.arange(5);width=.36
        pred=np.array([float(d['pred_risk_'+str(a)]) for a in range(5)])
        true=np.array([float(r['risk']) for r in rr]);lo=np.array([float(r['risk_lo']) for r in rr]);hi=np.array([float(r['risk_hi']) for r in rr])
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        ax.bar(x-width/2,pred,width,label='Predicted '+d['method']+' risk')
        ax.bar(x+width/2,true,width,yerr=np.stack([true-lo,hi-true]),label='Independent reference (99% MC CI)')
        ax.axhline(.1,color='gray',ls='--');ax.set(xticks=x,xticklabels=['stop','slow','coast','go','assert'],ylabel='Collision probability',
            title=f"First qualifying example: seed={d['seed']}, scene={d['scene']}, selected={d['chosen']}")
        ax.legend();fig.savefig(figdir/'ranking_example.png',dpi=160);plt.close(fig)
    # Save one full machine-readable table and a compact human-facing table.
    dump_csv(out/'all_methods_summary.csv',summary)
    text='| Domain | Reaction | Method | Rank error | False-safe / accepted | Observed collision | Expected collision | Progress (m) | Progress/oracle | Fallback | Online calls | CPU seconds |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for domain in ['id','driver_shift']:
        for strength in strengths:
            for m in LABELS:
                r=slookup[(domain,strength,1280,m)]
                def fmt(k):
                    v=float(r[k]);return 'NA' if not np.isfinite(v) else f'{v:.4f}'
                text+='| '+domain+' | '+str(strength)+' | '+m+' | '+' | '.join(fmt(k) for k in ['ranking_error','false_safe_conditional','collision_rate','expected_collision','progress_m','progress_relative_to_oracle','rejection_rate'])+' | '+r['model_calls']+' | '+fmt('cpu_seconds')+' |\n'
    (p/'full_budget_table.md').write_text('# B=1280：全部方法与全部环境\n\n'+text+'\n其他预算及置信区间见 `results/all_methods_summary.csv`。每行192个独立场景实例（4个种子，每种子48个场景）；CPU秒数为该行所有场景合计的实际预测/筛选耗时，不包含参考评估和校准。NA表示没有接受动作，条件错误率未定义，不能解释成零。\n')
    print('Generated figures, ranking tie/inversion decomposition and complete tables.')


if __name__=='__main__':main()
