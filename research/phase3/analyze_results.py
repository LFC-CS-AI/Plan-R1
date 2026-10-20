"""Saved-data statistics and plots; no world/model calls or new simulations."""
import collections
import csv
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MPLCONFIGDIR','/tmp/phase3-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motion_geometry import bridge,sat_gap,rectangle
from run_diagnostics import write_csv


def main():
    root=Path(__file__).parent;out=root/'results';fig=root/'figures';fig.mkdir(exist_ok=True)
    rows=list(csv.DictReader((out/'diagnostics.csv').open()));candidates=list(csv.DictReader((out/'candidate_bank.csv').open()))
    ledger=list(csv.DictReader((out/'compute_ledger.csv').open()));groups=collections.defaultdict(list)
    for r in candidates:groups[(r['seed'],r['kind'],r['scene'])].append(r)
    summary=[]
    for kind in ['all','crossing_vehicle','opposing_vehicle','crossing_pedestrian']:
        rr=[r for r in rows if r['initially_clear']=='1' and (kind=='all' or r['kind']==kind)]
        collision=[r for r in rr if r['ccd_status']=='collision']
        for m,col in [('native','native_collision')]+[(str(dt)+'s','sat_collision_dt_'+str(dt)) for dt in [.5,.1,.02,.005]]:
            miss=sum(r[col]=='0' for r in collision);det=sum(r[col]=='1' for r in rr)
            summary.append(dict(kind=kind,method=m,initially_clear=len(rr),interval_collisions=len(collision),
                missed_collisions=miss,miss_per_clear_scene=miss/len(rr),miss_per_interval_collision=miss/len(collision),
                false_positive_vs_nominal=sum(r[col]=='1' and r['ccd_status']=='safe' for r in rr),detected=det))
    write_csv(out/'detection_summary.csv',summary)
    seedrows=[]
    for seed in sorted(set(r['seed'] for r in rows)):
        rr=[r for r in rows if r['seed']==seed and r['initially_clear']=='1']
        seedrows.append(dict(seed=seed,initially_clear=len(rr),interval_collisions=sum(r['ccd_status']=='collision' for r in rr),
            temporal_misses=sum(r['ccd_status']=='collision' and r['sat_collision_dt_0.5']=='0' for r in rr)))
    write_csv(out/'seed_summary.csv',seedrows)
    cpu=collections.defaultdict(lambda:dict(seconds=0.,geometry_queries=0))
    for r in ledger:cpu[r['operation']]['seconds']+=float(r['seconds']);cpu[r['operation']]['geometry_queries']+=int(r['geometry_queries'])
    write_csv(out/'compute_summary.csv',[dict(operation=k,**v) for k,v in cpu.items()])
    selections=[];metrics=[]
    for flag in ['native_endpoint','sat_endpoint','sat_10hz','sat_50hz','ccd','tube']:
        for mode in ['filter','rerank']:
            decisions=[]
            for key,g in groups.items():
                eligible=[r for r in g if r['reject_'+flag]=='0' and (mode=='rerank' or r['top_preference']=='1')]
                winner=max(eligible,key=lambda r:float(r['preference'])) if eligible else None
                row=dict(seed=key[0],kind=key[1],scene=key[2],method=flag+'_'+mode,
                    candidate=int(winner['candidate']) if winner else -1,
                    accepted=int(winner is not None),collision=int(winner is not None and winner['reference_status']=='collision'),
                    progress=float(winner['progress']) if winner else 0.,unknown=int(winner is not None and winner['reference_status']=='unknown'))
                decisions.append(row);selections.append(row)
            cost=cpu['candidate_'+flag];accepted=sum(r['accepted'] for r in decisions);coll=sum(r['collision'] for r in decisions)
            metrics.append(dict(method=flag+'_'+mode,scenes=len(decisions),accepted=accepted,collisions=coll,
                collision_per_scene=coll/len(decisions),collision_per_accepted=coll/accepted if accepted else np.nan,
                rejection_rate=1-accepted/len(decisions),mean_selected_displacement_m=np.mean([r['progress'] for r in decisions]),
                actual_bank_evaluation_seconds=cost['seconds'],geometry_queries=cost['geometry_queries'],world_model_calls=0))
    write_csv(out/'selections.csv',selections);write_csv(out/'selection_summary.csv',metrics)
    fig1,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    base=[r for r in summary if r['kind']=='all']
    axes[0].bar([r['method'] for r in base],[r['missed_collisions'] for r in base]);axes[0].set(ylabel='Missed interval collisions',title='Initially-clear synthetic token intervals')
    labels=['10Hz filter','10Hz rerank','CCD rerank','tube rerank'];chosen=['sat_10hz_filter','sat_10hz_rerank','ccd_rerank','tube_rerank']
    vals=[next(r for r in metrics if r['method']==m) for m in chosen]
    axes[1].bar(labels,[r['rejection_rate'] for r in vals]);axes[1].tick_params(axis='x',rotation=20);axes[1].set(ylabel='Abstention fraction',title='No action executed after abstention')
    fig1.savefig(fig/'detection_and_selection.png',dpi=160);plt.close(fig1)
    # Example chosen transparently: greatest dense penetration among pure temporal misses.
    leaks=[r for r in rows if r['initially_clear']=='1' and r['ccd_status']=='collision' and r['sat_collision_dt_0.5']=='0']
    def arrays(r):
        pose=lambda n:np.array([float(r[n+'_'+a]) for a in ['x','y','yaw']]);box=lambda n:np.array([float(r[n+'_'+a]) for a in ['front','rear','left','right']])
        return [pose(n) for n in ['ego_start','ego_end','other_start','other_end']]+[box(n) for n in ['ego','other']]
    t=np.linspace(0,1,1001)
    def trace(r,kind='linear'):
        a0,a1,b0,b1,ba,bb=arrays(r);return sat_gap(bridge(a0,a1,t,kind),bridge(b0,b1,t,kind),ba,bb)
    example=min(leaks,key=lambda r:trace(r).min());g=trace(example);gh=trace(example,'hermite')
    (out/'example.json').write_text(json.dumps(dict(rule='Largest sampled penetration among all pure temporal misses; illustration only, prevalence uses all cases.',case=example),indent=2))
    a0,a1,b0,b1,ba,bb=arrays(example);fig2,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    ax[0].plot(t*.5,g,label='linear SE2');ax[0].plot(t*.5,gh,ls='--',label='bounded Hermite');ax[0].scatter([0,.5],g[[0,-1]],c='black',label='token endpoints');ax[0].axhline(0,c='gray',ls=':');ax[0].set(xlabel='Time (s)',ylabel='Maximum SAT axis gap (m)',title='Pure temporal aliasing');ax[0].legend()
    for timefrac in [0.,.5,1.]:
        for name,start,end,box,color in [('ego',a0,a1,ba,'tab:blue'),('other',b0,b1,bb,'tab:orange')]:
            pose=bridge(start,end,timefrac);c,u,v,h=rectangle(pose,box);corners=np.array([c+i*h[0]*u+j*h[1]*v for i,j in [(1,1),(1,-1),(-1,-1),(-1,1),(1,1)]])
            ax[1].plot(corners[:,0],corners[:,1],c=color,alpha=.4+.5*timefrac,label=f'{name} t={timefrac*.5:g}')
    ax[1].set_aspect('equal');ax[1].set(xlabel='x (m)',ylabel='y (m)',title=f"seed={example['seed']}, scene={example['scene']}");ax[1].legend(fontsize=7)
    fig2.savefig(fig/'temporal_example.png',dpi=160);plt.close(fig2)
    fig3,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for m in ['linear_sat_dt_0.5','linear_sat_dt_0.1','linear_sat_dt_0.02','linear_sat_dt_0.005','linear_interval_ccd']:
        ax.bar(m.replace('linear_',''),cpu[m]['seconds'])
    ax.tick_params(axis='x',rotation=15);ax.set(ylabel='Actual CPU-only elapsed seconds (3072 intervals)',title='Implementation timing, excludes neural model inference')
    fig3.savefig(fig/'geometry_cost.png',dpi=160);plt.close(fig3)
    print('Saved detection, seed, selection, compute summaries and three figures; no simulation rerun.')


if __name__=='__main__':main()
