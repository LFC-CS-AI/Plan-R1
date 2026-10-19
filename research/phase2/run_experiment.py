#!/usr/bin/env python3
"""CPU phase-two mechanism study. No Plan-R1 weights or first-stage reruns.

Run with Python >=3.9, numpy, scipy, matplotlib (plot_results.py).
Full: python run_experiment.py --out results
Smoke: python run_experiment.py --quick --out /workspace/work/phase2/smoke
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
import scipy
from reference_simulator import evaluate, make_scenes, DOMAINS
from world_model import fit_world
from baselines import fit_calibration, estimate_and_choose, binomial_interval


def dump_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def reference_batches(scenes,strength,domain,n,seed,config):
    parts=[]
    for j in range(0,len(scenes),8):
        parts.append(evaluate(scenes[j:j+8],strength,domain,n,seed+j,
                              config['dt'],config['horizon_seconds']))
    return {k:(sum(x[k] for x in parts) if k=='calls' else
               n if k=='latent_samples' else np.concatenate([x[k] for x in parts])) for k in parts[0]}


def ranking_stats(pred,truth,lo,hi,margin):
    S=len(pred);wrong=np.zeros(S);pairs=np.zeros(S)
    for a in range(1,5):
        for b in range(a+1,5):
            actual=truth[:,a]-truth[:,b]
            definite=(lo[:,a]>hi[:,b]+margin)|(lo[:,b]>hi[:,a]+margin)
            estimate=pred[:,a]-pred[:,b]
            wrong+=definite*np.where(np.abs(estimate)<1e-12,.5,(estimate*actual<0).astype(float))
            pairs+=definite
    return wrong,pairs


def summarize(rows,bootstrap=400):
    # Cluster bootstrap over experiment seeds, then scenes, retaining each
    # scene's coupled decision/reference outcomes. Fixed RNG for reporting.
    out=[]
    keys=sorted(set((r['domain'],r['strength'],r['budget'],r['method']) for r in rows))
    metric_names=['ranking_error','false_safe_unconditional','false_safe_conditional',
                  'definite_false_safe_conditional','collision_rate','expected_collision',
                  'progress_m','expected_progress_m','progress_relative_to_oracle',
                  'rejection_rate','harmful_switch_rate','elite_regret','joint_upper_coverage']
    def calc(group):
        accepted=np.array([r['accepted'] for r in group],float)
        false=np.array([r['false_safe'] for r in group],float)
        definite=np.array([r['definite_false_safe'] for r in group],float)
        den=sum(r['rank_pairs'] for r in group)
        return dict(ranking_error=sum(r['rank_wrong'] for r in group)/den if den else float('nan'),
            false_safe_unconditional=false.mean(),
            false_safe_conditional=false.sum()/accepted.sum() if accepted.sum() else float('nan'),
            definite_false_safe_conditional=definite.sum()/accepted.sum() if accepted.sum() else float('nan'),
            collision_rate=np.mean([r['collision'] for r in group]),
            expected_collision=np.mean([r['true_selected_risk'] for r in group]),
            progress_m=np.mean([r['progress_m'] for r in group]),
            expected_progress_m=np.mean([r['expected_progress_m'] for r in group]),
            progress_relative_to_oracle=np.mean([r['expected_progress_m'] for r in group])/np.mean([r['oracle_progress_m'] for r in group]),
            rejection_rate=1-accepted.mean(),
            harmful_switch_rate=np.mean([r['harmful_switch'] for r in group]),
            elite_regret=np.mean([r['elite_regret'] for r in group]),
            joint_upper_coverage=np.mean([r['joint_upper_covered'] for r in group]))
    for domain,strength,budget,method in keys:
        g=[r for r in rows if (r['domain'],r['strength'],r['budget'],r['method'])==(domain,strength,budget,method)]
        means=calc(g); rng=np.random.default_rng(877)
        seeds=sorted(set(r['seed'] for r in g)); byseed={s:[r for r in g if r['seed']==s] for s in seeds}
        draws=[]
        for _ in range(bootstrap):
            sample=[]
            for s in rng.choice(seeds,len(seeds),replace=True):
                v=byseed[s]; sample += [v[i] for i in rng.integers(0,len(v),len(v))]
            draws.append(calc(sample))
        item=dict(domain=domain,strength=strength,budget=budget,method=method,scenes=len(g),
                  accepted=sum(r['accepted'] for r in g),rank_pairs=sum(r['rank_pairs'] for r in g),
                  model_calls=sum(r['model_calls'] for r in g),cpu_seconds=sum(r['cpu_seconds'] for r in g))
        for k in metric_names:
            vals=np.array([d[k] for d in draws]); valid=vals[np.isfinite(vals)]
            item[k]=means[k]
            item[k+'_lo']=float(np.quantile(valid,.025)) if len(valid) else float('nan')
            item[k+'_hi']=float(np.quantile(valid,.975)) if len(valid) else float('nan')
        # A zero-event bootstrap interval alone can be misleading; supply an
        # exact-binomial event bound as an extra descriptive bound (iid scenes
        # assumption; shared calibration seed can still induce dependence).
        n=item['accepted']; k=sum(r['false_safe'] for r in g)
        item['fsa_binomial_upper95']=float(binomial_interval(k,n,.05)[1]) if n else float('nan')
        out.append(item)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results')
    ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    here=Path(__file__).resolve().parent
    config=json.loads((here/'protocol.json').read_text())
    if args.quick:
        config.update(seeds=[101],reaction_strengths=[0.,1.3],train_scenes=16,
            calibration_scenes=16,test_scenes=8,train_rollouts_per_action=64,
            calibration_rollouts_per_action=64,reference_rollouts_per_action=256,budgets=[80])
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); decisions=[]; refs=[]; ledger=[]; models=[]; calibrations=[]
    for seed in config['seeds']:
        train=make_scenes(seed*100+1,config['train_scenes'])
        cal_scenes=make_scenes(seed*100+2,config['calibration_scenes'])
        test=make_scenes(seed*100+3,config['test_scenes'])
        assert not any(np.array_equal(train[i],cal_scenes[j]) for i in range(len(train)) for j in range(len(cal_scenes)))
        for strength in config['reaction_strengths']:
            tick=time.perf_counter()
            train_ref=reference_batches(train,strength,'id',config['train_rollouts_per_action'],seed*10000+10,config)
            model=fit_world(train,train_ref,seed+10)
            ledger.append(dict(seed=seed,strength=strength,domain='id',phase='model_fit_reference',
                               reference_calls=train_ref['calls'],model_calls=0,cpu_seconds=time.perf_counter()-tick))
            models.append(dict(seed=seed,strength=strength,model=model))
            tick=time.perf_counter()
            cal_ref=reference_batches(cal_scenes,strength,'id',config['calibration_rollouts_per_action'],seed*10000+20,config)
            ledger.append(dict(seed=seed,strength=strength,domain='id',phase='calibration_reference',
                               reference_calls=cal_ref['calls'],model_calls=0,cpu_seconds=time.perf_counter()-tick))
            cals={}
            for budget in config['budgets']:
                tick=time.perf_counter()
                cals[budget]=fit_calibration(cal_scenes,cal_ref,model,budget,seed*100+40+budget,config['alpha'])
                ledger.append(dict(seed=seed,strength=strength,domain='id',phase='calibration_predictions_B'+str(budget),
                    reference_calls=0,model_calls=cals[budget]['offline_model_calls'],cpu_seconds=time.perf_counter()-tick))
                calibrations.append(dict(seed=seed,strength=strength,budget=budget,**cals[budget]))
            for domain in config['domains']:
                tick=time.perf_counter()
                truth=reference_batches(test,strength,domain,config['reference_rollouts_per_action'],seed*10000+30,config)
                dep=reference_batches(test,strength,domain,1,seed*10000+31,config)
                lo,hi=binomial_interval(truth['counts'],truth['latent_samples'],config['reference_interval_alpha'])
                ledger.append(dict(seed=seed,strength=strength,domain=domain,phase='test_reference_and_deployment_bank',
                    reference_calls=truth['calls']+dep['calls'],model_calls=0,cpu_seconds=time.perf_counter()-tick))
                oracle=np.max(np.where(truth['risk']<=config['risk_threshold'],np.arange(5)[None,:],0),axis=1)
                for i in range(len(test)):
                    for a in range(5):
                        refs.append(dict(seed=seed,strength=strength,domain=domain,scene=i,action=a,
                            ego_distance=test[i,0],ego_speed=test[i,1],other_distance=test[i,2],other_speed=test[i,3],
                            collision_count=int(truth['counts'][i,a]),samples=truth['latent_samples'],risk=truth['risk'][i,a],
                            risk_lo=lo[i,a],risk_hi=hi[i,a],progress_m=truth['progress'][i,a],yield_rate=truth['yield_rate'][i,a],
                            other_arrival=truth['other_arrival'][i,a],deployment_collision=int(dep['counts'][i,a]),
                            deployment_progress_m=dep['progress'][i,a]))
                for budget in config['budgets']:
                    for method in config['methods']:
                        tick=time.perf_counter()
                        pred,upper,chosen,calls=estimate_and_choose(method,test,model,budget,
                            seed*100+50+budget,cals[budget],config['alpha'],config['risk_threshold'],config['relative_risk_tolerance'])
                        elapsed=time.perf_counter()-tick
                        assert calls<=len(test)*budget
                        wrong,pairs=ranking_stats(pred,truth['risk'],lo,hi,config['ranking_margin'])
                        # AD-WM-style best-in-elite diagnostic, specialized to
                        # risk costs and top-1 from the nonfallback action bank.
                        elite=np.argmax((pred[:,1:]==pred[:,1:].min(1,keepdims=True))*np.arange(1,5),axis=1)+1
                        for i,a in enumerate(chosen):
                            accept=int(a!=0); q=truth['risk'][i,a]
                            row=dict(seed=seed,strength=strength,domain=domain,budget=budget,method=method,scene=i,
                                chosen=int(a),accepted=accept,rank_wrong=wrong[i],rank_pairs=pairs[i],
                                false_safe=int(accept and q>config['risk_threshold']),
                                definite_false_safe=int(accept and lo[i,a]>config['risk_threshold']),
                                collision=int(dep['counts'][i,a]),true_selected_risk=q,
                                predicted_selected_risk=pred[i,a],upper_selected_risk=upper[i,a],
                                progress_m=dep['progress'][i,a],expected_progress_m=truth['progress'][i,a],
                                oracle_action=int(oracle[i]),oracle_progress_m=truth['progress'][i,oracle[i]],
                                harmful_switch=int(a!=2 and q>truth['risk'][i,2]+config['ranking_margin']),
                                elite_regret=(truth['risk'][i,elite[i]]-truth['risk'][i,1:].min())/max(np.ptp(truth['risk'][i,1:]),1e-8),
                                joint_upper_covered=int(np.all(truth['risk'][i,1:]<=upper[i,1:]+1e-12)),
                                threshold_ambiguous=int(lo[i,a]<=config['risk_threshold']<=hi[i,a]),
                                model_calls=calls//len(test),cpu_seconds=elapsed/len(test))
                            for j in range(5):row['pred_risk_'+str(j)]=pred[i,j]
                            decisions.append(row)
                        ledger.append(dict(seed=seed,strength=strength,domain=domain,phase=method+'_B'+str(budget),
                            reference_calls=0,model_calls=calls,cpu_seconds=elapsed))
                print(f'completed seed={seed} strength={strength} domain={domain}; elapsed={time.perf_counter()-start:.1f}s',flush=True)
                # Checkpoint raw observations; safe to resume diagnosis if host fails.
                dump_csv(args.out/'decisions.csv',decisions)
                dump_csv(args.out/'reference.csv',refs)
                dump_csv(args.out/'compute_ledger.csv',ledger)
    tick=time.perf_counter()
    summary=summarize(decisions,bootstrap=100 if args.quick else 400)
    dump_csv(args.out/'summary.csv',summary)
    (args.out/'models.json').write_text(json.dumps(models,indent=2))
    (args.out/'calibration.json').write_text(json.dumps(calibrations,indent=2))
    meta=dict(config=config,quick=args.quick,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
              platform=platform.platform(),elapsed_seconds=time.perf_counter()-start,
              summary_bootstrap_seconds=time.perf_counter()-tick,
              total_online_model_calls=sum(r['model_calls'] for r in decisions),
              total_offline_model_calls=sum(r['model_calls'] for r in ledger if r['phase'].startswith('calibration_predictions')),
              total_reference_calls=sum(r['reference_calls'] for r in ledger),
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in here.glob('*.py')},
              protocol_sha256=hashlib.sha256((here/'protocol.json').read_bytes()).hexdigest(),
              conventions=dict(cpu_seconds='perf_counter elapsed wall time for actual CPU-only NumPy/SciPy execution; selector time excludes reference/calibration; all components in ledger',
               reference='independent reactive evaluator, Monte Carlo estimates not exact physical truth',
               statistical_unit='scene within experiment seed; same scenes and exogenous deployment draws paired across methods',
               cpp_baselines='no full-paper implementation or paper-level guarantee claimed'))
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta[k] for k in ['elapsed_seconds','total_online_model_calls','total_offline_model_calls','total_reference_calls']},indent=2),flush=True)


if __name__=='__main__':main()
