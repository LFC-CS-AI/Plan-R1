"""Post-main standard controls; original protocol and raw files remain intact."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from reference_simulator import make_scenes
from world_model import predict
from baselines import binomial_interval,conformal_quantile
from run_experiment import reference_batches,dump_csv,summarize


def main():
    root=Path(__file__).resolve().parent; out=root/'results'
    cfg=json.loads((root/'protocol.json').read_text())
    models=json.loads((out/'models.json').read_text())
    original=list(csv.DictReader((out/'decisions.csv').open()))
    rows=[]; ledger=[]; parameters=[]; start=time.perf_counter()
    for fit in models:
        seed,strength,model=fit['seed'],fit['strength'],fit['model']
        scenes=make_scenes(seed*100+2,cfg['calibration_scenes'])
        test=make_scenes(seed*100+3,cfg['test_scenes'])
        tick=time.perf_counter()
        ref=reference_batches(scenes,strength,'id',cfg['calibration_rollouts_per_action'],seed*10000+20,cfg)
        _,hi=binomial_interval(ref['counts'],ref['latent_samples'],.01/(len(scenes)*5))
        ledger.append(dict(seed=seed,strength=strength,domain='id',phase='extension_calibration_reference_replay',
            reference_calls=ref['calls'],model_calls=0,cpu_seconds=time.perf_counter()-tick))
        for budget in cfg['budgets']:
            tick=time.perf_counter()
            ev,calls=predict(scenes,model,budget//5,seed*100+40+budget,ensemble=True)
            mean=ev.mean(-1)
            offsets=(ref['risk']-mean).mean(0)
            quantiles=np.array([max(0.,conformal_quantile(hi[:,a]-mean[:,a],cfg['alpha'])) for a in range(5)])
            parameters.append(dict(seed=seed,strength=strength,budget=budget,offsets=offsets.tolist(),quantiles=quantiles.tolist()))
            ledger.append(dict(seed=seed,strength=strength,domain='id',phase='extension_calibration_predictions_B'+str(budget),
                reference_calls=0,model_calls=calls,cpu_seconds=time.perf_counter()-tick))
            for domain in cfg['domains']:
                templates=[r for r in original if int(r['seed'])==seed and float(r['strength'])==strength and r['domain']==domain and int(r['budget'])==budget and r['method']=='ensemble']
                for method in ['action_residual','actionwise_cp']:
                    tick=time.perf_counter()
                    ev,calls=predict(test,model,budget//5,seed*100+50+budget,ensemble=True)
                    pred=ev.mean(-1)
                    upper=np.clip(pred+(offsets if method=='action_residual' else quantiles),0.,1.)
                    # Residual calibration changes rankings; quantile screening doesn't.
                    if method=='action_residual':pred=upper.copy()
                    eligible=upper<=cfg['risk_threshold'];eligible[:,0]=True
                    chosen=np.max(np.where(eligible,np.arange(5)[None,:],0),axis=1)
                    elapsed=time.perf_counter()-tick
                    reference_rows=list(csv.DictReader((out/'reference.csv').open())) if not rows else reference_rows
                    block=[r for r in reference_rows if int(r['seed'])==seed and float(r['strength'])==strength and r['domain']==domain]
                    risk=np.array([float(r['risk']) for r in block]).reshape(-1,5)
                    low=np.array([float(r['risk_lo']) for r in block]).reshape(-1,5)
                    high=np.array([float(r['risk_hi']) for r in block]).reshape(-1,5)
                    from run_experiment import ranking_stats
                    wrong,pairs=ranking_stats(pred,risk,low,high,cfg['ranking_margin'])
                    elite=np.argmax((pred[:,1:]==pred[:,1:].min(1,keepdims=True))*np.arange(1,5),axis=1)+1
                    for i,a in enumerate(chosen):
                        base=dict(templates[i]); rr=block[i*5+a]; accept=int(a!=0)
                        base.update(method=method,chosen=int(a),accepted=accept,rank_wrong=wrong[i],rank_pairs=pairs[i],
                            false_safe=int(accept and risk[i,a]>cfg['risk_threshold']),
                            definite_false_safe=int(accept and low[i,a]>cfg['risk_threshold']),
                            collision=int(rr['deployment_collision']),true_selected_risk=risk[i,a],
                            predicted_selected_risk=pred[i,a],upper_selected_risk=upper[i,a],
                            progress_m=float(rr['deployment_progress_m']),expected_progress_m=float(rr['progress_m']),
                            harmful_switch=int(a!=2 and risk[i,a]>risk[i,2]+cfg['ranking_margin']),
                            elite_regret=(risk[i,elite[i]]-risk[i,1:].min())/max(np.ptp(risk[i,1:]),1e-8),
                            joint_upper_covered=int(np.all(risk[i,1:]<=upper[i,1:]+1e-12)),
                            threshold_ambiguous=int(low[i,a]<=cfg['risk_threshold']<=high[i,a]),
                            model_calls=calls//len(test),cpu_seconds=elapsed/len(test))
                        for j in range(5):base['pred_risk_'+str(j)]=pred[i,j]
                        rows.append(base)
                    ledger.append(dict(seed=seed,strength=strength,domain=domain,phase=method+'_B'+str(budget),
                        reference_calls=0,model_calls=calls,cpu_seconds=elapsed))
        print('completed extension',seed,strength,flush=True)
    # Convert copied numeric strings before shared aggregation.
    numeric=set(original[0])-{'method','domain'}
    for r in rows:
        for k in numeric:r[k]=float(r[k])
    dump_csv(out/'extension_decisions.csv',rows)
    dump_csv(out/'extension_ledger.csv',ledger)
    dump_csv(out/'extension_summary.csv',summarize(rows))
    (out/'extension_parameters.json').write_text(json.dumps(parameters,indent=2))
    (out/'extension_metadata.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-start,
        additional_online_model_calls=sum(r['model_calls'] for r in rows),
        additional_offline_model_calls=sum(r['model_calls'] for r in ledger if r['phase'].startswith('extension_calibration_predictions')),
        additional_reference_calls=sum(r['reference_calls'] for r in ledger),
        adaptation='post-main-inspection ordinary controls; no original outputs overwritten'),indent=2))


if __name__=='__main__':main()
