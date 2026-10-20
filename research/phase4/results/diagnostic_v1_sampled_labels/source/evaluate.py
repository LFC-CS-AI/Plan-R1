"""Independent held-out evaluation; all learned trajectories evaluated RAW.

Rejection is abstention only, not an executable braking policy. 'Collision'
means geometric overlap along the proposed open-loop 1s motion.
"""
import csv,json,time
from pathlib import Path
import numpy as np
import torch
from bridge_dynamics import path,pose
from geometry import sampled,reference
from train_heads import load_head,coefficient

HERE=Path(__file__).resolve().parent


def decision(method,scene,b,x,model=None):
    coeff=np.asarray(b['coeff']);queries=0;calls=0;family='residual' if method=='residual_head' else 'bridge';accept=True
    if model is not None:
        # Projected cubic residual has exactly the bridge function class after
        # preserving the .5/1 knots; same coefficients expose the identity.
        decoder='bridge_head' if method=='projected_residual' else method
        c=float(coefficient(model,x,decoder)[0])*b['cap_'+family];calls=1
    elif method.startswith('rerank'):
        hz=int(method.split('_')[1]);candidates=np.linspace(-1,1,9)*b['cap_bridge']
        margins=np.array([sampled(coeff,c,scene,hz=hz) for c in candidates]);queries=len(candidates)*(hz+1)
        c=float(candidates[np.argmax(margins)]);accept=bool(margins.max()>0)
    else:
        c=0.
        if method.startswith('filter'):
            hz=int(method.split('_')[1]);accept=sampled(coeff,c,scene,hz=hz)>0;queries=hz+1
    trajectory=pose(coeff,c,np.linspace(0,1,51),family)
    return dict(c=c,family=family,accepted=bool(accept),online_pose_queries=queries,head_calls=calls,progress=float(trajectory[-1,0]))


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run():
    torch.set_num_threads(1);r=HERE/'results';p=json.loads((HERE/'protocol.json').read_text());bases=json.loads((r/'bases.json').read_text())['bases'];scenes=json.loads((r/'test_scenes.json').read_text());data=np.load(r/'test_data.npz');X=data['features'];rows=[];summaries=[];cache={};start=time.perf_counter()
    specs=[(m,0,None) for m in ['fixed_bridge','filter_10','filter_50','rerank_10','rerank_50']]
    for seed in p['training_seeds']:
        for method in ['reward_selector','residual_head','bridge_head','projected_residual']:
            filename='bridge_head' if method=='projected_residual' else method
            specs.append((method,seed,load_head(r/'heads'/f'{filename}_{seed}.npz')))
    for method,seed,model in specs:
        wall=time.perf_counter();local=[];times=[]
        for j,s in enumerate(scenes):
            b=bases[s['base_index']];coeff=np.asarray(b['coeff']);x=X[j]
            tick=time.perf_counter();d=decision(method,s,b,x,model);times.append((time.perf_counter()-tick)*1000)
            key=(j,d['family'],d['c'])
            if key not in cache:cache[key]=reference(coeff,d['c'],s,d['family'])
            ref=cache[key];actual=pose(coeff,d['c'],np.array([.5,1.]),d['family']);knots=np.array(b['state']['knots'])[1:];delta=np.linalg.norm(actual[:,:2]-knots[:,:2],axis=-1).max();yawerr=np.abs((actual[:,2]-knots[:,2]+np.pi)%(2*np.pi)-np.pi).max()
            row=dict(method=method,seed=seed,scene_id=s['scene_id'],population=s['population'],token_id=b['token_id'],coefficient=d['c'],family=d['family'],accepted=int(d['accepted']),reference_status=ref['status'],collision=int(ref['status']=='collision'),unknown=int(ref['status']=='unknown'),progress_m=d['progress'],accepted_progress_proxy_m=d['progress']*d['accepted'],endpoint_position_error_m=float(delta),endpoint_yaw_error_rad=float(yawerr),token_compatible=int(delta<1e-6 and yawerr<1e-6),dynamics_feasible=1,online_pose_queries=d['online_pose_queries'],head_calls=d['head_calls'],world_model_calls=0,reference_queries=ref['queries'],inference_ms=times[-1]);rows.append(row);local.append(row)
        # Warming and timing exclude evaluator/labels, include 50Hz output
        # materialization, identical single CPU thread and batch size 1.
        sample=scenes[1];b=bases[sample['base_index']]
        for _ in range(20):decision(method,sample,b,X[1],model)
        lat=[]
        for _ in range(100):
            tick=time.perf_counter();decision(method,sample,b,X[1],model);lat.append((time.perf_counter()-tick)*1000)
        for population in ['all','broad','boundary']:
            selected=[z for z in local if population=='all' or z['population']==population];n=len(selected);accepted=sum(z['accepted'] for z in selected);coll=sum(z['collision']*z['accepted'] for z in selected)
            summary=dict(method=method,seed=seed,population=population,n=n,collision_rate_all_proposals=float(np.mean([z['collision'] for z in selected])),accepted_collision_rate=coll/max(accepted,1),collision_per_scene=coll/n,unknown_rate=float(np.mean([z['unknown'] for z in selected])),rejection_rate=1-accepted/n,mean_progress_m=float(np.mean([z['progress_m'] for z in selected])),accepted_progress_proxy_m=float(np.mean([z['accepted_progress_proxy_m'] for z in selected])),dynamics_feasible_rate=1.,token_compatible_rate=float(np.mean([z['token_compatible'] for z in selected])),mean_endpoint_error_m=float(np.mean([z['endpoint_position_error_m'] for z in selected])),parameters=0 if model is None else sum(v.numel() for v in model.parameters()),head_calls=sum(z['head_calls'] for z in selected),world_model_calls=0,online_pose_queries=sum(z['online_pose_queries'] for z in selected),reference_queries=sum(z['reference_queries'] for z in selected),warm_p50_ms=float(np.median(lat)),warm_p95_ms=float(np.percentile(lat,95)),scene_inference_p50_ms=float(np.median([z['inference_ms'] for z in selected])),evaluation_wall_seconds=time.perf_counter()-wall)
            summaries.append(summary)
        print(method,seed,'all collision',summaries[-3]['collision_rate_all_proposals'],'reject',summaries[-3]['rejection_rate'],flush=True)
    write_csv(r/'test_predictions.csv',rows);write_csv(r/'test_metrics.csv',summaries)
    # Feasibility oracle chooses from the shared bank only. A failed grid is
    # not an impossibility proof for general trajectories or even this family.
    feasible=[]
    for j,s in enumerate(scenes):
        b=bases[s['base_index']];coeff=np.asarray(b['coeff']);k=int(data['margins'][j,0].argmax());c=float(data['xgrid'][k]*b['cap_bridge']);base_ref=reference(coeff,0.,s);best_ref=reference(coeff,c,s)
        row=dict(scene_id=s['scene_id'],population=s['population'],token_id=b['token_id'],base_status=base_ref['status'],grid_best_c=c,grid_best_status=best_ref['status'],base_sample_gap=float(data['margins'][j,0,4]),grid_sample_gap=float(data['margins'][j,0,k]),max_bridge_displacement_m=b['max_bridge_displacement_m'])
        for buffer in p['feasibility_buffers_m']:
            base=reference(coeff,0.,s,buffer=buffer);best=reference(coeff,c,s,buffer=buffer);tag=str(buffer)
            row['base_'+tag]=base['status'];row['best_'+tag]=best['status'];row['rescue_'+tag]=int(base['status']=='collision' and best['status']=='safe')
        feasible.append(row)
    write_csv(r/'feasibility.csv',feasible)
    failures=[z for z in rows if z['method']=='bridge_head' and z['seed']==p['training_seeds'][0] and (z['collision'] or z['unknown'])]
    (r/'failure_cases.json').write_text(json.dumps(failures,indent=2))
    (r/'evaluation_ledger.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,predictions=len(rows),distinct_reference_trajectories=len(cache),feasibility_scenes=len(feasible),note='No pretrained or world-model calls. Filtering abstains; stopping safety is unmodeled. All raw learned rows use no online collision queries.'),indent=2))

if __name__=='__main__':run()
