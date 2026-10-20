"""Round-one CPU token diagnostics. No checkpoint, model training or phase1/2 run."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from motion_geometry import decode,bounds,corner_overlap,sampled_gaps,interval_ccd,tube_certificate,sat_gap
from native_probe import native_rewards,REPO

EGO_BOX=np.array([3.95,1.05,1.,1.])  # 5m length, rear-axle offset 1.45m
VEH_BOX=np.array([2.2,2.2,.9,.9])
PED_BOX=np.array([.4,.4,.4,.4])


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_vocabulary():
    return {k:v.numpy().astype(float) for k,v in torch.load(REPO/'tokens/tokens_1024.pt',map_location='cpu',weights_only=True).items()}


def feasible_ids(tokens,box,cfg):
    ids=[];kin=cfg['kinematics'];bds=[]
    for i,t in enumerate(tokens):
        bd=bounds(np.zeros(3),decode(np.zeros(3),t),box,'hermite')
        good=t[0]>.15 and bd['speed']<=kin['max_speed'] and bd['acceleration']<=kin['max_acceleration'] and bd['yaw_rate']<=kin['max_yaw_rate'] and bd.get('lower_speed',0)>kin['min_forward_speed']
        bds.append(bd)
        if good:ids.append(i)
    return np.array(ids),bds


def make_block(seed,kind,count,vocab,valid):
    rng=np.random.default_rng(seed);who='Pedestrian' if kind=='crossing_pedestrian' else 'Vehicle'
    aid=rng.choice(valid['Vehicle'],count);bid=rng.choice(valid[who],count)
    a0=np.zeros((count,3));b0=np.zeros((count,3))
    b0[:,:2]=rng.uniform([-12.,-9.],[18.,9.],(count,2))
    low,high=(160.,200.) if kind=='opposing_vehicle' else (60.,120.)
    b0[:,2]=np.deg2rad(rng.uniform(low,high,count))
    a1=decode(a0,vocab['Vehicle'][aid]);b1=decode(b0,vocab[who][bid])
    ba=np.broadcast_to(EGO_BOX,(count,4)).copy();bb=np.broadcast_to(PED_BOX if who=='Pedestrian' else VEH_BOX,(count,4)).copy()
    return aid,bid,a0,a1,b0,b1,ba,bb


def main():
    root=Path(__file__).resolve().parent;out=root/'results';out.mkdir(exist_ok=True)
    cfg=json.loads((root/'protocol.json').read_text());start=time.perf_counter();ledger=[];rows=[];candidate_rows=[]
    vocab=load_vocabulary();valid={};bds={}
    for who,box in [('Vehicle',EGO_BOX),('Pedestrian',PED_BOX)]:valid[who],bds[who]=feasible_ids(vocab[who],box,cfg)
    assert all(len(v)>16 for v in valid.values())
    # Preserve every token's declared kinematic inclusion/exclusion, not just winners.
    token_rows=[]
    for who in valid:
        for i,t in enumerate(vocab[who]):token_rows.append(dict(type=who,token_id=i,dx=t[0],dy=t[1],dyaw=t[2],included=int(i in valid[who]),**bds[who][i]))
    write_csv(out/'vocabulary_kinematics.csv',token_rows)
    for seed in cfg['seeds']:
        for kind_index,kind in enumerate(cfg['kinds']):
            block_seed=seed*10+kind_index
            aid,bid,a0,a1,b0,b1,ba,bb=make_block(block_seed,kind,cfg['scenes_per_seed_per_kind'],vocab,valid)
            tic=time.perf_counter();native,ttc,nt=native_rewards(a0,a1,b0,b1,ba,bb)
            ledger.append(dict(seed=seed,kind=kind,operation='original_reward_and_ttc',seconds=nt,geometry_queries=0))
            endpoint_corner=corner_overlap(a1,b1,ba,bb)
            assert np.array_equal(native,endpoint_corner), 'Independent corner predicate differs from original implementation'
            gaps={};hgap={}
            for dt in cfg['sample_steps_seconds']:
                tic=time.perf_counter();gaps[dt]=sampled_gaps(a0,a1,b0,b1,ba,bb,dt)
                ledger.append(dict(seed=seed,kind=kind,operation='linear_sat_dt_'+str(dt),seconds=time.perf_counter()-tic,geometry_queries=gaps[dt].size))
            tic=time.perf_counter();hg=sampled_gaps(a0,a1,b0,b1,ba,bb,.005,'hermite')
            ledger.append(dict(seed=seed,kind=kind,operation='hermite_sat_dt_0.005',seconds=time.perf_counter()-tic,geometry_queries=hg.size))
            for i in range(len(a0)):
                tic=time.perf_counter();ref=interval_ccd(a0[i],a1[i],b0[i],b1[i],ba[i],bb[i],min_dt=cfg['ccd_min_interval_seconds'])
                ledger.append(dict(seed=seed,kind=kind,operation='linear_interval_ccd',seconds=time.perf_counter()-tic,geometry_queries=ref['calls']))
                tic=time.perf_counter();href=interval_ccd(a0[i],a1[i],b0[i],b1[i],ba[i],bb[i],kind='hermite',min_dt=cfg['ccd_min_interval_seconds'])
                ledger.append(dict(seed=seed,kind=kind,operation='hermite_interval_ccd',seconds=time.perf_counter()-tic,geometry_queries=href['calls']))
                tic=time.perf_counter();tube=tube_certificate(a0[i],a1[i],b0[i],b1[i],ba[i],bb[i],cfg['tube_acceleration_bound'],cfg['tube_angular_acceleration_bound'])
                ledger.append(dict(seed=seed,kind=kind,operation='tube_certificate',seconds=time.perf_counter()-tic,geometry_queries=tube['calls']))
                initial=float(gaps[.5][i,0]);endpoint=float(gaps[.5][i,-1])
                row=dict(seed=seed,block_seed=block_seed,kind=kind,scene=i,ego_token=int(aid[i]),other_token=int(bid[i]),
                    initial_gap=initial,endpoint_gap=endpoint,initially_clear=int(initial>1e-7),
                    native_collision=int(native[i]),native_ttc_penalty=int(ttc[i]<.5),
                    ccd_status=ref['status'],ccd_witness=ref['witness_time'],ccd_queries=ref['calls'],
                    hermite_status=href['status'],hermite_witness=href['witness_time'],
                    hermite_dense_collision=int(hg[i].min()<-1e-9),tube_safe=int(tube['certified_safe']),tube_lower_bound=tube['lower_clearance_bound'],
                    tube_rho=tube['rho'],hermite_accel_ego=bds['Vehicle'][aid[i]]['acceleration'])
                for dt,g in gaps.items():row['sat_collision_dt_'+str(dt)]=int(g[i].min()<-1e-9)
                for name,pose in [('ego_start',a0[i]),('ego_end',a1[i]),('other_start',b0[i]),('other_end',b1[i])]:
                    for axis,val in zip(['x','y','yaw'],pose):row[name+'_'+axis]=float(val)
                for name,box in [('ego',ba[i]),('other',bb[i])]:
                    for axis,val in zip(['front','rear','left','right'],box):row[name+'_'+axis]=float(val)
                rows.append(row)
            # Candidate-bank test: unchanged surrogate preference, no model logits.
            count=cfg['candidate_scenes_per_seed_per_kind'];K=cfg['candidates_per_scene'];rng=np.random.default_rng(block_seed+700000)
            for i in range(count):
                if gaps[.5][i,0]<=1e-7:continue
                speed=np.linalg.norm(vocab['Vehicle'][aid[i],:2])/.5
                pool=valid['Vehicle'][abs(np.linalg.norm(vocab['Vehicle'][valid['Vehicle'],:2],axis=1)/.5-speed)<=cfg['candidate_speed_band_mps']]
                choices=rng.choice(pool,K,replace=len(pool)<K)
                starts=np.repeat(a0[i:i+1],K,axis=0);ends=decode(starts,vocab['Vehicle'][choices])
                ost=np.repeat(b0[i:i+1],K,axis=0);oen=np.repeat(b1[i:i+1],K,axis=0);ea=np.repeat(ba[i:i+1],K,axis=0);ob=np.repeat(bb[i:i+1],K,axis=0)
                pref=vocab['Vehicle'][choices,0]-2*abs(vocab['Vehicle'][choices,1])-abs(vocab['Vehicle'][choices,2]);top=int(np.argmax(pref))
                method_flags={};times={};querycounts={}
                for method,dt in [('sat_endpoint',.5),('sat_10hz',.1),('sat_50hz',.02)]:
                    tic=time.perf_counter();g=sampled_gaps(starts,ends,ost,oen,ea,ob,dt)
                    method_flags[method]=(g.min(1)<-1e-9);times[method]=time.perf_counter()-tic;querycounts[method]=g.size
                tic=time.perf_counter();method_flags['native_endpoint']=corner_overlap(ends,oen,ea,ob);times['native_endpoint']=time.perf_counter()-tic;querycounts['native_endpoint']=K
                tic=time.perf_counter();cr=[interval_ccd(starts[j],ends[j],ost[j],oen[j],ea[j],ob[j],min_dt=cfg['ccd_min_interval_seconds']) for j in range(K)]
                times['ccd']=time.perf_counter()-tic;querycounts['ccd']=sum(c['calls'] for c in cr);method_flags['ccd']=np.array([c['status']!='safe' for c in cr])
                tic=time.perf_counter();tr=[tube_certificate(starts[j],ends[j],ost[j],oen[j],ea[j],ob[j]) for j in range(K)]
                times['tube']=time.perf_counter()-tic;querycounts['tube']=K*11;method_flags['tube']=np.array([not c['certified_safe'] for c in tr])
                for j in range(K):
                    candidate_rows.append(dict(seed=seed,kind=kind,scene=i,candidate=j,token_id=int(choices[j]),preference=pref[j],progress=ends[j,0]-starts[j,0],
                        top_preference=int(j==top),reference_status=cr[j]['status'],reference_witness=cr[j]['witness_time'],
                        **{'reject_'+m:int(f[j]) for m,f in method_flags.items()}))
                for method in method_flags:
                    ledger.append(dict(seed=seed,kind=kind,operation='candidate_'+method,seconds=times[method],geometry_queries=querycounts[method]))
            print('completed',seed,kind,'elapsed',round(time.perf_counter()-start,2),flush=True)
    write_csv(out/'diagnostics.csv',rows);write_csv(out/'candidate_bank.csv',candidate_rows);write_csv(out/'compute_ledger.csv',ledger)
    meta=dict(protocol=cfg,elapsed_seconds=time.perf_counter()-start,process_cpu_note='Elapsed perf_counter wall time for CPU-only execution, not OS summed process time.',
        python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        feasible_token_counts={k:len(v) for k,v in valid.items()},diagnostic_rows=len(rows),candidate_rows=len(candidate_rows),
        source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')},
        protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        sampling_warning='Synthetic uniform geometric stress population using actual token IDs; no learned-policy distribution or real benchmark failure rate.')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:meta[k] for k in ['elapsed_seconds','feasible_token_counts','diagnostic_rows','candidate_rows']},indent=2))


if __name__=='__main__':main()
