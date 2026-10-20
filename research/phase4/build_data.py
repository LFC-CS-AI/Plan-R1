"""Prepare actual-token constrained paths and one shared offline label bank."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from bridge_dynamics import fit_base,certify,coefficient_cap,path,pose,LIMITS
from geometry import make_scene,refined_margins

HERE=Path(__file__).resolve().parent


def features(s,b):
    z=np.asarray(b['token']);p=np.asarray(s['other_p']);v=np.asarray(s['other_v'])
    return np.r_[z/[10,1,.1],p/[20,8],v/12,np.cos(s['other_yaw']),np.sin(s['other_yaw']),b['state']['v0']/20,b['state']['curvature0']/.1,1.].astype('float32')


def run():
    start=time.perf_counter();protocol=json.loads((HERE/'protocol.json').read_text());out=HERE/'results';out.mkdir(exist_ok=True)
    tokenpath=HERE.parents[1]/'tokens/tokens_1024.pt';tokens=torch.load(tokenpath,map_location='cpu')['Vehicle'].numpy();bases=[];reasons={'screened':0,'fit_failed':0,'certificate_failed':0}
    for i,z in enumerate(tokens):
        if not (2.5<=z[0]<=10 and abs(z[2])<=.1):reasons['screened']+=1;continue
        try:coeff,state=fit_base(z)
        except ValueError:reasons['fit_failed']+=1;continue
        if not certify(coeff,0.)[0]:reasons['certificate_failed']+=1;continue
        cb,bound=coefficient_cap(coeff,'bridge');cr,rb=coefficient_cap(coeff,'residual')
        grid=np.linspace(0,1,2001);dev=max(np.linalg.norm(path(coeff,c,grid)-path(coeff,0.,grid),axis=-1).max() for c in [-cb,cb])
        bases.append(dict(token_id=i,token=z.tolist(),coeff=coeff.tolist(),state=state,cap_bridge=cb,cap_residual=cr,bounds_bridge=bound,bounds_residual=rb,max_bridge_displacement_m=float(dev)))
    (out/'bases.json').write_text(json.dumps(dict(token_sha256=hashlib.sha256(tokenpath.read_bytes()).hexdigest(),limits=LIMITS,selection=reasons,bases=bases),indent=2))
    print('Certified original-token bases:',len(bases),flush=True)
    xgrid=np.linspace(-1,1,protocol['shared_label_candidates_per_family']);ledger={}
    for split,count in protocol['scene_counts'].items():
        tick=time.perf_counter();rng=np.random.default_rng(protocol['scene_seeds'][split]);scenes=[];X=[];labels=[];margins=[]
        for j in range(count):
            k=int(rng.integers(len(bases)));b=bases[k];coeff=np.asarray(b['coeff']);s=make_scene(rng,coeff,k,j%2==0);s['scene_id']=split+'_'+str(j);scenes.append(s);X.append(features(s,b));cost=[];gap=[]
            nominal=path(coeff,0.,1.)[0]
            for family in protocol['families']:
                cap=b['cap_'+family];fg=refined_margins(coeff,xgrid*cap,s,family).tolist();fc=[]
                for x,g in zip(xgrid,fg):
                    progress=path(coeff,x*cap,1.,family)[0];risk=1/(1+np.exp(np.clip((g-.005)/.006,-60,60)))
                    fc.append(risk+.02*max(0.,nominal-progress))
                cost.append(fc);gap.append(fg)
            labels.append(cost);margins.append(gap)
        (out/(split+'_scenes.json')).write_text(json.dumps(scenes,indent=2))
        np.savez_compressed(out/(split+'_data.npz'),features=np.array(X),labels=np.array(labels,dtype='float32'),margins=np.array(margins),xgrid=xgrid)
        ledger[split]=dict(scenes=count,label_trajectories=count*len(xgrid)*2,minimum_seed_pose_queries=count*len(xgrid)*2*201,seconds=time.perf_counter()-tick)
        print(split,ledger[split],flush=True)
    ledger['total_seconds']=time.perf_counter()-start;ledger['note']='Golden refinement and boundary construction geometry queries additional and not individually counted; included in walltime. Candidate label trajectories counted exactly; world model calls 0.'
    (out/'data_ledger.json').write_text(json.dumps(ledger,indent=2))

if __name__=='__main__':run()
