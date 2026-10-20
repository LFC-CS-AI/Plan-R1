"""Autograd plumbing for a proposed token+bridge objective, NOT model training.

Runs on saved synthetic candidates and one raw-token leak. Confirms interval
labels affect token-logit gradients, while endpoint-invisible bridge parameters
receive interior safety gradients. No learned-model performance claim.
"""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from motion_geometry import bounds


def endpoint_preserving_bridge(start,end,coefficient,t):
    """Cubic Hermite plus an interior bump, endpoints and endpoint tangents fixed.

    p(u)=Hermite(p0,p1,m0,m1)+16u²(1-u)² c. Both position and derivative of
    the added term vanish at u=0,1. This is one candidate parameterization;
    start/end speeds in a real planner must be state-conditioned and propagated.
    """
    delta=end[:2]-start[:2];length=torch.linalg.norm(delta)
    m0=length*torch.stack([torch.cos(start[2]),torch.sin(start[2])])
    m1=length*torch.stack([torch.cos(end[2]),torch.sin(end[2])])
    a=3*delta-2*m0-m1;b=-2*delta+m0+m1
    pos=start[:2]+t[:,None]*m0+(t*t)[:,None]*a+(t**3)[:,None]*b
    vel=m0+2*t[:,None]*a+3*(t*t)[:,None]*b
    w=16*t*t*(1-t)**2;dw=32*t-96*t*t+64*t**3
    pos=pos+w[:,None]*coefficient;vel=vel+dw[:,None]*coefficient
    return torch.cat([pos,torch.atan2(vel[:,1],vel[:,0])[:,None]],-1),vel


def torch_gap(pa,pb,ba,bb):
    def rect(p,box):
        u=torch.stack([torch.cos(p[:,2]),torch.sin(p[:,2])],-1)
        v=torch.stack([-torch.sin(p[:,2]),torch.cos(p[:,2])],-1)
        center=p[:,:2]+u*(box[0]-box[1])/2+v*(box[2]-box[3])/2
        return center,u,v,torch.stack([(box[0]+box[1])/2,(box[2]+box[3])/2])
    ca,ua,va,ha=rect(pa,ba);cb,ub,vb,hb=rect(pb,bb)
    axes=torch.stack([ua,va,ub,vb],-2)
    ra=(axes*ua[:,None]).sum(-1).abs()*ha[0]+(axes*va[:,None]).sum(-1).abs()*ha[1]
    rb=(axes*ub[:,None]).sum(-1).abs()*hb[0]+(axes*vb[:,None]).sum(-1).abs()*hb[1]
    return ((axes*(cb-ca)[:,None]).sum(-1).abs()-ra-rb).amax(-1)


def main():
    started=time.perf_counter()
    root=Path(__file__).parent;out=root/'results';rows=list(csv.DictReader((out/'candidate_bank.csv').open()))
    groups={}
    for r in rows:groups.setdefault((r['seed'],r['kind'],r['scene']),[]).append(r)
    # Same preference logits, compare endpoint vs interval cost gradients on every bank.
    raw_logits=torch.tensor([[float(r['preference']) for r in g] for g in groups.values()],dtype=torch.float64,requires_grad=True)
    interval=torch.tensor([[r['reference_status']!='safe' for r in g] for g in groups.values()],dtype=torch.float64)
    endpoint=torch.tensor([[int(r['reject_sat_endpoint']) for r in g] for g in groups.values()],dtype=torch.float64)
    pi=raw_logits.softmax(-1)
    lct=(pi*interval).sum(-1).mean();ld=(pi*endpoint).sum(-1).mean()
    gct=torch.autograd.grad(lct,raw_logits,retain_graph=True)[0];gd=torch.autograd.grad(ld,raw_logits)[0]
    assert (interval>endpoint).any() and torch.linalg.norm(gct-gd)>0
    cases=list(csv.DictReader((out/'diagnostics.csv').open()))
    r=next(r for r in cases if r['initially_clear']=='1' and r['ccd_status']=='collision' and r['sat_collision_dt_0.5']=='0')
    tensor=lambda n:torch.tensor([float(r[n+'_'+k]) for k in ['x','y','yaw']],dtype=torch.float64)
    box=lambda n:torch.tensor([float(r[n+'_'+k]) for k in ['front','rear','left','right']],dtype=torch.float64)
    a0,a1,b0,b1=[tensor(n) for n in ['ego_start','ego_end','other_start','other_end']]
    t=torch.linspace(0,1,101,dtype=torch.float64);c=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    path,deriv=endpoint_preserving_bridge(a0,a1,c,t)
    other,_=endpoint_preserving_bridge(b0,b1,torch.zeros_like(c),t)
    gap=torch_gap(path,other,box('ego'),box('other'));loss=F.softplus((.1-gap.min())/.1)
    gradient=torch.autograd.grad(loss,c,retain_graph=True)[0]
    endpoint_jac=torch.stack([torch.autograd.grad(path[j,k],c,retain_graph=True)[0] for j in [0,-1] for k in [0,1]])
    assert endpoint_jac.abs().max()==0 and torch.linalg.norm(gradient)>0
    result=dict(status='PASS',candidate_banks=len(groups),interval_expected_cost=float(lct),endpoint_expected_cost=float(ld),
        token_logit_gradient_change_norm=float(torch.linalg.norm(gct-gd)),
        endpoint_position_jacobian_max=float(endpoint_jac.abs().max()),elapsed_seconds=time.perf_counter()-started,
        interior_bridge_safety_gradient=gradient.tolist(),coefficient_update_applied=False,
        scope='Autograd verification only. No policy/head training, independent validation score, or certified learned controller.',
        missing='Bridge coefficients are NOT executed or optimized here; full dynamics limits, state-conditioned endpoint speeds and tracking remain to implement.')
    (out/'learning_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
