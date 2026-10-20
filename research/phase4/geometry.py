"""Dense offline labels and independent adaptive interval test evaluation.

Both evaluate specified synthetic motions, not actual traffic. Reusing the
rectangle SAT primitive does not make offline temporal sampling a certificate.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase3'))
from motion_geometry import sat_gap
from bridge_dynamics import pose, LIMITS,eval_poly,W_BRIDGE,W_RESIDUAL
from numpy.polynomial import Polynomial as P

EGO_BOX=np.array([3.95,1.05,1.,1.])
OTHER_BOX=np.array([2.2,2.2,.9,.9])


def other_pose(scene,t):
    t=np.asarray(t);p=np.asarray(scene['other_p'])+t[...,None]*np.asarray(scene['other_v'])
    return np.concatenate([p,np.full(t.shape+(1,),scene['other_yaw'])],-1)


def gaps(coeff,c,scene,t,family='bridge'):
    return sat_gap(pose(coeff,c,np.asarray(t),family),other_pose(scene,t),EGO_BOX,OTHER_BOX)


def sampled(coeff,c,scene,family='bridge',hz=200):
    t=np.linspace(0,1,hz+1);return float(gaps(coeff,c,scene,t,family).min())


def point_gaps(coeff,c,scene,t,family='bridge'):
    """Vectorized paired coefficients/times; positive time warp keeps yaw."""
    tau=t+c*P(W_BRIDGE if family=='bridge' else W_RESIDUAL)(t)
    pos=eval_poly(coeff,tau);vel=eval_poly(coeff,tau,1)
    pa=np.concatenate([pos,np.arctan2(vel[...,1],vel[...,0])[...,None]],-1)
    pb=np.broadcast_to(other_pose(scene,t),pa.shape)
    return sat_gap(pa,pb,EGO_BOX,OTHER_BOX)


def refined_margins(coeff,cs,scene,family='bridge'):
    """Numerical continuous minimization, seeded by ALL 200Hz local minima.

    Vectorized golden-section search refines each bracket to ~1e-9 s. Not a
    global analytical guarantee; independent interval reference audits it.
    """
    cs=np.asarray(cs);t=np.linspace(0,1,201);g=point_gaps(coeff,cs[:,None],scene,t[None,:],family)
    row,col=np.where((g[:,1:-1]<=g[:,:-2])&(g[:,1:-1]<=g[:,2:]));col=col+1
    result=np.minimum(g[:,0],g[:,-1])
    if not len(row):return result
    lo=t[col-1].copy();hi=t[col+1].copy();c=cs[row];phi=(np.sqrt(5)-1)/2
    a=hi-phi*(hi-lo);b=lo+phi*(hi-lo);ga=point_gaps(coeff,c,scene,a,family);gb=point_gaps(coeff,c,scene,b,family)
    for _ in range(34):
        left=ga<gb;hi=np.where(left,b,hi);lo=np.where(left,lo,a)
        # Recompute both vectorized points; avoids per-candidate Python loops.
        a=hi-phi*(hi-lo);b=lo+phi*(hi-lo);ga=point_gaps(coeff,c,scene,a,family);gb=point_gaps(coeff,c,scene,b,family)
    np.minimum.at(result,row,np.minimum(ga,gb));return result


def reference(coeff,c,scene,family='bridge',buffer=0.,min_dt=.0001):
    """A separating axis at the midpoint stays separating if point-motion
    bounds cannot consume its gap. Certificate conditional on certified ego
    dynamics and constant-velocity surrounding vehicle. Unknown never safe.
    """
    radius=max(np.hypot(x,y) for x in EGO_BOX[:2] for y in EGO_BOX[2:])
    L=LIMITS['speed']+radius*LIMITS['yaw_rate']+np.linalg.norm(scene['other_v'])
    g=gaps(coeff,c,scene,np.array([0.,1.]),family);calls=2;seen=float(g.min())
    if seen < buffer-1e-10:return dict(status='collision',queries=calls,min_seen=seen,witness_time=float(np.argmin(g)))
    intervals=np.array([[0.,1.]]);unknown=False
    while len(intervals):
        lo,hi=intervals.T;mid=(lo+hi)/2;g=gaps(coeff,c,scene,mid,family);calls+=len(mid)
        seen=min(seen,float(g.min()))
        if np.any(g < buffer-1e-10):
            j=int(np.argmin(g));return dict(status='collision',queries=calls,min_seen=seen,witness_time=float(mid[j]))
        unresolved=g<=buffer+L*(hi-lo)/2+1e-10
        tiny=(hi-lo)<=min_dt
        unknown=unknown or bool(np.any(unresolved&tiny))
        keep=unresolved&~tiny;lo=lo[keep];hi=hi[keep];mid=(lo+hi)/2
        intervals=np.concatenate([np.stack([lo,mid],-1),np.stack([mid,hi],-1)])
    return dict(status='unknown' if unknown else 'safe',queries=calls,min_seen=seen,witness_time=None)


def make_scene(rng,coeff,base_index,boundary):
    angle=rng.uniform(-np.pi,np.pi);velocity=rng.uniform(3,12)*np.array([np.cos(angle),np.sin(angle)])
    s=dict(base_index=int(base_index),population='boundary' if boundary else 'broad',other_v=velocity.tolist(),other_yaw=float(angle))
    if not boundary:
        s['other_p']=[float(rng.uniform(-5,25)),float(rng.uniform(-8,8))];return s
    # Construct near a collision-boundary of c=0 ONLY, without using an oracle
    # bridge or learned outputs. This deliberately enriched population is not
    # an estimate of prevalence in nuPlan or on-road traffic.
    target=float(rng.uniform(-.03,.03));tm=float(rng.choice([.25,.75]))
    pe=pose(coeff,0.,tm);ve=(pose(coeff,0.,tm+.0001)[:2]-pose(coeff,0.,tm-.0001)[:2])/.0002
    rel=velocity-ve;normal=np.array([-rel[1],rel[0]])/max(np.linalg.norm(rel),1e-9)
    normal*=rng.choice([-1,1]);center=pe[:2]+1.45*np.array([np.cos(pe[2]),np.sin(pe[2])])
    from scipy.optimize import brentq
    def f(distance):
        s['other_p']=(center-velocity*tm+distance*normal).tolist()
        return float(refined_margins(coeff,np.array([0.]),s)[0]-target)
    distance=brentq(f,0.,20.,xtol=1e-9)
    s['other_p']=(center-velocity*tm+distance*normal).tolist();s['target_gap']=target;s['construction_time']=tm
    return s
