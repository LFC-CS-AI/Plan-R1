"""C2 two-token paths, exact endpoint time warps, conservative derivative bounds.

Controls act on a full 1s path. Every endpoint-preserving warp fixes t=0,.5,1;
its first two derivatives at 0 equal identity. Initial v/a/curvature are shared.
No collision-based filtering is hidden in the decoder.
"""
import math
import numpy as np
from numpy.polynomial import Polynomial as P

T=1.
LIMITS=dict(speed=30.,acceleration=4.,jerk=8.,curvature=.2,yaw_rate=.6,min_speed=2.)


def derivative(poly,order=1):return np.array([P(c).deriv(order).coef for c in poly])


def eval_poly(coeff,t,order=0):
    return np.stack([P(c).deriv(order)(t) for c in coeff],axis=-1)


def bernstein(poly,lo,hi):
    c=np.asarray(poly);n=len(c)-1;power=np.zeros(n+1)
    for i in range(n+1):
        for k in range(i+1):power[k]+=c[i]*math.comb(i,k)*lo**(i-k)*(hi-lo)**k
    return np.array([sum(power[k]*math.comb(j,k)/math.comb(n,k) for k in range(j+1)) for j in range(n+1)])


def scalar_range(poly,lo,hi):
    c=bernstein(poly,lo,hi);return float(c.min()),float(c.max())


def vector_bounds(coeff,lo,hi,axis):
    c=np.stack([bernstein(p,lo,hi) for p in coeff],axis=-1)
    return float(np.linalg.norm(c,axis=-1).max()),float((c@axis).min())


def decode(pose,z):
    c,s=np.cos(pose[2]),np.sin(pose[2]);return np.array([pose[0]+c*z[0]-s*z[1],pose[1]+s*z[0]+c*z[1],(pose[2]+z[2]+np.pi)%(2*np.pi)-np.pi])


def fit_base(token):
    """Minimum integrated squared jerk degree7 with exact real-token knots/yaw.

    Initial speed is fixed once from the nominal token; initial longitudinal
    acceleration 0, initial curvature nominal yaw/displacement. Neither changes
    between compared bridges. At .5/1, speed magnitudes are free, tangents fixed.
    """
    token=np.asarray(token,dtype='float64')
    p0=np.zeros(3);p1=decode(p0,token);p2=decode(p1,token)
    v0=np.linalg.norm(token[:2])/.5;k0=token[2]/max(np.linalg.norm(token[:2]),1e-9)
    a0=np.array([0.,v0*v0*k0]);base=np.zeros((2,8));base[0,1]=v0;base[:,2]=a0/2
    degrees=np.arange(3,8);Q=np.array([[i*(i-1)*(i-2)*j*(j-1)*(j-2)/(i+j-5) for j in degrees] for i in degrees])
    L=np.linalg.cholesky(Q);invLT=np.linalg.inv(L.T);B=np.zeros((10,10));B[:5,:5]=invLT;B[5:,5:]=invLT
    A=[];rhs=[]
    for t,pose in [(.5,p1),(1.,p2)]:
        for d in range(2):
            row=np.zeros(10);row[d*5:(d+1)*5]=t**degrees;A.append(row);rhs.append(pose[d]-P(base[d])(t))
        normal=np.array([-np.sin(pose[2]),np.cos(pose[2])]);row=np.r_[normal[0]*degrees*t**(degrees-1),normal[1]*degrees*t**(degrees-1)]
        A.append(row);rhs.append(-normal@eval_poly(base,t,1))
    A=np.array(A);rhs=np.array(rhs);y=np.linalg.lstsq(A@B,rhs,rcond=1e-11)[0];sol=B@y
    base[:,3:]=sol.reshape(2,5)
    if np.max(abs(A@sol-rhs))>1e-7:raise ValueError('Inconsistent endpoint constraints')
    for t,pose in [(.5,p1),(1.,p2)]:
        v=eval_poly(base,t,1)
        if v@np.array([np.cos(pose[2]),np.sin(pose[2])])<=0:raise ValueError('Wrong tangent direction')
    return base,dict(v0=v0,a0=a0.tolist(),curvature0=k0,knots=[p0.tolist(),p1.tolist(),p2.tolist()])


W_BRIDGE=np.array([0.,0.,0.,.5,-1.5,1.])  # t^3(t-.5)(t-1)
W_RESIDUAL=np.array([0.,0.,0.,1.])


def warp(c,family):return P([0.,1.])+c*P(W_BRIDGE if family=='bridge' else W_RESIDUAL)


def path(coeff,c,t,family='bridge',order=0):
    w=warp(c,family);tau=w(t)
    if order==0:return eval_poly(coeff,tau)
    vel=eval_poly(coeff,tau,1);d1=w.deriv(1)(t)
    if order==1:return vel*np.asarray(d1)[...,None]
    acc=eval_poly(coeff,tau,2);d2=w.deriv(2)(t)
    if order==2:return acc*np.asarray(d1*d1)[...,None]+vel*np.asarray(d2)[...,None]
    jerk=eval_poly(coeff,tau,3);d3=w.deriv(3)(t)
    return jerk*np.asarray(d1**3)[...,None]+3*acc*np.asarray(d1*d2)[...,None]+vel*np.asarray(d3)[...,None]


def pose(coeff,c,t,family='bridge'):
    p=path(coeff,c,t,family);v=path(coeff,c,t,family,1)
    return np.concatenate([p,np.arctan2(v[...,1],v[...,0])[...,None]],axis=-1)


def certify(coeff,cap,family='bridge',limits=None,cells=24):
    """Certificate for ALL coefficients c in [-cap,+cap], not sampled states.

    Scalar polynomial ranges are bounded via Bernstein convex hulls per cell.
    Chain-rule triangle bounds certify acceleration/jerk; projected speed lower
    bound controls yaw rate and curvature. Conservative, not necessary bounds.
    """
    limits=LIMITS if limits is None else limits
    w=P(W_BRIDGE if family=='bridge' else W_RESIDUAL)
    d=[np.array([P(row).deriv(n).coef for row in coeff]) for n in [1,2,3]]
    axis=eval_poly(coeff,0.,1);axis=axis/np.linalg.norm(axis)
    maximum=dict(speed=0.,acceleration=0.,jerk=0.,curvature=0.,yaw_rate=0.);vmin=float('inf')
    for lo,hi in zip(np.linspace(0,1,cells+1)[:-1],np.linspace(0,1,cells+1)[1:]):
        wr=[scalar_range(w.deriv(i).coef,lo,hi) for i in range(4)]
        mag=[max(abs(x),abs(y)) for x,y in wr]
        tau_lo=lo-cap*mag[0];tau_hi=hi+cap*mag[0]
        f0=1-cap*mag[1];f1=1+cap*mag[1];f2=cap*mag[2];f3=cap*mag[3]
        if f0<=0:return False,maximum|dict(min_speed=0.)
        b1,vl=vector_bounds(d[0],tau_lo,tau_hi,axis);b2,_=vector_bounds(d[1],tau_lo,tau_hi,axis);b3,_=vector_bounds(d[2],tau_lo,tau_hi,axis)
        speed=b1*f1;low=vl*f0;acc=b2*f1*f1+b1*f2;j=b3*f1**3+3*b2*f1*f2+b1*f3
        values=dict(speed=speed,acceleration=acc,jerk=j,curvature=acc/max(low,1e-9)**2,yaw_rate=acc/max(low,1e-9))
        for k,val in values.items():maximum[k]=max(maximum[k],val)
        vmin=min(vmin,low)
    maximum['min_speed']=vmin
    good=vmin>=limits['min_speed'] and all(maximum[k]<=limits[k]+1e-9 for k in ['speed','acceleration','jerk','curvature','yaw_rate'])
    return good,maximum


def coefficient_cap(coeff,family,limits=None):
    if not certify(coeff,0.,family,limits)[0]:return 0.,None
    lo,hi=0.,1.
    for _ in range(22):
        mid=(lo+hi)/2
        if certify(coeff,mid,family,limits)[0]:lo=mid
        else:hi=mid
    cap=lo*.98
    return cap,certify(coeff,cap,family,limits)[1]
