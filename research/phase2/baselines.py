"""Conventional point, binomial, paired, residual and split-conformal baselines.

These are NOT full reproductions of CRAFT, AWM, AD-WM, RISE, or CPC.
"""
import numpy as np
from scipy.stats import beta
from world_model import predict


def binomial_interval(k,n,alpha=.1):
    k=np.asarray(k)
    lo=np.where(k==0,0.,beta.ppf(alpha/2,np.maximum(k,1),n-k+1))
    hi=np.where(k==n,1.,beta.ppf(1-alpha/2,k+1,np.maximum(n-k,1)))
    return lo,hi


def conformal_quantile(scores,alpha):
    scores=np.asarray(scores).reshape(-1)
    rank=int(np.ceil((len(scores)+1)*(1-alpha)))
    if rank>len(scores): return 1.
    return float(np.sort(scores)[rank-1])


def fit_calibration(scenes,reference,model,budget,seed,alpha):
    events,calls=predict(scenes,model,budget//5,seed,ensemble=True)
    phat=events.mean(-1)
    # Grounded labels have finite Monte Carlo precision. Union-bound confidence
    # over calibration scene/action labels, kept separate from sampling alpha.
    _,hi=binomial_interval(reference['counts'],reference['latent_samples'],.01/(len(scenes)*5))
    residual=hi[:,1:]-phat[:,1:]
    # A global residual intercept: a deliberately simple grounded correction.
    offset=float((reference['risk'][:,1:]-phat[:,1:]).mean())
    return dict(marginal_q=max(0.,conformal_quantile(residual,alpha)),
                simultaneous_q=max(0.,conformal_quantile(residual.max(1),alpha)),
                residual_offset=offset,offline_model_calls=calls)


def estimate_and_choose(method,scenes,model,budget,seed,cal,alpha,delta,tolerance):
    deterministic=method=='deterministic'
    n=1 if deterministic else budget//5
    events,calls=predict(scenes,model,n,seed,ensemble=method not in ['deterministic','mc'],
                         paired=method=='paired_difference',deterministic=deterministic)
    phat=events.mean(-1); upper=phat.copy()
    if method=='binomial_union':
        # Ensemble is a finite equally weighted mixture with fixed member counts;
        # stratified draws are independent but not iid Bernoulli. Use a valid
        # per-member CP bound then average, with a member/action union correction.
        upper=np.zeros_like(phat)
        for m in range(4):
            ev=events[:,:,m::4]
            upper+=binomial_interval(ev.sum(-1),ev.shape[-1],alpha/(4*4))[1]/4
    elif method=='paired_difference':
        base=events[:,2:3,:]
        pos=((events==1)&(base==0)).sum(-1)
        neg=((events==0)&(base==1)).sum(-1)
        # Bounds target the model mixture only. Member-wise averaging handles
        # fixed-stratum ensemble sampling, instead of assuming iid draws.
        dup=np.zeros_like(phat)
        for m in range(4):
            e=events[:,:,m::4]; b=base[:,:,m::4]
            kp=((e==1)&(b==0)).sum(-1); km=((e==0)&(b==1)).sum(-1)
            dup+=(binomial_interval(kp,e.shape[-1],alpha/(4*4*2))[1]-
                  binomial_interval(km,e.shape[-1],alpha/(4*4*2))[0])/4
        upper=np.where(dup<=tolerance,upper,1.)
        upper[:,2]=phat[:,2]
    elif method=='residual_calibration':
        phat=np.clip(phat+cal['residual_offset'],0,1); upper=phat.copy()
    elif method=='marginal_cp':
        upper=np.clip(phat+cal['marginal_q'],0,1)
    elif method=='simultaneous_cp':
        upper=np.clip(phat+cal['simultaneous_q'],0,1)
    elif method=='worst_member':
        upper=np.max(np.stack([events[:,:,m::4].mean(-1) for m in range(4)]),axis=0)
    eligible=upper<=delta
    # Highest nominal progress equals largest index in this fixed action bank.
    eligible[:,0]=True
    chosen=np.max(np.where(eligible,np.arange(5)[None,:],0),axis=1)
    return phat,upper,chosen,calls
