"""Small analytical arrival-time world-model surrogate, fit on training scenes.

No reference dynamics, collision routine, true domain parameters or test labels
are imported here. Bootstrap ensemble members are fitted, not hand-selected.
"""
import numpy as np

ACCELS = np.array([-6.,-2.,0.,1.,2.])


def ego_crossing_times(scenes):
    """Closed-form piecewise constant-acceleration interval crossings."""
    s=np.asarray(scenes); de,ve=s[:,0,None],s[:,1,None]
    a=ACCELS[None,:]
    def when(distance):
        duration=np.where(a==-6.,100.,1.2)
        stop_time=np.where(a<0,-ve/np.minimum(a,-1e-10),100.)
        duration=np.minimum(duration,stop_time)
        travelled=ve*duration+.5*a*duration**2
        finalv=np.maximum(ve+a*duration,0.)
        disc=np.maximum(ve**2+2*a*distance,0.)
        root=2*distance/np.maximum(ve+np.sqrt(disc),1e-12)
        within=(distance<=travelled) & ((a>=0) | (ve**2+2*a*distance>=0))
        after=duration+(distance-travelled)/np.maximum(finalv,1e-12)
        time=np.where(within,root,np.where(finalv>1e-6,after,1e6))
        return np.maximum(time,0.)
    return when(de-3.1),when(de),when(de+3.1)


def features(scenes):
    s=np.asarray(scenes); _,te,_=ego_crossing_times(s)
    te=np.minimum(te,12.)
    to=s[:,2,None]/s[:,3,None]
    gap=np.clip(te-to,-5.,5.)
    return np.stack([np.ones_like(gap),gap,np.abs(gap),te/5.,
                     np.broadcast_to((s[:,3,None]-8.)/2.,gap.shape),
                     np.broadcast_to(ACCELS[None,:]/3.,gap.shape)],axis=-1)


def sigmoid(x):
    return 1./(1.+np.exp(-np.clip(x,-30.,30.)))


def fit_member(scenes,reference,indices):
    X=features(scenes[indices]).reshape(-1,6)
    target=reference['yield_rate'][indices].reshape(-1)
    beta=np.zeros(6)
    # Fractional binomial logistic fit, ridge regularized; not a full model train.
    for _ in range(30):
        p=sigmoid(X@beta); w=np.maximum(p*(1-p),1e-4)
        grad=X.T@(p-target)+.5*beta
        hessian=(X.T*w)@X+.5*np.eye(6)
        beta-=np.linalg.solve(hessian,grad)
    baseline=scenes[indices,2,None]/scenes[indices,3,None]
    rates=reference['yield_rate'][indices]
    valid=rates>.15
    delays=(reference['other_arrival'][indices]-baseline)/np.maximum(rates,.15)
    delay=float(np.clip(np.median(delays[valid]),.1,5.)) if valid.any() else .1
    return dict(beta=beta.tolist(),yield_delay=delay)


def fit_world(scenes,reference,seed):
    rng=np.random.default_rng(seed)
    return dict(base=fit_member(scenes,reference,np.arange(len(scenes))),
                members=[fit_member(scenes,reference,rng.integers(0,len(scenes),len(scenes))) for _ in range(4)])


def predict(scenes,model,n,seed,ensemble=False,paired=False,deterministic=False):
    """One call per scene, candidate and stochastic complete interval prediction.

    Event outputs are [scene, action, sample]. All uncertainty distributions are
    fixed before test. The error is structural, plus optional real-domain shift.
    """
    s=np.asarray(scenes); S=len(s); A=5
    rng=np.random.default_rng(seed)
    noise_shape=(S,1 if paired else A,n)
    u=rng.random(noise_shape)
    speed_noise=rng.normal(0,.07,noise_shape)
    distance_noise=rng.normal(0,.35,noise_shape)
    jitter=rng.normal(0,.18,noise_shape)
    X=features(s)
    if ensemble:
        betas=np.array([m['beta'] for m in model['members']])[np.arange(n)%4]
        delay=np.array([m['yield_delay'] for m in model['members']])[np.arange(n)%4]
    else:
        betas=np.broadcast_to(np.array(model['base']['beta']),(n,6))
        delay=np.full(n,model['base']['yield_delay'])
    p=sigmoid(np.einsum('saf,nf->san',X,betas))
    if deterministic:
        u=np.full(noise_shape,.5); speed_noise=np.zeros(noise_shape)
        distance_noise=np.zeros(noise_shape); jitter=np.zeros(noise_shape)
    free_time=(s[:,2,None,None]+distance_noise)/np.maximum(s[:,3,None,None]*(1+speed_noise),.5)
    other_center=free_time+delay[None,None,:]*(u<p)+jitter
    other_half=3.1/np.maximum(s[:,3,None,None]*(1+speed_noise),.5)
    ego_in,ego_mid,ego_out=ego_crossing_times(s)
    collision=(ego_in[...,None]<=other_center+other_half)&(ego_out[...,None]>=other_center-other_half)
    collision &= (ego_in[...,None]<8.) & (other_center-other_half<8.)
    collision[:,0,:]=False
    return collision.astype(np.uint8),S*A*n


def nominal_progress(scenes,horizon=8.):
    s=np.asarray(scenes); ve=s[:,1,None]; a=ACCELS[None,:]
    duration=np.where(a==-6.,np.minimum(ve/6.,horizon),1.2)
    finalv=np.maximum(ve+a*duration,0.)
    return ve*duration+.5*a*duration**2+finalv*(horizon-duration)
