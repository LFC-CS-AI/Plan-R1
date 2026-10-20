"""Original endpoint tokens, two explicit bridges, SAT and interval certificates.

No claim that an endpoint token uniquely defines the within-token motion.
All safety statements below are conditional on the specified bridge/bounds.
"""
import numpy as np

DURATION=.5


def wrap(a):return (a+np.pi)%(2*np.pi)-np.pi


def decode(pose,token):
    p=np.asarray(pose);k=np.asarray(token);c,s=np.cos(p[...,2]),np.sin(p[...,2])
    return np.stack([p[...,0]+c*k[...,0]-s*k[...,1],
                     p[...,1]+s*k[...,0]+c*k[...,1],wrap(p[...,2]+k[...,2])],axis=-1)


def bridge(start,end,t,kind='linear'):
    """t normalized [0,1]. Hermite velocities align exactly with endpoint yaw."""
    start=np.asarray(start);end=np.asarray(end);t=np.asarray(t)
    if kind=='linear':
        return np.concatenate([start[...,:2]+(end[...,:2]-start[...,:2])*t[...,None],
            (start[...,2]+wrap(end[...,2]-start[...,2])*t)[...,None]],axis=-1)
    d=end[...,:2]-start[...,:2];length=np.linalg.norm(d,axis=-1)
    m0=length[...,None]*np.stack([np.cos(start[...,2]),np.sin(start[...,2])],-1)
    m1=length[...,None]*np.stack([np.cos(end[...,2]),np.sin(end[...,2])],-1)
    a=3*d-2*m0-m1;b=-2*d+m0+m1
    pos=start[...,:2]+m0*t[...,None]+a*(t*t)[...,None]+b*(t**3)[...,None]
    vel=m0+2*a*t[...,None]+3*b*(t*t)[...,None]
    yaw=np.arctan2(vel[...,1],vel[...,0])
    return np.concatenate([pos,yaw[...,None]],-1)


def bounds(start,end,box,kind='linear'):
    """Analytic upper bounds on reference-point speed, acceleration, yaw rate.

    Hermite derivative is a quadratic Bezier curve. Convex hull bounds speed
    and forward velocity, affine acceleration is maximized at an endpoint.
    yaw_rate <= |acceleration| / positive lower speed bound.
    """
    d=np.asarray(end)[:2]-np.asarray(start)[:2];length=np.linalg.norm(d)
    radius=max(np.hypot(x,y) for x in [box[0],box[1]] for y in [box[2],box[3]])
    if kind=='linear':
        speed=length/DURATION;yawrate=abs(wrap(end[2]-start[2]))/DURATION
        return dict(speed=speed,acceleration=0.,yaw_rate=yawrate,point_lipschitz=speed+radius*yawrate)
    if length<1e-12:return dict(speed=0.,acceleration=0.,yaw_rate=0.,point_lipschitz=0.)
    m0=length*np.array([np.cos(start[2]),np.sin(start[2])]);m1=length*np.array([np.cos(end[2]),np.sin(end[2])])
    a=3*d-2*m0-m1;b=-2*d+m0+m1
    controls=np.stack([m0,3*d-m0-m1,m1])/DURATION
    speed=np.linalg.norm(controls,axis=-1).max()
    lower=(controls@(d/length)).min()
    accel=max(np.linalg.norm(2*a),np.linalg.norm(2*a+6*b))/DURATION**2
    yawrate=accel/max(lower,1e-9)
    return dict(speed=float(speed),acceleration=float(accel),yaw_rate=float(yawrate),
                lower_speed=float(lower),point_lipschitz=float(speed+radius*yawrate))


def rectangle(pose,box):
    p=np.asarray(pose);b=np.asarray(box);yaw=p[...,2]
    u=np.stack([np.cos(yaw),np.sin(yaw)],-1);v=np.stack([-np.sin(yaw),np.cos(yaw)],-1)
    center=p[...,:2]+u*((b[...,0]-b[...,1])/2)[...,None]+v*((b[...,2]-b[...,3])/2)[...,None]
    half=np.stack([(b[...,0]+b[...,1])/2,(b[...,2]+b[...,3])/2],-1)
    return center,u,v,half


def sat_gap(pa,pb,ba,bb):
    """Maximum separating-axis gap: positive => Euclidean distance lower bound.

    Negative => positive-area overlap, not a signed Euclidean distance inside.
    Supports asymmetric front/rear reference offsets exactly.
    """
    ca,ua,va,ha=rectangle(pa,ba);cb,ub,vb,hb=rectangle(pb,bb)
    axes=np.stack([ua,va,ub,vb],-2);rel=cb-ca
    ra=np.abs(np.sum(axes*ua[...,None,:],-1))*ha[...,0,None]+np.abs(np.sum(axes*va[...,None,:],-1))*ha[...,1,None]
    rb=np.abs(np.sum(axes*ub[...,None,:],-1))*hb[...,0,None]+np.abs(np.sum(axes*vb[...,None,:],-1))*hb[...,1,None]
    return (np.abs(np.sum(axes*rel[...,None,:],-1))-ra-rb).max(-1)


def corner_overlap(pa,pb,ba,bb):
    """Strict corner-containment predicate used by original agent reward."""
    def inside(a,b,boxa,boxb):
        ca,u,v,half=rectangle(a,boxa)
        signs=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
        corners=ca[...,None,:]+u[...,None,:]*half[...,0,None,None]*signs[:,0,None]+v[...,None,:]*half[...,1,None,None]*signs[:,1,None]
        diff=corners-np.asarray(b)[...,None,:2];yaw=np.asarray(b)[...,2]
        x=diff[...,0]*np.cos(yaw)[...,None]+diff[...,1]*np.sin(yaw)[...,None]
        y=-diff[...,0]*np.sin(yaw)[...,None]+diff[...,1]*np.cos(yaw)[...,None]
        boxb=np.asarray(boxb)
        return ((x<boxb[...,0,None])&(x>-boxb[...,1,None])&(y<boxb[...,2,None])&(y>-boxb[...,3,None])).any(-1)
    return inside(pa,pb,ba,bb)|inside(pb,pa,bb,ba)


def sampled_gaps(a0,a1,b0,b1,ba,bb,dt,kind='linear'):
    t=np.linspace(0,1,int(round(DURATION/dt))+1)
    pa=bridge(np.asarray(a0)[:,None,:],np.asarray(a1)[:,None,:],t[None,:],kind)
    pb=bridge(np.asarray(b0)[:,None,:],np.asarray(b1)[:,None,:],t[None,:],kind)
    return sat_gap(pa,pb,np.asarray(ba)[:,None,:],np.asarray(bb)[:,None,:])


def interval_ccd(a0,a1,b0,b1,ba,bb,kind='linear',min_dt=.00005):
    """Adaptive certificate, never conflates unknown with observed collision.

    At midpoint, gap>sum(vertex speed bounds)*half_interval proves separation
    throughout it via Hausdorff motion bounds; overlap is a witness. Residual
    unresolved intervals are UNKNOWN, including possible grazing contacts.
    """
    L=bounds(a0,a1,ba,kind)['point_lipschitz']+bounds(b0,b1,bb,kind)['point_lipschitz']
    queue=[(0.,1.)];calls=0;unknown=False;min_seen=float('inf');witness=None
    while queue:
        lo,hi=queue.pop();mid=(lo+hi)/2
        g=float(sat_gap(bridge(a0,a1,mid,kind),bridge(b0,b1,mid,kind),ba,bb));calls+=1;min_seen=min(min_seen,g)
        if g < -1e-9:return dict(status='collision',calls=calls,witness_time=mid*DURATION,min_seen_gap=min_seen)
        if g>L*(hi-lo)*DURATION/2+1e-9:continue
        if (hi-lo)*DURATION<=min_dt:unknown=True;continue
        queue.extend([(lo,mid),(mid,hi)])
    # Also evaluate exact endpoints: important for diagnostics of existing reward.
    for t in [0.,1.]:
        g=float(sat_gap(bridge(a0,a1,t,kind),bridge(b0,b1,t,kind),ba,bb));calls+=1;min_seen=min(min_seen,g)
        if g < -1e-9:return dict(status='collision',calls=calls,witness_time=t*DURATION,min_seen_gap=min_seen)
    return dict(status='unknown' if unknown else 'safe',calls=calls,witness_time=witness,min_seen_gap=min_seen)


def tube_certificate(a0,a1,b0,b1,ba,bb,accel=6.,angular_accel=1.):
    """Sufficient separation for bounded-acceleration endpoint bridges.

    |p(t)-linear p(t)|<=A*D^2/8 and same angular bound; rectangle displacement
    adds radius*angular_error. This is an outer bound, not a reachable-set solver
    and not proof of collision when certificate fails. No recursive guarantee.
    """
    radius=lambda b:max(np.hypot(x,y) for x in [b[0],b[1]] for y in [b[2],b[3]])
    rho=(2*accel+(radius(ba)+radius(bb))*angular_accel)*DURATION**2/8
    L=bounds(a0,a1,ba)['point_lipschitz']+bounds(b0,b1,bb)['point_lipschitz']
    times=np.linspace(0,1,11);g=np.array([sat_gap(bridge(a0,a1,t),bridge(b0,b1,t),ba,bb) for t in times])
    lower=float(g.min()-L*(DURATION/10)/2-rho)
    return dict(certified_safe=lower>1e-9,lower_clearance_bound=lower,rho=rho,calls=len(times))
