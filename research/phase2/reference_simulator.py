"""Independent 2-D reactive reference evaluator; never imported by the predictor.

Two perpendicular lanes, axis-aligned rectangular vehicles. This module uses
time-stepped jerk-limited dynamics, delayed perception and a latched yield state.
The predictor uses arrival intervals instead, not this dynamics implementation.
"""
from dataclasses import dataclass
import numpy as np

ACCELERATIONS = np.array([-6.0, -2.0, 0.0, 1.0, 2.0])
ACTION_NAMES = ['stop', 'slow', 'coast', 'go', 'assert']
PULSE_SECONDS = 1.2
HALF_CONFLICT = 3.1  # 4.4m vehicle length / 2 + 1.8m width / 2


@dataclass(frozen=True)
class Domain:
    cooperative_probability: float = .7
    perception_delay: float = .24
    distance_bias: float = 0.


DOMAINS = {'id': Domain(), 'driver_shift': Domain(.4, .55, 1.)}


def make_scenes(seed, count):
    rng = np.random.default_rng(seed)
    # Uniform scene population, not rejection-sampled for risk or model failures.
    de = rng.uniform(18., 30., count)
    ve = rng.uniform(7., 10., count)
    vo = rng.uniform(6., 10., count)
    offset = rng.uniform(-1.25, 1.25, count)
    do = np.clip(vo * (de / ve + offset), 10., 42.)
    return np.stack([de, ve, do, vo], axis=1)


def _inside_segment(p0, p1):
    """Time fraction interval in [-3.1,3.1], allowing stationary segments."""
    delta = p1 - p0
    moving = delta > 1e-12
    enter = np.where(moving, (-HALF_CONFLICT-p0)/np.maximum(delta,1e-12), 0.)
    leave = np.where(moving, (HALF_CONFLICT-p0)/np.maximum(delta,1e-12), 1.)
    valid = (p1 >= -HALF_CONFLICT) & (p0 <= HALF_CONFLICT)
    return np.maximum(enter, 0.), np.minimum(leave, 1.), valid


def evaluate(scenes, strength, domain, samples, seed, dt=.08, horizon=8.):
    """Paired potential outcomes for all five actions using shared exogenous draws.

    Returns probability estimates/counts, progress until first collision, yield
    frequency and crossing time. Reference draws never go to the selector.
    One reference rollout = one (scene, action, latent realization).
    """
    rng = np.random.default_rng(seed)
    cfg = DOMAINS[domain]
    s = np.asarray(scenes); S, A, N = len(s), 5, samples
    shape = (S, A, N)
    coop = rng.random((S,1,N)) < cfg.cooperative_probability
    delay = np.clip(cfg.perception_delay+rng.normal(0,.10,(S,1,N)), 0., .9)
    speed_factor = np.clip(1.+rng.normal(0,.07,(S,1,N)), .75, 1.25)
    pos_jitter = rng.normal(0,.35,(S,1,N))
    brake = rng.uniform(4.5, 6.5, (S,1,N))
    desired_speed = s[:,3,None,None] * speed_factor
    x = np.broadcast_to(-s[:,0,None,None], shape).copy()
    v = np.broadcast_to(s[:,1,None,None], shape).copy()
    y = np.broadcast_to(-s[:,2,None,None]+pos_jitter+cfg.distance_bias,shape).copy()
    ov = np.broadcast_to(desired_speed,shape).copy()
    oa = np.zeros(shape)
    yielded = np.zeros(shape, dtype=bool)
    collided = np.zeros(shape, dtype=bool)
    progress = np.zeros(shape)
    crossed_at = np.full(shape, horizon+1.)
    original_x = x.copy()
    history_x, history_v = [], []
    delay_steps = np.rint(delay/dt).astype(int)
    max_delay = int(np.ceil(.9/dt))+1
    t = 0.
    for step in range(int(round(horizon/dt))):
        history_x.append(x.copy()); history_v.append(v.copy())
        if len(history_x)>max_delay: history_x.pop(0); history_v.pop(0)
        px, pv = np.zeros(shape), np.zeros(shape)
        # Delay bins refer to the same physical noise variables across actions.
        for d in range(min(step, max_delay-1)+1):
            choose = np.minimum(delay_steps, step)==d
            px = np.where(choose, history_x[-1-d],px)
            pv = np.where(choose, history_v[-1-d],pv)
        ego_ttc = np.maximum(-px,0.)/np.maximum(pv,.2)
        other_ttc = np.maximum(-y,0.)/np.maximum(ov,.2)
        conflict = (np.abs(ego_ttc-other_ttc)<1.1) & (ego_ttc<other_ttc+.75)
        active = (px < HALF_CONFLICT) & (y < -HALF_CONFLICT) & (px > -22.)
        yielded |= coop & conflict & active & (strength>0)
        holding = yielded & (px < HALF_CONFLICT+.8) & (y < -HALF_CONFLICT)
        stopping_distance = np.maximum(-y-HALF_CONFLICT-.5, .3)
        desired_accel = np.clip(1.5*(desired_speed-ov),-4.,2.)
        yield_accel = -strength*ov**2/(2.*stopping_distance)
        desired_accel = np.where(holding,np.maximum(yield_accel,-brake),desired_accel)
        # Finite jerk and perception latency are absent from the surrogate.
        oa += np.clip(desired_accel-oa,-8.*dt,8.*dt)
        next_ov = np.clip(ov+oa*dt,0.,13.)
        next_y = y+.5*(ov+next_ov)*dt
        ego_acc = np.broadcast_to(ACCELERATIONS[None,:,None],shape).copy()
        ego_acc[:,1:,:] *= float(t<PULSE_SECONDS-1e-8)
        next_v = np.clip(v+ego_acc*dt,0.,14.)
        next_x = x+.5*(v+next_v)*dt
        ex, lx, ix = _inside_segment(x,next_x)
        ey, ly, iy = _inside_segment(y,next_y)
        hit = ix & iy & (np.maximum(ex,ey)<=np.minimum(lx,ly))
        first = hit & ~collided
        contact_fraction = np.clip(np.maximum(ex,ey),0.,1.)
        step_progress = np.where(first,(next_x-x)*contact_fraction,next_x-x)
        progress += np.where(collided,0.,step_progress)
        collided |= hit
        crossing = (y<0.) & (next_y>=0.) & (crossed_at>horizon)
        crossed_at = np.where(crossing,t+dt*(-y)/np.maximum(next_y-y,1e-12),crossed_at)
        x,v,y,ov = next_x,next_v,next_y,next_ov
        t += dt
    counts=collided.sum(-1)
    assert np.all(counts[:,0]==0), 'Emergency stop must stay outside conflict area.'
    return dict(counts=counts, risk=counts/N, progress=progress.mean(-1),
                yield_rate=yielded.mean(-1), other_arrival=crossed_at.mean(-1),
                calls=S*A*N, latent_samples=N)
