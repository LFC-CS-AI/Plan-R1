"""Numerical cross-checks of analytic constraints, bounds and evaluator."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial as P
from bridge_dynamics import path,pose,certify,LIMITS,W_BRIDGE
from geometry import reference,sampled

HERE=Path(__file__).resolve().parent


def run():
    bases=json.loads((HERE/'results/bases.json').read_text())['bases'];checks=0;errors=[];grid=np.linspace(0,1,2001)
    for b in bases:
        coeff=np.asarray(b['coeff']);initial=[path(coeff,0.,0.,order=i) for i in range(3)]
        for family in ['bridge','residual']:
            cap=b['cap_'+family];good,bounds=certify(coeff,cap,family);assert good;checks+=1
            for c in [-cap,0,cap]:
                for order in range(3):assert np.max(abs(path(coeff,c,0.,family,order)-initial[order]))<1e-9;checks+=1
                v=path(coeff,c,grid,family,1);a=path(coeff,c,grid,family,2);j=path(coeff,c,grid,family,3);speed=np.linalg.norm(v,axis=-1);cross=np.abs(v[:,0]*a[:,1]-v[:,1]*a[:,0]);measured=dict(speed=speed.max(),min_speed=speed.min(),acceleration=np.linalg.norm(a,axis=-1).max(),jerk=np.linalg.norm(j,axis=-1).max(),curvature=(cross/speed**3).max(),yaw_rate=(cross/speed**2).max())
                for k,value in measured.items():
                    if k=='min_speed':assert value+1e-7>=bounds[k]
                    else:assert value<=bounds[k]+1e-6
                    checks+=1
                # Global polynomial composition is C-infinity. Check adjacent
                # samples agree with its analytic derivative at the join.
                for order in range(3):
                    dt=1e-6;fd=(path(coeff,c,.5+dt,family,order)-path(coeff,c,.5-dt,family,order))/(2*dt)
                    assert np.max(abs(fd-path(coeff,c,.5,family,order+1)))<1e-4;checks+=1
                if family=='bridge':
                    knots=np.array(b['state']['knots']);q=pose(coeff,c,np.array([0,.5,1.]),family);assert np.max(abs(q-knots))<1e-7;checks+=1
        # Project t^3 residual by degree4/5 correction: r -3t^4+2t^5
        # equals 2*w. With coefficient rescaling, same trajectories exactly.
        t=grid;assert np.max(abs(t**3-3*t**4+2*t**5-2*P(W_BRIDGE)(t)))<1e-14;checks+=1
    scenes=json.loads((HERE/'results/test_scenes.json').read_text())
    for s in scenes[:64]:
        b=bases[s['base_index']];coeff=np.asarray(b['coeff']);ref=reference(coeff,0.,s);g=sampled(coeff,0.,s,hz=10000)
        if ref['status']=='safe':assert g>0
        if ref['status']=='collision':
            from geometry import gaps
            assert gaps(coeff,0.,s,np.array([ref['witness_time']]))[0]<0
        checks+=1
    result=dict(passed=True,checks=checks,bases=len(bases),dynamics_grid_points=2001,reference_dense_hz=10000,note='Cross-checks, not proof that all possible token pairs have a feasible bridge; certificates conditional on stated kinematics.')
    (HERE/'results/constraint_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
