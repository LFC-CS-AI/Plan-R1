"""Tests that can catch a geometry/time diagnostic being wrong."""
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from motion_geometry import decode,bridge,bounds,rectangle,sat_gap,corner_overlap,interval_ccd
from native_probe import native_rewards


def main():
    rng=np.random.default_rng(734);box=np.array([2.2,2.2,.9,.9])
    # Crossed bars overlap but no vertex is inside the other rectangle.
    a=np.array([0.,0.,0.]);b=np.array([0.,0.,np.pi/2])
    assert sat_gap(a,b,box,box)<0 and not corner_overlap(a,b,box,box)
    native,_,_=native_rewards(a[None],a[None],b[None],b[None],box[None],box[None])
    assert not native[0]
    # Pure temporal counterexample, same orientation: relative translation sweeps
    # through while both endpoints are separated. Fixed-heading analytic witness.
    a0=np.array([-6.,0.,0.]);a1=np.array([6.,0.,0.]);b0=np.array([0.,0.,np.pi/2])
    assert sat_gap(a0,b0,box,box)>0 and sat_gap(a1,b0,box,box)>0
    assert interval_ccd(a0,a1,b0,b0,box,box)['status']=='collision'
    assert interval_ccd(a0,a1,b0+np.array([0.,10.,0]),b0+np.array([0.,10.,0]),box,box)['status']=='safe'
    for _ in range(600):
        poses=rng.uniform([-8,-8,-np.pi],[8,8,np.pi],(2,3));polys=[]
        for pose in poses:
            c,u,v,h=rectangle(pose,box)
            polys.append(Polygon([c+i*h[0]*u+j*h[1]*v for i,j in [(1,1),(1,-1),(-1,-1),(-1,1)]]))
        assert bool(sat_gap(poses[0],poses[1],box,box)<0)==polys[0].intersects(polys[1])
    # Bounds are deliberately upper bounds, check sampled derivative envelopes.
    end=np.array([5.,.1,.04]);t=np.linspace(0,1,10001);z=np.zeros(3)
    path=bridge(z,end,t,'hermite');bd=bounds(z,end,box,'hermite')
    vel=np.diff(path[:,:2],axis=0)/(t[1]/2)
    assert np.linalg.norm(vel,axis=1).max()<=bd['speed']+1e-4
    assert np.abs(np.diff(np.unwrap(path[:,2]))/(t[1]/2)).max()<=bd['yaw_rate']+1e-4
    np.testing.assert_allclose(path[[0,-1]],np.stack([z,end]),atol=1e-10)
    result=dict(status='PASS',sat_vs_independent_shapely_pairs=600,native_crossed_bar_miss_confirmed=True,
                translation_temporal_collision_confirmed=True,hermite_endpoint_and_bound_checks=True)
    (Path(__file__).parent/'results/geometry_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
