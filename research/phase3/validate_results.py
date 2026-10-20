"""Post-main diagnostics; retain original protocol/data, do not tune a method."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from shapely.geometry import Polygon
from motion_geometry import bridge,rectangle,sat_gap,interval_ccd
from run_diagnostics import write_csv,load_vocabulary,feasible_ids,EGO_BOX,PED_BOX


def main():
    root=Path(__file__).parent;out=root/'results';start=time.perf_counter()
    rows=list(csv.DictReader((out/'diagnostics.csv').open()));cfg=json.loads((root/'protocol.json').read_text())
    meta=json.loads((out/'metadata.json').read_text())
    for name,sha in meta['source_hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    vocab=load_vocabulary();bd={k:feasible_ids(vocab[k],EGO_BOX if k=='Vehicle' else PED_BOX,cfg)[1] for k in ['Vehicle','Pedestrian']}
    get=lambda r,name:np.array([float(r[name+'_'+a]) for a in ['x','y','yaw']])
    box=lambda r,name:np.array([float(r[name+'_'+a]) for a in ['front','rear','left','right']])
    def poly(p,b):
        c,u,v,h=rectangle(p,b);return Polygon([c+i*h[0]*u+j*h[1]*v for i,j in [(1,1),(1,-1),(-1,-1),(-1,1)]])
    leaks=[];querycount=0;witnesses=0
    for r in rows:
        if r['ccd_status']=='safe':assert r['sat_collision_dt_0.005']=='0'
        if r['hermite_status']=='safe':assert r['hermite_dense_collision']=='0'
        if r['ccd_status']=='collision':
            t=float(r['ccd_witness'])/.5
            pa=bridge(get(r,'ego_start'),get(r,'ego_end'),t);pb=bridge(get(r,'other_start'),get(r,'other_end'),t)
            assert poly(pa,box(r,'ego')).intersects(poly(pb,box(r,'other')));witnesses+=1
        if r['initially_clear']=='1' and r['ccd_status']=='collision' and r['sat_collision_dt_0.5']=='0':leaks.append(r)
    # All temporal leaks, no hand-picked subset, Cartesian +/-10cm translation.
    robust=[];details=[];strict_count=0
    for r in leaks:
        a0,a1,b0,b1=[get(r,n) for n in ['ego_start','ego_end','other_start','other_end']];ba,bb=[box(r,n) for n in ['ego','other']]
        count=0
        for dx in [-.1,0.,.1]:
            for dy in [-.1,0.,.1]:
                delta=np.array([dx,dy,0.]);rr=interval_ccd(a0,a1,b0+delta,b1+delta,ba,bb,min_dt=cfg['ccd_min_interval_seconds']);querycount+=rr['calls']
                endpoint=min(float(sat_gap(a0,b0+delta,ba,bb)),float(sat_gap(a1,b1+delta,ba,bb)))
                count+=int(rr['status']=='collision' and endpoint>1e-7)
        ego_bd=bd['Vehicle'][int(r['ego_token'])]
        other_bd=bd['Pedestrian' if r['kind']=='crossing_pedestrian' else 'Vehicle'][int(r['other_token'])]
        steering_ok=ego_bd['yaw_rate']/ego_bd['lower_speed']<=.25 and (r['kind']=='crossing_pedestrian' or other_bd['yaw_rate']/other_bd['lower_speed']<=.25)
        strict_count+=int(steering_ok)
        ts=np.linspace(0,1,2001);g=sat_gap(bridge(a0,a1,ts),bridge(b0,b1,ts),ba,bb)
        robust.append(dict(seed=r['seed'],kind=r['kind'],scene=r['scene'],temporal_leak_perturbations=count,total_perturbations=9,
            dense_025ms_min_sat_gap=float(g.min()),dense_025ms_collision_duration=float((g<0).sum()*.00025),
            conservative_curvature_bound_le_025=int(steering_ok),missed_10hz=int(r['sat_collision_dt_0.1']=='0')))
    write_csv(out/'leak_robustness.csv',robust)
    report=dict(status='PASS',original_diagnostic_code_hashes_match=True,
        all_linear_collision_witnesses_shapely_checked=witnesses,ccd_safe_has_no_dense_counterexample=True,
        temporal_leaks=len(leaks),post_hoc_translation_radius_m=.1,perturbations_per_leak=9,
        leaks_surviving_all_9_perturbations=sum(r['temporal_leak_perturbations']==9 for r in robust),
        leaks_surviving_at_least_5_perturbations=sum(r['temporal_leak_perturbations']>=5 for r in robust),
        leaks_also_satisfying_conservative_curvature_bound_025=strict_count,
        linear_hermite_event_disagreements=sum(r['ccd_status']!=r['hermite_status'] for r in rows),
        additional_geometry_queries=querycount+len(leaks)*2001,elapsed_seconds=time.perf_counter()-start,
        scope='After-main sensitivity analysis on all temporal leak cases. Conditional robustness, not an unbiased population collision-rate estimate.')
    (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
