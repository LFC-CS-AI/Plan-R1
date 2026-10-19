"""Numerical discretization and independent-reference precision diagnostics."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from reference_simulator import evaluate,make_scenes
from baselines import binomial_interval


def main():
    p=Path(__file__).resolve().parent/'results';rows=[];t=time.perf_counter()
    scenes=make_scenes(991,4)
    calls=0
    for strength in [0.,.65,1.3]:
        for domain in ['id','driver_shift']:
            coarse=evaluate(scenes,strength,domain,2048,772,dt=.08)
            fine=evaluate(scenes,strength,domain,2048,772,dt=.04)
            larger=evaluate(scenes,strength,domain,8192,773,dt=.08)
            calls+=coarse['calls']+fine['calls']+larger['calls']
            lo,hi=binomial_interval(coarse['counts'],2048,.01)
            for i in range(4):
                for a in range(5):
                    rows.append(dict(strength=strength,domain=domain,scene=i,action=a,
                        coarse_risk=coarse['risk'][i,a],fine_risk=fine['risk'][i,a],
                        larger_n_risk=larger['risk'][i,a],coarse_lo=lo[i,a],coarse_hi=hi[i,a]))
    with (p/'numerical_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    diff=np.array([abs(r['coarse_risk']-r['fine_risk']) for r in rows])
    delta=np.array([abs(r['coarse_risk']-r['larger_n_risk']) for r in rows])
    (p/'numerical_validation.json').write_text(json.dumps(dict(scenes_per_condition=4,
        timing_seconds=time.perf_counter()-t,reference_calls=calls,
        mean_dt_risk_difference=float(diff.mean()),max_dt_risk_difference=float(diff.max()),
        mean_independent_mc_risk_difference=float(delta.mean()),max_independent_mc_risk_difference=float(delta.max()),
        note='Sensitivity diagnostic only; not proof of exact continuous dynamics or universal MC precision.'),indent=2))
    print('Numerical diagnostics complete',flush=True)


if __name__=='__main__':main()
