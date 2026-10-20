"""Audit recorded Plan-R1 poses without loading/training a model.

NPZ (allow_pickle=False): ego[S,T,3], others[S,N,T,3], ego_box[S,4],
other_box[S,N,4], times[T] with .5s spacing. Include current pose at time 0.
JSON sidecar records checkpoint SHA, scenario IDs and predicted/reference roles.
Optional ego_tokens[S,T-1] enables exact vocabulary/decode consistency checks.
These are geometric diagnostics of recorded poses, not real collision truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from motion_geometry import decode,interval_ccd,sampled_gaps
from run_diagnostics import write_csv,load_vocabulary


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('npz',type=Path)
    ap.add_argument('--provenance',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    provenance=json.loads(a.provenance.read_text());data=np.load(a.npz,allow_pickle=False)
    ego,others,eb,ob,times=[data[k] for k in ['ego','others','ego_box','other_box','times']]
    assert ego.ndim==3 and others.ndim==4 and ego.shape[0]==others.shape[0] and ego.shape[1]==others.shape[2]
    assert ego.shape[-1]==others.shape[-1]==3 and len(times)==ego.shape[1]
    assert eb.shape==(len(ego),4) and ob.shape==others.shape[:2]+(4,)
    assert all(np.isfinite(x).all() for x in [ego,others,eb,ob,times])
    assert np.allclose(np.diff(times),.5,atol=1e-8) and np.all(eb>0) and np.all(ob>0)
    if 'ego_tokens' in data:
        ids=data['ego_tokens'];v=load_vocabulary()['Vehicle'];assert ids.shape==ego.shape[:2][:-1]+(ego.shape[1]-1,)
        assert np.issubdtype(ids.dtype,np.integer) and np.all((ids>=0)&(ids<len(v)))
        reconstruction=decode(ego[:,:-1],v[ids])
        np.testing.assert_allclose(reconstruction[...,:2],ego[:,1:,:2],atol=1e-3,rtol=0)
        assert np.max(abs((reconstruction[...,2]-ego[:,1:,2]+np.pi)%(2*np.pi)-np.pi))<1e-3
    start=time.perf_counter();rows=[]
    for s in range(len(ego)):
        for n in range(others.shape[1]):
            for t in range(len(times)-1):
                aa,az=ego[s,t],ego[s,t+1];bb,bz=others[s,n,t],others[s,n,t+1]
                ref=interval_ccd(aa,az,bb,bz,eb[s],ob[s,n])
                r=dict(scene=s,agent=n,interval=t,time_start=times[t],status=ref['status'],witness=ref['witness_time'],queries=ref['calls'])
                for dt in [.5,.1,.02]:r['collision_dt_'+str(dt)]=int(sampled_gaps(aa[None],az[None],bb[None],bz[None],eb[s:s+1],ob[s,n:n+1],dt).min()<-1e-9)
                rows.append(r)
    a.out.mkdir(parents=True,exist_ok=True);write_csv(a.out/'saved_output_diagnostics.csv',rows)
    (a.out/'saved_output_audit.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(a.npz.read_bytes()).hexdigest(),provenance=provenance,
        intervals=len(rows),elapsed_seconds=time.perf_counter()-start,
        warning='Predicted endpoint audit conditional on an explicit bridge; not independent physical/closed-loop validation.'),indent=2))
    print('Audited',len(rows),'recorded ego-agent intervals. Provenance preserved; no inference performed.')


if __name__=='__main__':main()
