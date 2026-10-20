"""Audit numerical continuous labels against held-out interval queries."""
import json,time
from pathlib import Path
import numpy as np
from geometry import reference
HERE=Path(__file__).resolve().parent


def run():
    r=HERE/'results';bases=json.loads((r/'bases.json').read_text())['bases'];rng=np.random.default_rng(8901);rows=[];start=time.perf_counter()
    for split in ['train','validation','test']:
        scenes=json.loads((r/(split+'_scenes.json')).read_text());data=np.load(r/(split+'_data.npz'))
        for j in rng.choice(len(scenes),16,replace=False):
            s=scenes[j];b=bases[s['base_index']];coeff=np.asarray(b['coeff'])
            for fi,family in enumerate(['bridge','residual']):
                for k,x in enumerate(data['xgrid']):
                    g=float(data['margins'][j,fi,k]);ref=reference(coeff,float(x*b['cap_'+family]),s,family)
                    wrong=(g>1e-7 and ref['status']=='collision') or (g < -1e-7 and ref['status']=='safe')
                    rows.append(dict(split=split,scene_id=s['scene_id'],family=family,k=k,refined_gap=g,interval_status=ref['status'],contradiction=bool(wrong)))
    result=dict(checked=len(rows),contradictions=sum(z['contradiction'] for z in rows),unknown=sum(z['interval_status']=='unknown' for z in rows),seconds=time.perf_counter()-start,rows=rows)
    (r/'label_audit.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='rows'})
    assert result['contradictions']==0,'Numerical label fidelity failed; do not call this continuous-label evidence.'

if __name__=='__main__':run()
