"""Replay one original block and audit all saved decision/budget records."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from reference_simulator import make_scenes
from baselines import estimate_and_choose
from run_experiment import reference_batches


def main():
    root=Path(__file__).resolve().parent;out=root/'results';start=time.perf_counter()
    cfg=json.loads((root/'protocol.json').read_text());meta=json.loads((out/'metadata.json').read_text())
    refs=list(csv.DictReader((out/'reference.csv').open()))
    rows=list(csv.DictReader((out/'decisions.csv').open()))+list(csv.DictReader((out/'extension_decisions.csv').open()))
    lookup={(int(r['seed']),float(r['strength']),r['domain'],int(r['scene']),int(r['action'])):r for r in refs}
    for r in rows:
        seed=int(float(r['seed']));scene=int(float(r['scene']));a=int(float(r['chosen']))
        rr=lookup[(seed,float(r['strength']),r['domain'],scene,a)]
        assert float(r['true_selected_risk'])==float(rr['risk'])
        assert float(r['collision'])==float(rr['deployment_collision'])
        assert float(r['expected_progress_m'])==float(rr['progress_m'])
        assert int(float(r['false_safe']))==int(a!=0 and float(rr['risk'])>.1)
        assert int(float(r['accepted']))==int(a!=0)
        assert int(float(r['model_calls']))==(5 if r['method']=='deterministic' else int(float(r['budget'])))
    for name,digest in meta['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()==meta['protocol_sha256']
    # Exact replay of 48 x 5 x 2048 independent reference outcomes, one block.
    seed=101;strength=1.3;domain='id';scenes=make_scenes(seed*100+3,cfg['test_scenes'])
    truth=reference_batches(scenes,strength,domain,cfg['reference_rollouts_per_action'],seed*10000+30,cfg)
    for i in range(len(scenes)):
        for a in range(5):
            rr=lookup[(seed,strength,domain,i,a)]
            assert int(rr['collision_count'])==truth['counts'][i,a]
            assert float(rr['progress_m'])==truth['progress'][i,a]
    model=next(x['model'] for x in json.loads((out/'models.json').read_text()) if x['seed']==seed and x['strength']==strength)
    calibrations=json.loads((out/'calibration.json').read_text());calls=0
    for budget in cfg['budgets']:
        cal=next(x for x in calibrations if x['seed']==seed and x['strength']==strength and x['budget']==budget)
        for method in cfg['methods']:
            pred,upper,chosen,ncalls=estimate_and_choose(method,scenes,model,budget,seed*100+50+budget,
                cal,cfg['alpha'],cfg['risk_threshold'],cfg['relative_risk_tolerance'])
            calls+=ncalls
            block=[r for r in rows if int(float(r['seed']))==seed and float(r['strength'])==strength and r['domain']==domain and int(float(r['budget']))==budget and r['method']==method]
            for i,r in enumerate(block):
                assert int(r['chosen'])==chosen[i]
                for a in range(5):assert float(r['pred_risk_'+str(a)])==pred[i,a]
                assert float(r['upper_selected_risk'])==upper[i,chosen[i]]
    result=dict(status='PASS',saved_decision_rows_checked=len(rows),reference_rows=len(refs),
        exact_reference_replay_scenes=len(scenes),exact_selector_replays=27,
        original_source_hashes_match=True,original_protocol_hash_matches=True,
        additional_reference_calls=truth['calls'],additional_online_model_calls=calls,
        elapsed_seconds=time.perf_counter()-start,
        scope='Consistency and one-block exact replay, not full independent implementation replication.')
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
