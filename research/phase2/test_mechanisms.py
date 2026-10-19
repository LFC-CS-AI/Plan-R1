"""Small tests for mechanisms and evaluation integrity, not paper performance."""
import json
from pathlib import Path
import numpy as np
from reference_simulator import make_scenes,evaluate
from world_model import ego_crossing_times,fit_world,predict
from baselines import conformal_quantile,estimate_and_choose


def main():
    s=make_scenes(123,8)
    ref=evaluate(s,1.3,'id',128,4)
    assert np.all(ref['risk'][:,0]==0)
    assert np.all(ref['progress']>=0)
    again=evaluate(s,1.3,'id',128,4)
    for k in ref:
        np.testing.assert_array_equal(ref[k],again[k])
    passive=evaluate(s,0.,'id',128,4)
    assert np.all(passive['yield_rate']==0)
    assert ref['yield_rate'].sum()>0
    assert np.max(np.abs(ref['risk']-passive['risk']))>0
    model=fit_world(s,ref,5)
    events,calls=predict(s,model,16,9,ensemble=True)
    assert events.shape==(8,5,16) and calls==8*5*16
    assert np.all(events[:,0]==0)
    # Predictor is an independent module and does not import evaluator internals.
    source=(Path(__file__).parent/'world_model.py').read_text()
    assert 'from reference_simulator' not in source and 'import reference_simulator' not in source
    for method in json.loads((Path(__file__).parent/'protocol.json').read_text())['methods']:
        pred,up,chosen,c=estimate_and_choose(method,s,model,80,8,
            dict(residual_offset=0.,marginal_q=.1,simultaneous_q=.2),.1,.1,.03)
        assert c==(40 if method=='deterministic' else 640)
        assert np.all((chosen>=0)&(chosen<5))
    assert conformal_quantile(np.arange(9),.1)==8
    assert conformal_quantile(np.arange(9),.01)==1.  # infinity sentinel clipped to risk upper bound
    ein,emid,eout=ego_crossing_times(s)
    assert np.all(ein<=emid) and np.all(emid<=eout)
    print('PASS: deterministic replay, fallback safety, reactive response, independent predictor, shapes, exact budget counters, quantile and event ordering.')


if __name__=='__main__':main()
