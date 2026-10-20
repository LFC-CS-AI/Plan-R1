"""Cross-artifact consistency without retraining or adding test labels."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from train_heads import load_head,coefficient
HERE=Path(__file__).resolve().parent


def run():
    r=HERE/'results';summary=json.loads((r/'training_summary.json').read_text());tests=json.loads((r/'constraint_tests.json').read_text());audit=json.loads((r/'label_audit.json').read_text());assert tests['passed'];assert audit['contradictions']==0
    hashes=[];features={};ids=[]
    for split in ['train','validation','test']:
        scenes=json.loads((r/(split+'_scenes.json')).read_text());a=np.load(r/(split+'_data.npz'));assert len(a['features'])==len(scenes);assert np.isfinite(a['labels']).all();assert np.isfinite(a['margins']).all();features[split]=a['features'];ids.append({json.dumps([s['base_index'],s['other_p'],s['other_v']]) for s in scenes})
        hashes.append(dict(split=split,scene_sha256=hashlib.sha256((r/(split+'_scenes.json')).read_bytes()).hexdigest(),data_sha256=hashlib.sha256((r/(split+'_data.npz')).read_bytes()).hexdigest()))
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]),'Identical physical scenes crossed splits'
    assert len(summary)==12;assert all(z['optimizer_updates']==1200 and z['parameter_update_l2']>0 and z['parameters']==1769 for z in summary)
    pred=list(csv.DictReader((r/'test_predictions.csv').open()));assert len(pred)==512*21
    bases=json.loads((r/'bases.json').read_text())['bases'];scenes=json.loads((r/'test_scenes.json').read_text())
    for z in summary:
        model=load_head(r/'heads'/f"{z['method']}_{z['seed']}.npz");values=coefficient(model,features['test'],z['method']);rows=[t for t in pred if t['method']==z['method'] and int(t['seed'])==z['seed']];assert len(rows)==512
        for i,row in enumerate(rows):
            family='residual' if z['method']=='residual_head' else 'bridge';cap=bases[scenes[i]['base_index']]['cap_'+family]
            # Batched vs single-sample floating point can differ slightly.
            assert abs(values[i]*cap-float(row['coefficient']))<2e-6
            assert int(row['online_pose_queries'])==0 and int(row['accepted'])==1
    a=json.loads((r/'summary.json').read_text());assert a['total_optimizer_steps']==14400
    result=dict(passed=True,independent_scene_splits=True,head_reload_and_predictions_verified=12,head_training_updates=14400,test_prediction_rows=len(pred),label_audit_contradictions=0,source_data_hashes=hashes,real_pretrained_outputs=0)
    (r/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run()
