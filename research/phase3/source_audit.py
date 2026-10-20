"""Capture source provenance; verify decoded poses against actual transition AST."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import types
import numpy as np
import torch
from torch_geometric.data import HeteroData
from native_probe import REPO
from motion_geometry import decode
from utils import transform_point_to_global_coordinate,wrap_angle,move_dict_to_device


def main():
    here=Path(__file__).resolve().parent;out=here/'results';out.mkdir(exist_ok=True)
    targets={
      'transforms/token_builder.py':['__init__','__call__'],
      'model/PlanR1.py':['transition','rollout','reward_fn','plan_inference'],
      'rewards/agent_collision_reward.py':['forward'],
      'rewards/obstacle_collision_reward.py':['forward'],
      'rewards/ttc_reward.py':['forward'],
      'simulation/planr1_planner.py':['compute_planner_trajectory','global_trajectory_to_states'],
      'simulation/planr1_planner_with_refinement.py':['_get_closed_loop_trajectory'],
      'utils/process_data.py':['compute_corner_positions','interpolate_traj_using_midpoints'],
      'datasets/nuplan_utils.py':['get_features'],
      'tuplan_garage/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py':['_calculate_no_at_fault_collision']}
    evidence=[]
    for file,names in targets.items():
        path=REPO/file;tree=ast.parse(path.read_text());functions=[]
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
                functions.append(dict(function=node.name,start_line=node.lineno,end_line=node.end_lineno))
        evidence.append(dict(file=file,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),functions=functions))
    tree=ast.parse((REPO/'model/PlanR1.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='transition')
    mod=ast.Module(body=[node],type_ignores=[]);ns=dict(torch=torch,
       transform_point_to_global_coordinate=transform_point_to_global_coordinate,wrap_angle=wrap_angle,
       move_dict_to_device=move_dict_to_device)
    exec(compile(mod,'original_PlanR1_transition_AST','exec'),ns)
    tokenfile=REPO/'tokens/tokens_1024.pt';vocab=torch.load(tokenfile,map_location='cpu',weights_only=True)
    start=np.array([[3.,-2.,.7],[-1.,4.,-2.3]])
    data=HeteroData();data['agent'].type=torch.tensor([0,1])
    data['agent'].infer_position=torch.tensor(start[:,:2],dtype=torch.float32)[:,None,:]
    data['agent'].infer_heading=torch.tensor(start[:,2],dtype=torch.float32)[:,None]
    data['agent'].infer_token=torch.zeros((2,0),dtype=torch.long)
    data['agent'].infer_token_mask=torch.ones((2,1),dtype=torch.bool)
    data['agent'].infer_valid_mask=torch.ones((2,1),dtype=torch.bool)
    ids=torch.tensor([803,37]);dummy=types.SimpleNamespace(token_dict=vocab)
    result=ns['transition'](dummy,data,ids)
    actual=np.concatenate([result['agent'].infer_position[:,-1].numpy(),result['agent'].infer_heading[:,-1,None].numpy()],-1)
    expected=decode(start,np.stack([vocab['Vehicle'][803].numpy(),vocab['Pedestrian'][37].numpy()]))
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0')
    original={f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in tracked if f and not f.startswith('research/')}
    (out/'source_audit.json').write_text(json.dumps(dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        evidence=evidence,original_file_sha256=original,
        token=dict(path='tokens/tokens_1024.pt',bytes=tokenfile.stat().st_size,sha256=hashlib.sha256(tokenfile.read_bytes()).hexdigest(),
            shapes={k:list(v.shape) for k,v in vocab.items()}),
        transition_ast_verification=dict(status='PASS',scope='Exact original transition function body; not full model inference.',max_abs_error=float(abs(actual-expected).max())),
        real_model_preflight=dict(checkpoints_found=[str(p.relative_to(REPO)) for p in REPO.rglob('*.ckpt')],
            verified_real_outputs_found=0,reason='No checkpoint or recorded model-output/feature bundle supplied. No random-weight outputs substituted.'),
        dormant_midpoint_helper='Definition and export only: repository search found no invocation in reward or planner path.'),indent=2))
    print('PASS original transition AST parity; source hashes saved; no real checkpoint/output bundle available.')


if __name__=='__main__':main()
