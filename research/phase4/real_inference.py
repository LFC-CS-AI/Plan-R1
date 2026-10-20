"""Opt-in CPU inference from an actual trusted checkpoint and real features.

Never downloads weights, builds random-weight outputs, or substitutes toy data.
Untested checkpoint execution in this round: required files were unavailable
locally. --architecture-only counts meta parameters without inference.
"""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint');p.add_argument('--features');p.add_argument('--out');p.add_argument('--architecture-only',action='store_true');a=p.parse_args()
    from model import PlanR1
    token=ROOT/'tokens/tokens_1024.pt'
    if a.architecture_only:
        with torch.device('meta'):model=PlanR1(mode='plan',token_dict_path=str(token))
        count=sum(v.numel() for v in model.parameters());print(json.dumps(dict(meta_only=True,parameters=count,fp32_parameter_bytes=count*4,inference_run=False)));return
    if not all([a.checkpoint,a.features,a.out]):p.error('--checkpoint, --features and --out are all required; no random-weight fallback')
    for name in [a.checkpoint,a.features]:
        if not Path(name).is_file():raise FileNotFoundError(name)
    from torch_geometric.data import HeteroData,Batch
    from transforms.token_builder import TokenBuilder
    torch.set_num_threads(1);torch.manual_seed(0);start=time.perf_counter()
    # Lightning defaults to strict state loading. Overrides only storage path,
    # deployment mode and deterministic top-1 selection; learned weights intact.
    model=PlanR1.load_from_checkpoint(a.checkpoint,map_location='cpu',strict=True,token_dict_path=str(token),mode='plan',plan_top_k=1,pred_top_k=1).eval();model.freeze()
    raw=torch.load(a.features,map_location='cpu');data=raw if isinstance(raw,HeteroData) else HeteroData(raw)
    # Accept a NuplanDataset processed dict before TokenBuilder, not invented
    # tensors. If a transformed input is provided this transformation repeats
    # history tokenization consistently from the underlying raw agent fields.
    transform=TokenBuilder(str(token),interval=model.interval,num_historical_steps=model.num_historical_steps,mode='plan');data=transform(data);batch=Batch.from_data_list([data]);loaded=time.perf_counter()
    current_pos=batch['agent']['infer_position'][:,-1:].clone();current_heading=batch['agent']['infer_heading'][:,-1:].clone();current_valid=batch['agent']['infer_valid_mask'][:,-1:].clone()
    with torch.inference_mode():result,pos,heading,valid=model.plan_inference(batch)
    ended=time.perf_counter();dest=Path(a.out);dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'pretrained_outputs.npz',position=torch.cat([current_pos,pos],1).cpu().numpy(),heading=torch.cat([current_heading,heading],1).cpu().numpy(),valid=torch.cat([current_valid,valid],1).cpu().numpy(),token=result['agent']['infer_token'].cpu().numpy(),box=result['agent']['box'].cpu().numpy())
    metadata=dict(checkpoint_sha256=sha(a.checkpoint),features_sha256=sha(a.features),token_sha256=sha(token),checkpoint_loaded=True,strict_state_loading=True,real_input_assertion='Caller supplies genuine processed NuPlan features; provenance must be audited separately',load_and_preprocess_seconds=loaded-start,inference_seconds=ended-loaded,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parameters=sum(v.numel() for v in model.parameters()),position_shape=list(pos.shape),current_endpoint_included=True)
    (dest/'provenance.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))

if __name__=='__main__':main()
