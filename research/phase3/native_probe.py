"""Run the UNMODIFIED original reward classes on small synthetic token inputs."""
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from rewards.agent_collision_reward import AgentCollisionReward
from rewards.ttc_reward import TTCReward


def native_rewards(a0,a1,b0,b1,ba,bb):
    collision=[];ttc=[];start=time.perf_counter()
    for j in range(0,len(a0),64):
        sl=slice(j,j+64);A=np.asarray(a0)[sl];B=np.asarray(b0)[sl]
        C=np.asarray(a1)[sl];D=np.asarray(b1)[sl];S=len(A)
        first=np.stack([A,B],1).reshape(-1,3);last=np.stack([C,D],1).reshape(-1,3)
        # Four history endpoints ending at t=0, one future endpoint at t=.5.
        hist=np.stack([first+(last-first)*t for t in [-3.,-2.,-1.,0.]],1)
        poses=np.concatenate([hist,last[:,None,:]],1)
        data={'agent':dict(infer_position=torch.tensor(poses[...,:2],dtype=torch.float32),
            infer_heading=torch.tensor(poses[...,2],dtype=torch.float32),
            infer_valid_mask=torch.ones((2*S,5),dtype=torch.bool),
            box=torch.tensor(np.stack([np.asarray(ba)[sl],np.asarray(bb)[sl]],1).reshape(-1,4),dtype=torch.float32),
            batch=torch.arange(S).repeat_interleave(2),ptr=torch.arange(0,2*S+1,2))}
        with torch.no_grad():
            done,_=AgentCollisionReward()(data);score=TTCReward()(data)
        collision.extend(done.numpy().reshape(-1).tolist());ttc.extend(score.numpy().reshape(-1).tolist())
    return np.array(collision),np.array(ttc),time.perf_counter()-start
