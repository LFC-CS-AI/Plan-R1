"""Actual CPU optimization: identical MLP size, shared data and label bank.

This is component learning, NOT a reproduction of full Plan-R1 RL training.
"""
import csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent


class Head(nn.Module):
    def __init__(self,n=12,g=9):
        super().__init__();self.net=nn.Sequential(nn.Linear(n,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,g))
    def forward(self,x):return self.net(x)


def bank_loss(logits,bank,method):
    p=logits.softmax(-1)
    if method=='reward_selector':return (p*bank).sum(-1)
    # Continuous coefficient, linear interpolation in the SAME nine labels.
    u=(p*torch.linspace(0,bank.shape[1]-1,bank.shape[1])).sum(-1)
    i=u.floor().long().clamp(0,bank.shape[1]-2);f=u-i
    return bank.gather(1,i[:,None])[:,0]*(1-f)+bank.gather(1,(i+1)[:,None])[:,0]*f


def load_head(path):
    a=np.load(path);model=Head();model.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in a.files});model.eval();return model


def coefficient(model,x,method):
    with torch.no_grad():
        logits=model(torch.as_tensor(x,dtype=torch.float32).reshape(-1,12))
        if method=='reward_selector':a=logits.argmax(-1).float()/4-1
        else:a=(logits.softmax(-1)*torch.linspace(-1,1,9)).sum(-1)
    return a.numpy()


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=json.loads((HERE/'protocol.json').read_text());r=HERE/'results';(r/'heads').mkdir(exist_ok=True)
    tr=np.load(r/'train_data.npz');va=np.load(r/'validation_data.npz');X=torch.from_numpy(tr['features']);V=torch.from_numpy(va['features']);records=[];summaries=[]
    for seed in p['training_seeds']:
        for method in ['reward_selector','residual_head','bridge_head']:
            torch.manual_seed(seed);model=Head();initial=torch.cat([v.detach().flatten().clone() for v in model.parameters()]);opt=torch.optim.Adam(model.parameters(),lr=p['learning_rate'])
            fi=1 if method=='residual_head' else 0;Y=torch.from_numpy(tr['labels'][:,fi]);Z=torch.from_numpy(va['labels'][:,fi]);best=float('inf');beststate=None;bestepoch=0;start=time.perf_counter();steps=0
            with torch.no_grad():before=float(bank_loss(model(X),Y,method).mean())
            for epoch in range(1,p['epochs']+1):
                model.train();order=torch.randperm(len(X));losses=[]
                for batch in order.split(p['batch_size']):
                    opt.zero_grad();loss=bank_loss(model(X[batch]),Y[batch],method).mean();loss.backward();opt.step();steps+=1;losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():vl=float(bank_loss(model(V),Z,method).mean())
                records.append(dict(seed=seed,method=method,epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=vl))
                if vl<best:best=vl;bestepoch=epoch;beststate={k:v.detach().clone() for k,v in model.state_dict().items()}
            elapsed=time.perf_counter()-start;model.load_state_dict(beststate)
            final=torch.cat([v.detach().flatten() for v in model.parameters()]);delta=float((final-initial).norm());np.savez_compressed(r/'heads'/f'{method}_{seed}.npz',**{k:v.numpy() for k,v in beststate.items()})
            with torch.no_grad():after=float(bank_loss(model(X),Y,method).mean())
            summary=dict(seed=seed,method=method,parameters=sum(v.numel() for v in model.parameters()),optimizer_updates=steps,selected_epoch=bestepoch,initial_train_loss=before,selected_train_loss=after,validation_loss=best,parameter_update_l2=delta,training_seconds=elapsed)
            summaries.append(summary);print(json.dumps(summary),flush=True)
    with (r/'training_curves.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (r/'training_summary.json').write_text(json.dumps(summaries,indent=2))

if __name__=='__main__':run()
