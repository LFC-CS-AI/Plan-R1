"""Reproducible source ranges, hashes and CPU dependency inventory."""
import ast,hashlib,json,os,platform,sys
from pathlib import Path
import importlib.metadata as im
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def run():
    records=[]
    for name in ['model/PlanR1.py','transforms/token_builder.py','modules/map_encoder.py','modules/backbone.py','datasets/nuplan_dataset.py','datasets/nuplan_utils.py']:
        f=ROOT/name;source=f.read_text();tree=ast.parse(source);defs=[]
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):defs.append(dict(name=node.name,line=node.lineno,end_line=node.end_lineno))
        records.append(dict(file=name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),definitions=defs))
    (HERE/'results/source_audit.json').write_text(json.dumps(dict(base_commit='fd7d0364c97c2bc77423e8aeaec5699a4dfd32f7',files=records),indent=2))
    packages={}
    for name in ['numpy','scipy','torch','torch-geometric','torch-scatter','pytorch-lightning','matplotlib','shapely']:
        try:packages[name]=im.version(name)
        except im.PackageNotFoundError:packages[name]=None
    cpu='unavailable'
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):cpu=line.split(':',1)[1].strip();break
    (HERE/'results/environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),cpu=cpu,cpu_count=os.cpu_count(),packages=packages,inference_threads=1,pretrained_inference_run=False),indent=2));print('Source and environment audit saved.')

if __name__=='__main__':run()
