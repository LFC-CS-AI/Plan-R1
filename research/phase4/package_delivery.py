"""Package EXISTING artifacts only; verify CRC, bytes and scope. No experiments."""
import hashlib,json,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main():
    files=sorted(f for f in HERE.rglob('*') if f.is_file() and 'archives' not in f.relative_to(HERE).parts and f.name not in ['artifact_manifest.json','delivery_verification.json'])
    for f in files:
        assert '__pycache__' not in f.parts and f.suffix!='.pyc'
    entries=[dict(path=str(f.relative_to(HERE)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]
    manifest=dict(scope='phase4 research payload; excludes manifest itself, archives/ and delivery_verification.json to avoid recursive archive hashes',total_files=len(entries),total_bytes=sum(z['bytes'] for z in entries),files=entries)
    m=HERE/'artifact_manifest.json';m.write_text(json.dumps(manifest,indent=2));files.append(m)
    out=HERE/'archives';out.mkdir(exist_ok=True);dest=out/'plan_r1_phase4_research.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(files):
            name='phase4/'+str(f.relative_to(HERE));info=zipfile.ZipInfo(name,date_time=(2026,10,20,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,f.read_bytes())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
        for f in files:assert z.read('phase4/'+str(f.relative_to(HERE)))==f.read_bytes()
    result=dict(zip_path=str(dest.relative_to(HERE)),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),entries=len(files),crc_verified=True,all_entries_identical_to_local_payload=True,payload_files_excluding_manifest=len(entries),payload_bytes_excluding_manifest=manifest['total_bytes'],phase4_only=True,no_checkpoint_or_large_dataset=True,base_commit='fd7d0364c97c2bc77423e8aeaec5699a4dfd32f7',note='Delivery-only verification; no research/training rerun. ZIP excludes itself and this delivery metadata.')
    (HERE/'delivery_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
