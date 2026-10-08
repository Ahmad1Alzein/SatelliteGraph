"""Acquire pinned model files atomically; preserve differing existing assets."""
from pathlib import Path
import argparse
import json
import hashlib
import urllib.request


def assets():
    return json.loads((Path(__file__).resolve().parents[1]/'pipelines/semantic_temporal/model_assets.json').read_text())['assets']


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): result.update(chunk)
    return result.hexdigest()


def ensure_models(destination,*,methods=('old','new')):
    destination=Path(destination).resolve();report={'assets':[]}
    for asset in assets():
        if asset['method'] not in methods: continue
        path=(destination/asset['path']).resolve()
        if not path.is_relative_to(destination): raise ValueError('Model asset escapes destination')
        if path.exists():
            if path.stat().st_size!=asset['size'] or digest(path)!=asset['sha256']:
                raise ValueError(f'Existing differing checkpoint preserved: {path}; move it aside explicitly or choose a new destination')
            status='verified_existing'
        else:
            if not asset.get('url'): raise ValueError(f'No verified download URL for {path}; consult the model publisher')
            path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_name(path.name+'.part')
            try:
                request=urllib.request.Request(asset['url'],headers={'User-Agent':'SatelliteGraph-model-download/1.0'})
                with urllib.request.urlopen(request,timeout=120) as source,partial.open('wb') as target:
                    while chunk:=source.read(1024*1024): target.write(chunk)
                if partial.stat().st_size!=asset['size'] or digest(partial)!=asset['sha256']:
                    raise ValueError(f'Model size/hash verification failed: {asset["path"]}; no checkpoint promoted')
                partial.replace(path);status='downloaded'
            finally:
                if partial.exists(): partial.unlink()
        report['assets'].append({**asset,'status':status,'resolved_path':str(path)})
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'model_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',default=str(Path(__file__).resolve().parents[1]/'checkpoints'))
    parser.add_argument('--methods',nargs='+',choices=('old','new'),default=['old','new'])
    args=parser.parse_args();print(json.dumps(ensure_models(args.destination,methods=tuple(args.methods)),indent=2))
