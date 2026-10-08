from pathlib import Path
import importlib.util
import importlib.metadata
from .storage import read_json,sha256


def check_environment(context=None):
    root=Path(context.options.get('project_root') or Path(__file__).resolve().parents[3]) if context else Path(__file__).resolve().parents[3]
    packages={'numpy':'numpy','pandas':'pandas','scipy':'scipy','cv2':'opencv-python','rasterio':'rasterio','shapely':'shapely',
              'torch':'torch','torchvision':'torchvision','transformers':'transformers','ultralytics':'ultralytics','safetensors':'safetensors','networkx':'networkx',
              'sam2':'SAM-2','hydra':'hydra-core','iopath':'iopath','easydict':'easydict','yaml':'PyYAML','PIL':'Pillow','affine':'affine','tqdm':'tqdm'}
    libraries={}
    for module,distribution in packages.items():
        available=importlib.util.find_spec(module) is not None
        try: version=importlib.metadata.version(distribution) if available else None
        except importlib.metadata.PackageNotFoundError: version='local'
        libraries[module]={'available':available,'version':version}
    manifest=read_json(Path(__file__).resolve().parents[1]/'model_assets.json');models=[]
    for asset in manifest['assets']:
        path=root/'checkpoints'/asset['path']
        if context and asset['method']=='new' and context.options.get('semantic_models'):
            path=Path(context.options['semantic_models'])/Path(asset['path']).relative_to('semantic_temporal')
        status='missing' if not path.is_file() else 'verified' if path.stat().st_size==asset['size'] and sha256(path)==asset['sha256'] else 'hash_mismatch'
        models.append({**asset,'resolved_path':str(path),'status':status})
    if libraries['torch']['available']:
        import torch
        cuda=torch.cuda.is_available();cuda_build=torch.version.cuda
    else: cuda=False;cuda_build=None
    ready=all(lib['available'] for lib in libraries.values()) and all(m['status']=='verified' for m in models)
    return {'status':'ready' if ready else 'needs_setup','libraries':libraries,'models':models,
            'cuda_available':cuda,'torch_cuda_build':cuda_build,'default_device':'cpu',
            'general_requires_osgeo':False,'reproduction_requires_osgeo':True,'gnn_checkpoint':'train your own compatible checkpoint; no universal pretrained change model',
            'setup_command':'python -m pip install -r requirements.txt','download_command':'python scripts/download_pipeline_models.py'}
