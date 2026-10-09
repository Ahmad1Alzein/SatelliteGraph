"""Atomic descriptions and masked rasters shared by all runtime stages."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import time
import uuid
import numpy as np
import rasterio
from affine import Affine


def json_value(value):
    if is_dataclass(value): return {k:json_value(v) for k,v in asdict(value).items()}
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    return value


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False), encoding='utf-8')
    try:
        for attempt in range(10):
            try:temporary.replace(path);break
            except PermissionError:
                if attempt==9:raise
                time.sleep(min(.05*2**attempt,.25))
    finally:
        if temporary.exists():temporary.unlink()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write_raster(path, array, grid, *, nodata=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = array if array.ndim == 3 else array[None]
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=data.dtype, crs=grid.crs,
                       transform=Affine(*grid.affine), nodata=nodata, compress='deflate') as out:
        out.write(data)


def save_manifest(directory, paths, **info):
    paths = dict(paths)
    description = Path(directory)/'artifact.json'
    if description.is_file(): paths['description'] = description
    atomic_json(Path(directory)/'manifest.json', {**info, 'files':{
        str(Path(p).relative_to(directory)):sha256(p) for p in paths.values() if Path(p).is_file()}})


def verify_manifest(directory):
    directory = Path(directory)
    for name, digest in read_json(directory/'manifest.json')['files'].items():
        p = (directory/name).resolve()
        if not p.is_relative_to(directory.resolve()) or not p.is_file() or sha256(p)!=digest:
            raise ValueError(f'Artifact changed or missing: {name}')
    provenance=directory/'checkpoint_provenance.json'
    if provenance.is_file():
        value=read_json(provenance);checkpoint=Path(value['checkpoint'])
        for path,digest in ((checkpoint,value['sha256']),(checkpoint.with_suffix('.json'),value['metadata_sha256'])):
            if not path.is_file() or sha256(path)!=digest: raise ValueError(f'External checkpoint changed or missing: {path}')


def load_segmentation(path):
    from .contracts import SegmentationArtifact,GridSpec
    value=read_json(path);grid=value['grid']
    grid=GridSpec(grid['crs'],tuple(grid['affine']),grid['units'],tuple(grid['physical_pixel_size']) if grid['physical_pixel_size'] else None)
    return SegmentationArtifact(value['year'],value['tile_key'],Path(value['directory']),grid,tuple(value['shape_hw']),{k:Path(v) for k,v in value['paths'].items()},value['lineage'])


def bind_result_manifest(output,stage,paths,inputs):
    output=Path(output).resolve()
    def record(path):
        path=Path(path).resolve();external=not path.is_relative_to(output)
        return {'path':str(path if external else path.relative_to(output)),'external':external,'sha256':sha256(path)}
    destination=output/'results/rendering'/stage/'manifest.json'
    atomic_json(destination,{'stage':stage,'outputs':[record(p) for p in paths.values() if Path(p).is_file()],
        'inputs':[record(p) for p in sorted(set(map(Path,inputs))) if p.is_file()]})


def verify_result_manifest(output,stage):
    output=Path(output).resolve();value=read_json(output/'results/rendering'/stage/'manifest.json')
    for item in value['outputs']+value['inputs']:
        path=Path(item['path']) if item['external'] else output/item['path']
        if not item['external'] and not path.resolve().is_relative_to(output):raise ValueError('Result manifest path escapes run')
        if not path.is_file() or sha256(path)!=item['sha256']:raise ValueError(f'Result artifact changed or missing: {path}')
    return value
