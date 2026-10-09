"""Recover audited placement for the supplied samples without changing source data."""
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
from affine import Affine
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json,sha256
from pipelines.semantic_temporal.runtime.inputs import inspect_inputs


def prepare_samples(project_root: Path,output: Path) -> dict:
    root=Path(project_root).resolve();output=Path(output).resolve()
    manifest_path=root/'pipelines/semantic_temporal/reference/input_manifest.json';manifest=read_json(manifest_path)
    folders=[root/'tiles17_23'/str(y) for y in (2013,2023)]
    source_paths=[manifest_path];image_records=[]
    expected_names={f'tile_{i:05d}.png' for i in range(7)}
    for year,folder in zip((2013,2023),folders):
        if {p.name for p in folder.glob('*.png')}!=expected_names:raise ValueError('Sample recovery requires exactly the seven verified images per year')
        for name in sorted(expected_names):
            path=folder/name;digest=sha256(path);expected=manifest.get(f'data/dataset/{year}/{name}')
            if not expected or digest!=expected:raise ValueError(f'Unverified sample hash: {path}')
            source_paths.append(path);image_records.append({'year':year,'name':name,'sha256':digest})
    original_csv=[folder/'metadata.csv' for folder in folders];source_paths.extend(original_csv)
    with original_csv[0].open(newline='',encoding='utf-8-sig') as stream:original=list(csv.DictReader(stream))
    if {r['tile_name'] for r in original}!=expected_names or len(original)!=7:raise ValueError('Sample metadata must contain seven unique tiles')
    recovered=[];saved_records=[];source_transform=None;source_crs=None
    for row in sorted(original,key=lambda r:r['tile_name']):
        name=row['tile_name'];saved_path=root/'artifacts/annual_2013_updated/stage2'/f'shouf_2013_epsg32636_1p5m__{Path(name).stem}'/'13_refinement/tile_metadata.json'
        saved=read_json(saved_path);source_paths.append(saved_path)
        source=Affine(*saved['source_affine_transform']);tile=Affine(*saved['tile_affine_transform'])
        x,y=int(row['x0']),int(row['y0'])
        with Image.open(folders[0]/name) as image:w,h=image.size
        if saved['global_x0']!=x or saved['global_y0']!=y or saved['width']!=w or saved['height']!=h:
            raise ValueError(f'Inconsistent saved coordinates or dimensions: {name}')
        expected=source*Affine.translation(x,y)
        if not np.allclose(tuple(tile)[:6],tuple(expected)[:6],atol=1e-7,rtol=0):raise ValueError(f'Inconsistent saved coordinates: {name}')
        if source_transform is not None and (not np.allclose(tuple(source)[:6],source_transform,atol=1e-7,rtol=0) or saved['crs']!=source_crs):
            raise ValueError('Inconsistent source coordinates/CRS across saved tiles')
        source_transform=tuple(source)[:6];source_crs=saved['crs']
        if saved.get('projected_bounds'):
            points=np.array([tile*p for p in ((0,0),(w,0),(w,h),(0,h))]);bounds=[*points.min(0),*points.max(0)]
            if not np.allclose(bounds,saved['projected_bounds'],atol=1e-7,rtol=0):raise ValueError(f'Inconsistent saved projected coordinates: {name}')
        recovered.append({**row,**{f'tile_affine_{key}':value for key,value in zip('abcdef',tuple(tile)[:6])},'target_crs':saved['crs']})
        saved_records.append({'image':name,'saved_metadata':str(saved_path),'sha256':sha256(saved_path),'tile_affine':tuple(tile)[:6]})
    # Validate the actual 2023 source-window pixels before writing derived inputs.
    verified_2023=inspect_inputs({'folders':[str(folders[1])],'years':[2023]},root)[0]
    if verified_2023.source_raster is None:raise ValueError('2023 sample audit requires its georeferenced RGB source raster')
    source_paths.append(verified_2023.source_raster)
    before={str(path):sha256(path) for path in source_paths}
    if output.is_relative_to(folders[0]) or output.is_relative_to(folders[1]):raise ValueError('Derived sample metadata must be outside the original sample folders')
    output.mkdir(parents=True,exist_ok=True);metadata=[output/'2013_metadata.csv',output/'2023_metadata.csv']
    fields=list(dict.fromkeys(key for row in recovered for key in row))
    with metadata[0].open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(recovered)
    metadata[1].write_bytes(original_csv[1].read_bytes())
    inspect_inputs({'folders':[str(p) for p in folders],'years':[2013,2023],'metadata':[str(p) for p in metadata]},root)
    if any(sha256(Path(path))!=digest for path,digest in before.items()):raise ValueError('Source changed during sample preparation')
    audit=output/'placement_audit.json'
    atomic_json(audit,{'status':'verified','image_count':14,'images':image_records,'saved_2013_records':saved_records,
        'source_hashes':before,'derived_hashes':{str(path):sha256(path) for path in metadata},
        '2013_status':'Placement recovered from saved metadata and verified PNG hashes; original source raster is unavailable.',
        '2023_status':'RGB pixels verified against source windows; georeferencing accuracy is not independently established.',
        'original_source_raster_created':False})
    return {'folders':[str(p) for p in folders],'metadata':[str(p) for p in metadata],'audit':str(audit)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import json
    print(json.dumps(prepare_samples(args.project_root,args.output),indent=2))
