from pathlib import Path
import csv
import numpy as np
import pytest
import rasterio
from affine import Affine
from tests.semantic_fixtures import make_year
from pipelines.semantic_temporal.runtime.storage import sha256,atomic_json,read_json


def sample_fixture(root):
    names=tuple(f'tile_{i:05d}.png' for i in range(7));offsets=tuple((i*8,0) for i in range(7));manifest={}
    for year in (2013,2023):
        folder=make_year(root/'tiles17_23',year=year,names=names,offsets=offsets)
        for name in names:manifest[f'data/dataset/{year}/{name}']=sha256(folder/name)
        if year==2013:
            with rasterio.open(folder/'source.tif') as src:source=src.transform
            for name,(x,y) in zip(names,offsets):
                path=root/'artifacts/annual_2013_updated/stage2'/f'shouf_2013_epsg32636_1p5m__{Path(name).stem}'/'13_refinement/tile_metadata.json'
                atomic_json(path,{'global_x0':x,'global_y0':y,'width':18,'height':12,'crs':'EPSG:32636',
                    'source_affine_transform':list(source)[:6],'tile_affine_transform':list(source*Affine.translation(x,y))[:6]})
    atomic_json(root/'pipelines/semantic_temporal/reference/input_manifest.json',manifest)
    return root


def test_recovered_placement_matches_saved_source(tmp_path):
    from scripts.prepare_2013_2023_samples import prepare_samples
    root=sample_fixture(tmp_path/'project');result=prepare_samples(root,tmp_path/'derived')
    with Path(result['metadata'][0]).open(newline='') as stream:rows=list(csv.DictReader(stream))
    for i,row in enumerate(rows):
        saved=read_json(root/'artifacts/annual_2013_updated/stage2'/f'shouf_2013_epsg32636_1p5m__tile_{i:05d}'/'13_refinement/tile_metadata.json')
        tile_affine=Affine(*(float(row['tile_affine_'+key]) for key in 'abcdef'))
        source_affine=Affine(*saved['source_affine_transform']);x0=int(row['x0']);y0=int(row['y0'])
        np.testing.assert_allclose(tuple(tile_affine)[:6],tuple(source_affine*Affine.translation(x0,y0))[:6])
    from pipelines.semantic_temporal.runtime.inputs import inspect_inputs
    years=inspect_inputs({'folders':result['folders'],'years':[2013,2023],'metadata':result['metadata']},root)
    assert all(len(year.tiles)==7 for year in years)


def test_recovery_never_modifies_sources(tmp_path):
    from scripts.prepare_2013_2023_samples import prepare_samples
    root=sample_fixture(tmp_path/'project');paths=[p for p in root.rglob('*') if p.is_file()]
    source_hashes_before={p:sha256(p) for p in paths}
    prepare_samples(root,tmp_path/'derived')
    source_hashes_after={p:sha256(p) for p in paths}
    assert source_hashes_after==source_hashes_before


@pytest.mark.parametrize('damage',['hash','coordinates'])
def test_recovery_rejects_unverified_samples(tmp_path,damage):
    from scripts.prepare_2013_2023_samples import prepare_samples
    root=sample_fixture(tmp_path/'project')
    if damage=='hash':
        with (root/'tiles17_23/2013/tile_00000.png').open('ab') as stream:stream.write(b'changed')
    else:
        path=root/'artifacts/annual_2013_updated/stage2/shouf_2013_epsg32636_1p5m__tile_00000/13_refinement/tile_metadata.json'
        record=read_json(path);record['tile_affine_transform'][2]+=2;atomic_json(path,record)
    with pytest.raises(ValueError,match='hash|coordinates'):prepare_samples(root,tmp_path/'derived')
