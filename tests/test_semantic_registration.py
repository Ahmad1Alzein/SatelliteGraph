from pathlib import Path
import json
import numpy as np
from tests.semantic_fixtures import make_year


def test_shared_grid_retains_actual_offsets_and_unobserved_mask(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison, prepare_tile
    first=make_year(tmp_path,alpha=True)
    second=make_year(tmp_path,2023,names=('different.png',),alpha=True)
    context=prepare_comparison(build_context({'folders':[str(first),str(second)]},tmp_path/'run'))
    rgb,valid,grid=prepare_tile(context.years[0].tiles[0],context)
    assert rgb.shape==(12,18,3)
    assert valid.sum()==180
    assert grid.affine[2]==500004.5
    assert grid.physical_pixel_size==(1.5,1.5)


def test_supplied_registration_changes_grid_not_original_pixels(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison,prepare_tile
    first=make_year(tmp_path); second=make_year(tmp_path,2023)
    controls=tmp_path/'controls.json'
    moving=np.array([[0,0],[10,0],[0,10],[10,10],[20,0],[0,20]],float)+[500000,3700000]
    controls.write_text(json.dumps({'moving_xy':moving.tolist(),'reference_xy':(moving+[3,-1.5]).tolist()}))
    context=prepare_comparison(build_context({'folders':[str(first),str(second)],'registration_json':str(controls)},tmp_path/'run'))
    _,_,grid=prepare_tile(context.years[0].tiles[0],context)
    assert np.isclose(grid.affine[2],500007.5)
    assert np.isclose(grid.affine[5],3699995.5)


def test_coverage_is_geometric_not_based_on_filenames(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison,coverage_pairs
    first=make_year(tmp_path)
    second=make_year(tmp_path,2023,names=('other.png','far.png'),offsets=((3,2),(70,40)))
    context=prepare_comparison(build_context({'folders':[str(first),str(second)]},tmp_path/'run'))
    assert coverage_pairs(context)==[('2017/a.png','2023/other.png')]


def test_requested_shared_resolution_changes_shape_and_preserves_footprint(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison,prepare_tile
    folder=make_year(tmp_path)
    context=prepare_comparison(build_context({'folders':[str(folder)],'target_resolution':.75},tmp_path/'run'))
    rgb,valid,grid=prepare_tile(context.years[0].tiles[0],context)
    assert rgb.shape==(24,36,3)
    assert grid.affine[0]==.75
    assert grid.affine[2]==500004.5
