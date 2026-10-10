import pytest
from tests.semantic_fixtures import make_year


def inspect(options, tmp_path):
    from pipelines.semantic_temporal.runtime.inputs import inspect_inputs
    return inspect_inputs(options, tmp_path)


def test_location_metadata_and_mask_are_preserved(tmp_path):
    from pipelines.semantic_temporal.runtime.inputs import read_tile
    folder = make_year(tmp_path, alpha=True)
    tile = inspect({'folders':[str(folder)]}, tmp_path)[0].tiles[0]
    assert tile.offset_xy == (3, 2)
    assert tile.shape_hw == (12, 18)
    assert tile.grid.affine == (1.5, 0.0, 500004.5, 0.0, -1.5, 3699997.0)
    rgb, valid = read_tile(tile)
    assert rgb.shape == (12, 18, 3)
    assert valid.sum() == 180


def test_single_image_does_not_process_other_tiles(tmp_path):
    folder = make_year(tmp_path, names=('a.png','b.png'), offsets=((3,2),(30,2)))
    years = inspect({'images':[str(folder/'b.png')]}, tmp_path)
    assert len(years[0].tiles) == 1
    assert years[0].tiles[0].offset_xy == (30, 2)


def test_missing_and_duplicate_metadata_are_rejected(tmp_path):
    folder = make_year(tmp_path)
    meta = folder/'metadata.csv'
    original = meta.read_text(); meta.unlink()
    with pytest.raises(ValueError, match='metadata'): inspect({'folders':[str(folder)],'metadata':[str(meta)]}, tmp_path)
    meta.write_text(original + original.splitlines()[1] + '\n')
    with pytest.raises(ValueError, match='duplicate'): inspect({'folders':[str(folder)]}, tmp_path)


def test_png_metadata_without_map_retains_original_pixel_coordinates(tmp_path):
    folder = make_year(tmp_path)
    (folder/'source.tif').unlink()
    tile = inspect({'folders':[str(folder)], 'pixel_size':[1.5]}, tmp_path)[0].tiles[0]
    assert tile.grid.crs is None
    assert tile.grid.units == 'pixels'
    assert tile.grid.affine[2] == 3
    assert tile.grid.physical_pixel_size == (1.5, 1.5)


def test_actual_years_and_differing_tile_names_are_supported(tmp_path):
    first = make_year(tmp_path, 2005)
    second = make_year(tmp_path, 2024, names=('other.png','extra.png'), offsets=((3,2),(30,2)))
    years = inspect({'folders':[str(first),str(second)]}, tmp_path)
    assert [year.year for year in years] == [2005,2024]
    assert [len(year.tiles) for year in years] == [1,2]
    assert years[0].tiles[0].key != years[1].tiles[0].key


def test_pixel_window_disagreement_is_rejected(tmp_path):
    folder = make_year(tmp_path)
    meta = folder/'metadata.csv'; meta.write_text(meta.read_text().replace('a.png,3,2','a.png,4,2'))
    with pytest.raises(ValueError, match='pixels'): inspect({'folders':[str(folder)]}, tmp_path)


def test_ambiguous_rasters_require_override(tmp_path):
    import shutil
    folder=make_year(tmp_path);shutil.copyfile(folder/'source.tif',folder/'other.tif')
    with pytest.raises(ValueError,match='Ambiguous'): inspect({'folders':[str(folder)]},tmp_path)
    assert inspect({'folders':[str(folder)],'source_rasters':[str(folder/'source.tif')]},tmp_path)


@pytest.mark.parametrize('bad_row',['a.png,-1,2,18,12','a.png,3,2,17,12','a.png,90,60,18,12'])
def test_invalid_original_extents_are_rejected(tmp_path,bad_row):
    folder=make_year(tmp_path)
    (folder/'metadata.csv').write_text('tile_name,x0,y0,width,height\n'+bad_row+'\n')
    with pytest.raises(ValueError): inspect({'folders':[str(folder)]},tmp_path)


def test_direct_uint16_geotiff_has_recorded_grid_and_full_range_scaling(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from pipelines.semantic_temporal.runtime.inputs import read_tile
    folder=tmp_path/'custom name';folder.mkdir();image=folder/'rgb.tif'
    values=np.full((3,5,7),32768,np.uint16);values[:,0,0]=0
    with rasterio.open(image,'w',driver='GTiff',height=5,width=7,count=3,dtype='uint16',nodata=0,crs='EPSG:32636',transform=from_origin(500000,3700000,2,2)) as dst: dst.write(values)
    tile=inspect({'images':[str(image)],'years':[2019]},tmp_path)[0].tiles[0]
    rgb,valid=read_tile(tile)
    assert tile.shape_hw==(5,7) and tile.grid.physical_pixel_size==(2,2)
    assert (rgb[1:]==128).all() and not valid[0,0]


def test_more_than_three_tiles_and_chronological_years(tmp_path):
    first=make_year(tmp_path,2008,names=('a.png','b.png','c.png','d.png'),offsets=((3,2),(23,2),(43,2),(63,2)))
    second=make_year(tmp_path,2024)
    assert len(inspect({'folders':[str(first),str(second)]},tmp_path)[0].tiles)==4
    with pytest.raises(ValueError,match='chronological'): inspect({'folders':[str(second),str(first)]},tmp_path)


def test_context_records_resolved_custom_model_hashes(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.storage import sha256
    folder=make_year(tmp_path);model=tmp_path/'my-model.pt';model.write_bytes(b'custom model')
    context=build_context({'folders':[str(folder)],'project_root':str(tmp_path),
                          'old_config':{'stage2':{'instance_model':{'backend':'sam2','checkpoint':str(model)}}}},tmp_path/'run')
    assert context.models['old.instance_model.checkpoint']['path']==str(model)
    assert context.models['old.instance_model.checkpoint']['sha256']==sha256(model)


def test_new_context_can_report_missing_old_configuration(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    folder=make_year(tmp_path)
    context=build_context({'folders':[str(folder)],'project_root':str(tmp_path),
                          'config':str(tmp_path/'absent-old-config.yaml')},tmp_path/'run')
    assert context.models['old.configuration']['status']=='missing'
