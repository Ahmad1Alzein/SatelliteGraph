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


def test_original_pixel_controls_use_physical_scale_without_fake_crs(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison,prepare_tile
    first=make_year(tmp_path);second=make_year(tmp_path,2023)
    for folder in (first,second): (folder/'source.tif').unlink()
    points=np.array([[0,0],[30,0],[0,30],[30,30],[60,0],[0,60]],float)
    controls=tmp_path/'controls.json';controls.write_text(json.dumps({'moving_xy':points.tolist(),'reference_xy':points.tolist()}))
    context=prepare_comparison(build_context({'folders':[str(first),str(second)],'pixel_size':[1.5,1.5],'registration_json':str(controls)},tmp_path/'run'))
    rgb,valid,grid=prepare_tile(context.years[0].tiles[0],context)
    assert grid.crs is None and grid.units=='metres'
    assert grid.affine[2:]==(4.5,0.,1.5,3.)


def test_geographic_and_rotated_sources_prepare_metric_axis_aligned_grids(tmp_path):
    import rasterio
    from affine import Affine
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison,prepare_tile
    folder=make_year(tmp_path)
    with rasterio.open(folder/'source.tif','r+') as src:
        src.crs='EPSG:4326';src.transform=Affine(.000015,.000001,35.5,.000001,-.000015,33.8)
    context=prepare_comparison(build_context({'folders':[str(folder)]},tmp_path/'run'))
    rgb,valid,grid=prepare_tile(context.years[0].tiles[0],context)
    assert grid.units=='metres' and grid.crs=='EPSG:32636'
    assert grid.affine[1]==grid.affine[3]==0 and valid.any()


def test_current_image_registration_recovers_measured_shift(tmp_path):
    import cv2,rasterio
    from PIL import Image
    from rasterio.transform import from_origin
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    rng=np.random.default_rng(27);image=rng.integers(25,230,(192,256,3),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(3,3),0)
    shifted=cv2.warpAffine(image,np.array([[1,0,3],[0,1,-2]],np.float32),(256,192))
    folders=[]
    for year,array,name in ((2017,image,'first.png'),(2023,shifted,'different.png')):
        folder=tmp_path/str(year);folder.mkdir();folders.append(str(folder))
        Image.fromarray(array).save(folder/name)
        (folder/'metadata.csv').write_text(f'tile_name,x0,y0\n{name},0,0\n')
        with rasterio.open(folder/'source.tif','w',driver='GTiff',height=192,width=256,count=3,dtype='uint8',crs='EPSG:32636',transform=from_origin(500000,3700000,1.5,1.5)) as dst: dst.write(array.transpose(2,0,1))
    context=prepare_comparison(build_context({'folders':folders,'register_images':True},tmp_path/'run'))
    affine=np.array(context.options['registration']['affine']).reshape(2,3)
    points=np.array([[500000,3700000],[500384,3700000],[500000,3699712],[500384,3699712]],float)
    displacement=points@affine[:,:2].T+affine[:,2]-points
    np.testing.assert_allclose(displacement,np.tile([4.5,3],(4,1)),atol=.12)
