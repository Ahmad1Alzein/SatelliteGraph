import numpy as np
from PIL import Image
from shapely.geometry import box, mapping

from tests.test_change_report_pngs import report_fixture


def test_small_event_keeps_local_context_and_limits_enlargement(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.changes import event_crops
    _, _, frame, catalog = report_fixture(tmp_path, 1)
    frame['source_pixel_size'] = 1.5
    _, record = event_crops(frame, catalog['events'][0])
    bounds = record['before_bounds']
    assert bounds == record['after_bounds']
    assert bounds[2] - bounds[0] >= 224 * 1.5 / 2
    assert record['maximum_enlargement'] <= 2


def test_local_crop_samples_native_rgb_instead_of_blurred_comparison(tmp_path):
    from affine import Affine
    from pipelines.semantic_temporal.runtime.reports.changes import event_crops
    _, _, frame, catalog = report_fixture(tmp_path, 1)
    rgb = np.zeros((400, 400, 3), np.uint8)
    rgb[:, 100:200] = (230, 40, 30)
    frame['source_pixel_size'] = 1
    for year in frame['years']:
        frame['data'][year]['rgb_original'][:] = 17
        frame['data'][year]['native_sources'] = [dict(rgb=rgb, valid=np.ones((400, 400), bool),
            affine=Affine(1, 0, 0, 0, -1, 400), crs=None, target_crs=None, registration=Affine.identity())]
    event = dict(catalog['events'][0], geometry=mapping(box(145, 195, 155, 205)))
    panels, record = event_crops(frame, event)
    pixels = np.asarray(panels[0])
    assert (pixels[:, :, 0] == 230).sum() > 1000
    assert record['rgb_source'] == 'native_original'
    assert record['before_bounds'][2] - record['before_bounds'][0] < 400


def test_bare_images_and_filename_row_need_no_extra_data(tmp_path):
    from pipelines.semantic_temporal.runtime.inputs import inspect_inputs
    first = tmp_path / 'before'; second = tmp_path / 'after'
    for folder in (first, second):
        folder.mkdir()
        for name in ('tile_10.png', 'tile_2.png', 'tile_1.png'):
            Image.new('RGB', (30, 20), 'gray').save(folder / name)
    years = inspect_inputs({'images': [str(first/'tile_1.png'), str(second/'tile_1.png')]}, tmp_path)
    assert [y.year for y in years] == [1, 2]
    assert all(y.tiles[0].metadata is None and y.tiles[0].grid.units == 'pixels' for y in years)
    row = inspect_inputs({'folders': [str(first)]}, tmp_path)[0].tiles
    assert [t.image.name for t in row] == ['tile_1.png', 'tile_2.png', 'tile_10.png']
    assert [t.offset_xy for t in row] == [(0, 0), (30, 0), (60, 0)]


def test_pixel_registration_detects_translation(tmp_path):
    import cv2
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    rng = np.random.default_rng(8)
    rgb = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
    rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
    paths = [tmp_path/'before.png', tmp_path/'after.png']
    Image.fromarray(rgb).save(paths[0])
    Image.fromarray(cv2.warpAffine(rgb, np.float32([[1,0,9],[0,1,5]]), (240,180))).save(paths[1])
    context = prepare_comparison(build_context({'images':list(map(str, paths)), 'project_root':str(tmp_path)}, tmp_path/'run'))
    np.testing.assert_allclose(np.asarray(context.options['registration']['affine']).reshape(2,3)[:,2], [9,5], atol=.5)
    assert context.options['coordinate_units'] == 'pixels'


def test_automatic_boundary_margin_uses_measured_alignment_residual():
    from types import SimpleNamespace
    from pipelines.semantic_temporal.runtime.temporal import boundary_margin
    context=SimpleNamespace(options={'registration':{'units':'pixels','inlier_p95_pixels':1.75}})
    assert boundary_margin(context)==1.75
    context.options['boundary_tolerance']=0
    assert boundary_margin(context)==0


def test_candidate_caption_does_not_assert_demolition():
    from pipelines.semantic_temporal.runtime.reports.changes import _event_text
    event=dict(event_id='weak',kind='conversion',source_class=1,target_class=4,class_id=4,
        status='season_sensitive',area=50,descriptors={'building_loss':['disappearance']})
    lines,_=_event_text(event,{})
    assert 'Building disappeared' not in lines[0]
    assert 'class-label difference' in lines[0].lower()


def test_pixel_pair_registration_accepts_different_image_resolutions(tmp_path):
    import cv2
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    rgb=np.random.default_rng(18).integers(0,256,(200,240,3),dtype=np.uint8)
    rgb=cv2.GaussianBlur(rgb,(5,5),0)
    paths=[tmp_path/'before.png',tmp_path/'after.png']
    Image.fromarray(rgb).save(paths[0])
    Image.fromarray(cv2.resize(rgb,None,fx=2,fy=2)).save(paths[1])
    context=prepare_comparison(build_context({'images':list(map(str,paths)),'project_root':str(tmp_path)},tmp_path/'run'))
    matrix=np.asarray(context.options['registration']['affine']).reshape(2,3)
    np.testing.assert_allclose(matrix[:,:2],np.eye(2)*2,atol=.03)
    assert context.options['registration']['units']=='pixels'


def test_native_preview_respects_scientific_tile_ownership():
    from affine import Affine
    from pipelines.semantic_temporal.runtime.contracts import GridSpec
    from pipelines.semantic_temporal.runtime.reports.common import sample_original_rgb
    owner=np.zeros((20,20),bool);owner[:,:15]=True
    sources=[]
    for x,color,ownership in [(0,(255,0,0),owner),(10,(0,0,255),~np.pad(owner[:,10:],((0,0),(0,10))))]:
        rgb=np.empty((20,20,3),np.uint8);rgb[:]=color
        sources.append(dict(rgb=rgb,valid=np.ones((20,20),bool),affine=Affine.translation(x,0),
            crs=None,target_crs=None,registration=Affine.identity(),ownership=ownership,
            ownership_affine=Affine.translation(x,0)))
    frame={'grid':GridSpec(None,(1,0,0,0,1,0),'pixels',None),'data':{1:{'native_sources':sources}}}
    pixels,valid=sample_original_rgb(frame,1,(17,5,19,7),(2,2))
    assert valid.all()
    assert np.all(pixels==(0,0,255))


def test_browser_and_png_share_cautious_prediction_labels():
    from pipelines.semantic_temporal.runtime.preview import event_rows
    from pipelines.semantic_temporal.runtime.reports.changes import _event_text
    event=dict(event_id='gnn:x',kind='gain',class_id=1,status='predicted',area=20,geometry=mapping(box(0,0,4,5)))
    row=event_rows([event],{'nodes':[]},{})[0]
    title=_event_text(event,{})[0][0]
    assert row['type']==title
    assert 'New building' not in row['type']
