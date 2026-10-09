from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from pipelines.semantic_temporal.runtime.contracts import RunContext
from pipelines.semantic_temporal.runtime.storage import read_json
from tests.test_semantic_graphs import make_artifact


def stage_fixture(root,shape=(9,18),unknown=False):
    context=RunContext(Path(root)/'run',(),{}, {},'fixture')
    labels=np.ones(shape,np.int32);classes=np.full(shape,0 if unknown else 3,np.uint8)
    artifacts=[make_artifact(context.output/'segmentation',labels,classes,year=y,key='synthetic-layout-test-long-image-name') for y in (2013,2023)]
    return context,artifacts


def test_separate_stage_pngs_are_retained(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.stages import render_stage_results
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    context,artifacts=stage_fixture(tmp_path)
    segmentation=render_stage_results(context,'segmentation')
    paths=[p for p in segmentation.values() if p.suffix=='.png']
    assert len(paths)==2 and all(p.is_file() for p in paths)
    before={p:p.read_bytes() for p in paths}
    for a in artifacts:combine_year_graphs([build_tile_graph(a,'new',context)],context)
    graphs=render_stage_results(context,'graph')
    assert len([p for p in graphs.values() if p.suffix=='.png'])==4
    assert all(p.read_bytes()==before[p] for p in paths)
    index=read_json(context.output/'results/graphs/index.json')
    for record in index['images']:
        assert 'Trees' in record['class_legend'] and 'No image data' not in record['class_legend']
        assert record['year'] in (2013,2023)
        assert record['edge_count']==0


def test_tiny_image_legend_is_outside_pixels(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.stages import render_stage_results
    context,_=stage_fixture(tmp_path,shape=(2,7))
    render_stage_results(context,'segmentation')
    index=read_json(context.output/'results/segmentation/index.json')
    for record in index['images']:
        with Image.open(context.output/record['path']) as rendered:
            assert rendered.height>2
            assert record['legend_bounds'][1]>=record['image_bounds'][3]
            assert record['legend_bounds'][3]<=rendered.height
            assert record['legend_bounds'][2]<=rendered.width
        assert 'Unknown' in record['class_legend']


def test_rgb_sampling_keeps_original_landmark_and_missing_pixels():
    from pipelines.semantic_temporal.runtime.reports.common import sample_rgb
    from tests.test_all_class_evidence import frame_for
    frame=frame_for(np.ones((12,18),np.uint8),np.ones((12,18),np.uint8))
    frame['data'][2013]['rgb_original'][4,0]=[220,10,15]
    frame['data'][2013]['valid'][0]=False
    rgb,valid=sample_rgb(frame,2013,(0,88,18,100),(18,12))
    assert np.array_equal(rgb[4,0],[220,10,15])
    assert not valid[0].any()


def test_geometry_hole_preserves_underlying_rgb():
    from pipelines.semantic_temporal.runtime.reports.common import paint_geometry
    image=Image.new('RGB',(30,20),(100,100,100))
    polygon=Polygon([(0,0),(30,0),(30,20),(0,20)],holes=[[(10,5),(20,5),(20,15),(10,15)]])
    paint_geometry(image,polygon,(0,0,30,20),(230,20,20),alpha=100)
    assert image.getpixel((15,10))==(100,100,100)
    assert image.getpixel((3,3))!=(100,100,100)


def test_tiny_panel_caption_has_room_to_remain_readable():
    from pipelines.semantic_temporal.runtime.reports.common import horizontal_panels,font
    from PIL import ImageDraw
    caption='Segments and boundaries'
    body=horizontal_panels([Image.new('RGB',(7,2))],[caption])
    required=ImageDraw.Draw(body).textlength(caption,font=font())
    assert body.width>=required+8


def test_graph_legend_uses_readable_spatial_edge_names(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.stages import render_stage_results
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    context,_=stage_fixture(tmp_path)
    labels=np.ones((10,20),np.int32);labels[:,10:]=2
    a=make_artifact(context.output/'segmentation',labels,np.where(labels==1,2,3).astype(np.uint8),year=2013,key='edges')
    combine_year_graphs([build_tile_graph(a,'new',context)],context)
    render_stage_results(context,'graph')
    index=read_json(context.output/'results/graphs/index.json')
    record=next(r for r in index['images'] if r.get('image')=='edges')
    assert 'Share a border' in record['edge_legend']
    assert not any('fwd' in s or 'context_near' in s for s in record['edge_legend'])


def test_empty_graph_still_has_class_legend_and_zero_edge_counts(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.stages import render_stage_results
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    labels=np.zeros((4,9),np.int32)
    artifact=make_artifact(context.output/'segmentation',labels,np.full(labels.shape,255,np.uint8),year=2013)
    build_tile_graph(artifact,'new',context)
    render_stage_results(context,'graph')
    record=read_json(context.output/'results/graphs/index.json')['images'][0]
    assert record['node_count']==record['edge_count']==0
    assert 'No image data' not in record['class_legend']
    with Image.open(context.output/record['path']) as png:
        assert png.width>=640 and png.height>4
