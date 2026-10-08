import numpy as np
import rasterio
from tests.semantic_fixtures import make_year
from tests.test_semantic_segmentation import TreeModels


def test_new_export_is_consumable_by_real_old_refinement_loader(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    from pipelines.semantic_temporal.runtime.segmentation import segment_tile
    from pipelines.semantic_temporal.runtime.adapters import export_old_refinement_contract,load_old_segmentation
    from src.refinement import load_refinement_result
    folder=make_year(tmp_path,alpha=True)
    context=prepare_comparison(build_context({'folders':[str(folder)]},tmp_path/'run'))
    tile=context.years[0].tiles[0];artifact=segment_tile(tile,context,models=TreeModels())
    directory=export_old_refinement_contract(artifact,context)
    loaded=load_refinement_result(directory)
    assert loaded.semantic_map[loaded.valid_mask].tolist()==[4]*720
    assert not loaded.operations
    returned=load_old_segmentation(directory,tile)
    with rasterio.open(returned.paths['classes']) as src: classes=src.read(1)
    assert np.all(classes[loaded.valid_mask]==3)
    assert returned.grid.affine==artifact.grid.affine
    assert returned.lineage


def test_existing_old_stage2r_can_be_adapted_without_pixel_loss(tmp_path):
    from pipelines.semantic_temporal.runtime.adapters import load_old_segmentation
    from pipelines.semantic_temporal.runtime.contracts import TileInput,GridSpec
    from tests.test_stage3 import _make_fake_project,_fake_config
    from src.stage2 import run_stage2
    from src.stage2.backend import FakeSamBackend
    from src.stage2.semantic import FakeSemanticBackend
    from src.refinement import apply_refinement
    project,key=_make_fake_project(tmp_path);config=_fake_config(project)
    run_stage2(key,project_root=project,config=config,sam_backend=FakeSamBackend(),semantic_backend=FakeSemanticBackend(),show_progress=False)
    refined=apply_refinement(key,project_root=project,config=config,workflow_mode='automatic',show_progress=False)
    tile=TileInput(key,2019,project/'data/tiles/scene/tile_00000.png',None,'scene',(4,3),(10,12),GridSpec('EPSG:32633',(10,0,500040,0,-10,3999970),'metres',(10,10)),(10,12))
    artifact=load_old_segmentation(refined.output_directory,tile)
    with rasterio.open(artifact.paths['labels']) as src: labels=src.read(1)
    assert np.array_equal(labels,refined.instance_map)
    with rasterio.open(artifact.paths['valid']) as src: valid=src.read(1)>0
    assert np.array_equal(valid,refined.valid_mask)
