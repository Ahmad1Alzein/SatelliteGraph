import numpy as np
import rasterio
from scipy import ndimage
from tests.semantic_fixtures import make_year


class TreeModels:
    def probabilities(self,image,scale):
        h,w=image.shape[:2]; prob=np.zeros((9,round(h*scale),round(w*scale)),np.float32)
        prob[5]=1
        return prob

    def parcels(self,image,shape,conf):
        return np.zeros((0,*shape),bool),np.zeros(0)


def test_non_square_masked_tile_is_fully_segmented_and_georeferenced(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    from pipelines.semantic_temporal.runtime.segmentation import segment_tile
    folder=make_year(tmp_path,alpha=True)
    context=prepare_comparison(build_context({'folders':[str(folder)]},tmp_path/'run'))
    artifact=segment_tile(context.years[0].tiles[0],context,models=TreeModels())
    with rasterio.open(artifact.paths['labels']) as src: labels=src.read(1)
    with rasterio.open(artifact.paths['valid']) as src: valid=src.read(1)>0
    assert labels.shape==(24,36)
    assert np.all(labels[valid]>0)
    assert np.all(labels[~valid]==0)
    assert valid.sum()==720
    assert artifact.grid.affine[0]==.75
    assert artifact.grid.affine[2]==500004.5
    for ident in np.unique(labels[valid]): assert ndimage.label(labels==ident)[1]==1


def test_segmentation_requires_physical_scale_for_metric_rules(tmp_path):
    import pytest
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    from pipelines.semantic_temporal.runtime.segmentation import segment_tile
    folder=make_year(tmp_path); (folder/'source.tif').unlink()
    context=prepare_comparison(build_context({'folders':[str(folder)]},tmp_path/'run'))
    with pytest.raises(ValueError,match='pixel size'):
        segment_tile(context.years[0].tiles[0],context,models=TreeModels())


def test_unknown_observations_and_disconnected_islands_survive(tmp_path):
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    from pipelines.semantic_temporal.runtime.segmentation import segment_tile
    from PIL import Image
    folder=make_year(tmp_path,alpha=True)
    tile_image=next(folder.glob('*.png'))
    rgba=np.array(Image.open(tile_image));rgba[:,8:10,3]=0;Image.fromarray(rgba).save(tile_image)
    class UnknownModels(TreeModels):
        def probabilities(self,image,scale):
            result=super().probabilities(image,scale);result[5]=0;result[0]=1;return result
    context=prepare_comparison(build_context({'folders':[str(folder)]},tmp_path/'run'))
    artifact=segment_tile(context.years[0].tiles[0],context,models=UnknownModels())
    with rasterio.open(artifact.paths['labels']) as src: labels=src.read(1)
    with rasterio.open(artifact.paths['classes']) as src: classes=src.read(1)
    with rasterio.open(artifact.paths['valid']) as src: valid=src.read(1)>0
    assert np.all(classes[valid]==0) and np.all(classes[~valid]==255)
    assert np.all(labels[valid]>0) and np.all(labels[~valid]==0)
    assert len(np.unique(labels[valid]))==2
