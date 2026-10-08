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
    with pytest.raises(ValueError, match='metadata'): inspect({'folders':[str(folder)]}, tmp_path)
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
