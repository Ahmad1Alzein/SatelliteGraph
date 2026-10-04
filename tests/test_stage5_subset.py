"""Tile selection from a completed annual combined graph."""
from shapely.geometry import shape

from tests.test_stage5 import graph
from src.stage4.schema import validate_combined_graph
from src.stage5.subset import clip_combined_graph_to_tile


def test_tile_selection_clips_regions_to_one_png_extent():
    document = graph([(0, 0, 15, 10), (15, 0, 25, 10)])
    document["graph"]["selected_tile_ids"] = ["tile0", "tile1"]
    selected = clip_combined_graph_to_tile(document, "tile0", (0, 0, 10, 10))
    validate_combined_graph(selected)
    assert selected["graph"]["selected_tile_ids"] == ["tile0"]
    assert len(selected["nodes"]) == 1
    assert shape(selected["nodes"][0]["geometry_source_pixel"]).bounds == (0, 0, 10, 10)
    assert selected["nodes"][0]["area_pixels"] == 100
