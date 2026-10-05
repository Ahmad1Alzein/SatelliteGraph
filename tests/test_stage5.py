"""Run without pytest: python -m unittest discover -s tests -p test_stage5.py."""
import copy
from collections import Counter
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box, mapping, shape

from src.stage4.schema import SCHEMA_VERSION as STAGE4_SCHEMA
from src.stage5 import build_temporal_graph, load_stage5_result
from src.stage5.coordinates import project_snapshot
from src.stage5.matching import _context_similarity
from src.stage5.registration import fit_registration


def test_change_event_identity_distinguishes_shared_support_components():
    from src.stage5.events import _event_id
    inputs = (1, "expanded", 2013, 2023, ["a"], ["b"])
    first = _event_id(*inputs, ["class-a"], ["class-b"], box(0, 0, 3, 3))
    second = _event_id(*inputs, ["class-a"], ["class-b"], box(1, 0, 4, 3))
    third = _event_id(*inputs, ["class-c"], ["class-b"], box(0, 0, 3, 3))
    assert len({first, second, third}) == 3


def test_local_coordinate_consensus_recovers_unmatched_building_with_shape_change():
    from shapely.ops import unary_union
    from src.stage5.matching import match_snapshots, MatchPolicy

    old = [box(0,0,20,20), box(30,0,40,10), box(30,15,40,25),
           box(0,30,10,40), box(30,30,40,40)]
    new = [box(5,2,20,15), box(31,0,41,10), box(31,15,41,25),
           box(1,30,11,40), box(31,30,41,40)]
    footprint = box(-5,-5,50,45)
    old.append(footprint.difference(unary_union(old)))
    new.append(footprint.difference(unary_union(new)))
    first = {"nodes":[{"node_id":f"old-{i}","segment_type":{"id":7 if i<5 else 0}}
                      for i in range(6)],"edges":[]}
    second = {"nodes":[{"node_id":f"new-{i}","segment_type":{"id":7 if i<5 else 0}}
                       for i in range(6)],"edges":[]}
    result = match_snapshots(first, second, old, new, MatchPolicy())
    rescued = next(link for link in result["links"] if link["i"] == 0)
    assert rescued["j"] == 0
    assert rescued["relation"] == "correspondence"
    assert rescued["evidence"]["rescue_method"] == "local_coordinate_consensus"
    assert rescued["evidence"]["score"] < 0.60
    assert abs(rescued["evidence"]["predicted_shift_xy_m"][0] - 1.0) < 0.1


class CoordinateRescueTests(unittest.TestCase):
    def test_unmatched_building_with_shape_change(self):
        test_local_coordinate_consensus_recovers_unmatched_building_with_shape_change()


def test_nearby_temporal_links_receive_distinct_colors():
    from src.stage5.storage import _assign_local_colors
    positions = {"a":(0, 100), "b":(2, 100), "c":(100, 100),
                 "d":(0, 0), "e":(3, 0), "f":(100, 0)}
    edges = [{"source":"a", "target":"d", "edge_id":"one"},
             {"source":"b", "target":"e", "edge_id":"two"},
             {"source":"c", "target":"f", "edge_id":"three"}]
    colors = _assign_local_colors(edges, positions, radius=12)
    assert colors[0] != colors[1]
    assert len(colors) == len(edges)
    crowded = [{"source":f"s{i}", "target":f"t{i}", "edge_id":f"link{i}"}
               for i in range(42)]
    crowded_positions = {**{f"s{i}":(i*0.01,100) for i in range(42)},
                         **{f"t{i}":(i*0.01,0) for i in range(42)}}
    assert len(set(_assign_local_colors(crowded,crowded_positions,radius=12))) == 42


def test_edge_type_overview_has_one_fixed_color_per_relation():
    from src.stage5.events import EVENT_RELATIONS
    from src.stage5.storage import EDGE_TYPE_COLORS
    assert EVENT_RELATIONS <= EDGE_TYPE_COLORS.keys()
    assert {"adjacent", "correspondence", "split", "merge"} <= EDGE_TYPE_COLORS.keys()
    assert len(set(EDGE_TYPE_COLORS.values())) == len(EDGE_TYPE_COLORS)


def test_public_outcome_palette_distinguishes_class_and_action():
    from src.stage5.storage import _outcome_color
    relations = ["building_constructed","building_demolished","building_unchanged",
                 "tree_cover_gained","tree_cover_grew","tree_cover_lost",
                 "tree_cover_shrank","tree_cover_unchanged"]
    assert len({_outcome_color(relation) for relation in relations}) == len(relations)


def test_guided_png_registration_recovers_cross_sensor_shift(tmp_path):
    """A coarse geographic prior plus distributed landmarks must recover drift."""
    import cv2
    from rasterio.transform import from_origin
    from src.stage5.registration import staged_tile_control_points

    rng = np.random.default_rng(17)
    base = rng.integers(25, 210, (384, 384), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    for _ in range(90):
        x, y = rng.integers(15, 369, 2)
        cv2.circle(base, (int(x), int(y)), int(rng.integers(2, 11)),
                   int(rng.integers(20, 240)), -1)
    moving = cv2.warpAffine(base, np.float32([[1, 0, -17], [0, 1, -21]]),
                            (384, 384), borderMode=cv2.BORDER_REFLECT)
    moving = cv2.convertScaleAbs(moving, alpha=0.82, beta=32)
    ref_path, moving_path = tmp_path/"reference.png", tmp_path/"moving.png"
    cv2.imwrite(str(ref_path), base)
    cv2.imwrite(str(moving_path), moving)
    affine = from_origin(500000, 3700000, 1.5, 1.5)
    points, reference, diagnostics = staged_tile_control_points(
        moving_path, affine, "EPSG:32636", ref_path, affine, "EPSG:32636",
        "EPSG:32636")
    fit = fit_registration(points, reference, max_residual_m=4)
    matrix = np.asarray(fit["affine"]).reshape(2, 3)
    assert diagnostics["method"] == "phase_guided_sift_ransac"
    assert diagnostics["inlier_count"] >= 20
    center = np.array([500288, 3699712.0])
    corrected = matrix[:, :2] @ center + matrix[:, 2]
    assert np.allclose(corrected-center, [25.5, -31.5], atol=3.5)


def graph(rectangles, *, origin=0, classes=None, year=2005, crs="EPSG:32636"):
    nodes = []
    for index, bounds in enumerate(rectangles):
        polygon = box(*bounds)
        nodes.append({"node_id": f"node{index}", "instance_id": index + 1,
                      "member_node_ids": [f"tile-node{index}"],
                      "geometry_source_pixel": mapping(polygon),
                      "area_pixels": int(polygon.area),
                      "segment_type": {"id": (classes or [1]*len(rectangles))[index]},
                      "mean_semantic_confidence": 0.95})
    return {"schema_version": STAGE4_SCHEMA, "graph": {
        "id": f"graph{year}", "source_image_id": f"scene{year}",
        "selected_tile_ids": [f"tile{i}" for i in range(4)],
        "directed": False, "multigraph": False, "node_count": len(nodes),
        "edge_count": 0, "crs": crs,
        "source_affine_transform": [1, 0, 500000 + origin, 0, -1, 3700000]},
        "nodes": nodes, "edges": []}


def build(a, b, **kwargs):
    return build_temporal_graph([
        {"year": 2005, "document": a}, {"year": 2015, "document": b}], **kwargs)


class TemporalGraphTests(unittest.TestCase):
    def test_neighborhood_similarity_uses_cosine_and_class_distribution(self):
        self.assertAlmostEqual(_context_similarity(Counter({1: 2, 3: 1}),
                                                    Counter({1: 4, 3: 2})), 1.0)
        self.assertAlmostEqual(_context_similarity(Counter({1: 3}), Counter({3: 3})), 0.0)
        self.assertAlmostEqual(_context_similarity(Counter(), Counter({1: 2})), 0.5)

    def test_shifted_pixel_origins_match_same_world_object(self):
        a = graph([(10, 0, 20, 10)])
        b = graph([(0, 0, 10, 10)], origin=10, year=2015)
        result = build(a, b)
        self.assertEqual(len(result["temporal_edges"]), 1)
        edge = result["temporal_edges"][0]
        self.assertEqual(edge["relation"], "correspondence")
        self.assertAlmostEqual(edge["evidence"]["iou"], 1)
        self.assertEqual(result["nodes"][0]["track_id"], result["nodes"][1]["track_id"])
        self.assertAlmostEqual(shape(result["nodes"][0]["geometry_world"]).area, 100)

    def test_preview_stacks_raw_images_vertically(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root/"first.png", root/"second.png"
            Image.new("RGB", (40, 40), (245, 20, 20)).save(first)
            Image.new("RGB", (40, 40), (20, 200, 20)).save(second)
            sources = [
                {"year": year, "document": graph([(0, 0, 40, 40)], year=year),
                 "preview_tiles": [{"path": str(path), "x0": 0, "y0": 0,
                                    "width": 40, "height": 40}]}
                for year, path in ((2005, first), (2015, second))
            ]
            build_temporal_graph(sources, output_directory=root/"result")
            pixels = np.asarray(Image.open(root/"result/temporal_graph.png").convert("RGB"))
            self.assertGreater(pixels.shape[0], pixels.shape[1])
            top, bottom = pixels[:pixels.shape[0]//2], pixels[pixels.shape[0]//2:]
            self.assertGreater(np.count_nonzero((top[:,:,0]>180)&(top[:,:,1]<80)), 1000)
            self.assertGreater(np.count_nonzero((bottom[:,:,1]>130)&(bottom[:,:,0]<80)), 1000)

    def test_missing_crs_and_geographic_target_rejected(self):
        a = graph([(0, 0, 10, 10)], crs=None)
        with self.assertRaisesRegex(ValueError, "CRS"):
            build(a, graph([(0, 0, 10, 10)], year=2015))
        with self.assertRaisesRegex(ValueError, "projected"):
            build(graph([(0, 0, 10, 10)]), graph([(0, 0, 10, 10)], year=2015), target_crs="EPSG:4326")

    def test_semantic_change_does_not_prevent_correspondence(self):
        result = build(graph([(0, 0, 10, 10)], classes=[1]),
                       graph([(0, 0, 10, 10)], classes=[7], year=2015))
        self.assertTrue(result["temporal_edges"][0]["change"]["semantic_changed"])

    def test_split_and_merge_are_lineage_not_same_track(self):
        a = graph([(0, 0, 10, 10)])
        b = graph([(0, 0, 5, 10), (5, 0, 10, 10)], year=2015)
        result = build(a, b)
        self.assertEqual([e["relation"] for e in result["temporal_edges"]], ["split", "split"])
        self.assertEqual(len({n["track_id"] for n in result["nodes"]}), 3)
        self.assertEqual({e["relation"] for e in build(b, a)["temporal_edges"]}, {"merge"})

    def test_distant_objects_are_not_merged_by_one_broad_later_region(self):
        a = graph([(0,0,10,10),(100,0,110,10)], classes=[5,5])
        b = graph([(0,0,110,10)], classes=[5], year=2015)
        result = build(a,b)
        self.assertEqual(result["temporal_edges"],[])
        self.assertEqual(result["graph"]["matching_refinement"][0]["rejected_by_reason"],
                         {"distant_lineage_siblings":2})
        self.assertEqual({c["reason"] for c in result["comparisons"][0]["candidates"]},
                         {"distant_lineage_siblings"})

    def test_nonoverlapping_coverage_is_not_disappearance(self):
        result = build(graph([(0, 0, 10, 10)]), graph([(100, 0, 110, 10)], year=2015))
        self.assertEqual(result["temporal_edges"], [])
        self.assertEqual({s["status"] for s in result["comparisons"][0]["node_states"]}, {"outside_common_coverage"})

    def test_partial_coverage_has_no_full_area_change(self):
        result = build(graph([(0, 0, 10, 10)]), graph([(5, 0, 10, 10)], year=2015))
        edge = result["temporal_edges"][0]
        self.assertIsNone(edge["change"]["area_delta_m2"])
        self.assertTrue(edge["change"]["coverage_limited"])

    def test_duplicate_years_rejected(self):
        a = graph([(0, 0, 10, 10)])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_temporal_graph([{"year": 2005, "document": a}]*2)

    def test_equal_many_to_many_candidates_abstain(self):
        a = graph([(0,0,5,10), (5,0,10,10)])
        b = graph([(0,0,10,5), (0,5,10,10)], year=2015)
        result = build(a, b, policy={"minimum_score": 0.4})
        self.assertEqual(result["temporal_edges"], [])
        self.assertEqual({c["reason"] for c in result["comparisons"][0]["candidates"]}, {"ambiguous"})

    def test_three_years_keep_a_forward_track(self):
        result = build_temporal_graph([{"year": year, "document": graph([(0,0,10,10)], year=year)}
                                       for year in [2020, 2005, 2015]])
        self.assertEqual(result["graph"]["years"], [2005,2015,2020])
        self.assertEqual(len(result["temporal_edges"]), 2)
        self.assertEqual(len({n["track_id"] for n in result["nodes"]}), 1)

    def test_near_threshold_alternative_still_makes_match_ambiguous(self):
        a = graph([(0,0,6,10), (6,0,10,10)])
        b = graph([(0,0,10,5.2), (0,5.2,10,10)], year=2015)
        baseline = build(a,b,policy={"minimum_score":0.4})
        candidates = baseline["comparisons"][0]["candidates"]
        best = max(candidates, key=lambda c:c["score"])
        competitor = max(c["score"] for c in candidates if c is not best
                         and (c["source"] == best["source"] or c["target"] == best["target"]))
        self.assertLess(best["score"] - competitor, 0.05)
        result = build(a,b,policy={"minimum_score":(best["score"]+competitor)/2})
        self.assertEqual(result["temporal_edges"], [])

    def test_unknown_semantics_are_not_confirmed_change(self):
        result = build(graph([(0,0,10,10)], classes=[8]), graph([(0,0,10,10)], classes=[7],year=2015))
        self.assertIsNone(result["temporal_edges"][0]["change"]["semantic_changed"])

    def test_boundary_touch_is_not_temporal_overlap(self):
        result = build(graph([(0,0,10,10)]), graph([(10,0,20,10)], year=2015))
        self.assertEqual(result["temporal_edges"], [])

    def test_projected_feet_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "metre"):
            build(graph([(0,0,10,10)]), graph([(0,0,10,10)],year=2015), target_crs="EPSG:2263")

    def test_registration_corrects_whole_later_graph(self):
        points = [[500000,3700000], [500010,3700000], [500000,3699990],
                  [500010,3699990], [500005,3699995], [500003,3699998]]
        moving = (np.asarray(points)+[20,10]).tolist()
        b = graph([(0,0,10,10)], origin=20, year=2015)
        b["graph"]["source_affine_transform"][5] += 10
        result = build_temporal_graph([
            {"year": 2005, "document": graph([(0,0,10,10)])},
            {"year": 2015, "document": b,
             "registration": {"moving_xy": moving, "reference_xy": points}}])
        self.assertAlmostEqual(result["temporal_edges"][0]["evidence"]["iou"], 1, places=5)

    def test_spatial_edges_preserved_and_arrays_bidirectional(self):
        from src.stage4.schema import make_combined_edge_id
        a = graph([(0,0,5,10), (5,0,10,10)])
        a["edges"] = [{"edge_id": make_combined_edge_id("node0", "node1"),
                       "source": "node0", "target": "node1", "same_semantic_class": True,
                       "shared_boundary_length_pixels": 10,
                       "contact_centroid_source_pixel_xy": [5,5]}]
        a["graph"]["edge_count"] = 1
        b = copy.deepcopy(a)
        b["graph"]["id"] = "graph2015"
        with tempfile.TemporaryDirectory() as directory:
            result = build(a,b,output_directory=directory)
            self.assertEqual(len(result["spatial_edges"]),2)
            np.testing.assert_allclose(result["spatial_edges"][0]["contact_world_xy"], [500005,3699995])
            with np.load(Path(directory)/"graph_arrays.npz") as arrays:
                self.assertEqual(arrays["spatial_edge_index"].shape,(2,4))

    def test_spatial_edge_projects_boundary_anchor_when_available(self):
        from src.stage4.schema import make_combined_edge_id

        earlier = graph([(0,0,5,10), (5,0,10,10)])
        earlier["edges"] = [{"edge_id": make_combined_edge_id("node0", "node1"),
                             "source": "node0", "target": "node1", "same_semantic_class": True,
                             "shared_boundary_length_pixels": 10,
                             "contact_centroid_source_pixel_xy": [5,5],
                             "contact_anchor_source_pixel_xy": [5,2]}]
        earlier["graph"]["edge_count"] = 1
        later = copy.deepcopy(earlier)
        later["graph"]["id"] = "graph2015"
        result = build(earlier, later)
        self.assertEqual(len(result["spatial_edges"]), 2)
        for edge in result["spatial_edges"]:
            np.testing.assert_allclose(edge["contact_world_xy"], [500005,3699998])

    def test_registration_control_points_must_support_selected_footprint(self):
        far_away = [[0,0],[100,0],[0,100],[100,100],[50,30],[20,80]]
        with self.assertRaisesRegex(ValueError,"cover"):
            build_temporal_graph([
                {"year":2005,"document":graph([(0,0,10,10)])},
                {"year":2015,"document":graph([(0,0,10,10)],year=2015),
                 "registration":{"moving_xy":far_away,"reference_xy":far_away}}])

    def test_cli_parser_accepts_series(self):
        from src.cli import parser
        args = parser().parse_args(["stage5", "build", "--series", "series.json"])
        self.assertEqual(args.series, "series.json")

    def test_policy_rejects_nan(self):
        with self.assertRaises(ValueError):
            build(graph([(0,0,10,10)]), graph([(0,0,10,10)],year=2015), policy={"minimum_score": float("nan")})

    def test_transform_order_rotation_and_pixel_edges(self):
        a = graph([(0, 0, 10, 10)])
        a["graph"]["source_affine_transform"] = [0, -2, 500000, 2, 0, 3700000]
        projected = project_snapshot(a, "EPSG:32636")
        self.assertAlmostEqual(projected[0].area, 400)
        self.assertEqual(projected[0].bounds, (499980, 3700000, 500000, 3700020))

    def test_different_crs_same_location(self):
        a = graph([(0, 0, 10, 10)])
        b = copy.deepcopy(a)
        from rasterio.warp import transform_geom
        b["graph"]["crs"] = "EPSG:4326"
        b["graph"]["source_affine_transform"] = [1, 0, 0, 0, 1, 0]
        b["nodes"][0]["geometry_source_pixel"] = transform_geom(
            "EPSG:32636", "EPSG:4326", mapping(project_snapshot(a, "EPSG:32636")[0]))
        self.assertLess(project_snapshot(b, "EPSG:32636")[0].symmetric_difference(
            project_snapshot(a, "EPSG:32636")[0]).area, 0.01)

    def test_robust_registration_rejects_outlier_and_degeneracy(self):
        moving = np.array([[0,0], [100,0], [0,100], [100,100], [50,30], [20,80], [70,60], [40,90]], float)
        reference = moving + [12, -7]
        reference[-1] += 500
        fit = fit_registration(moving, reference, max_residual_m=1)
        self.assertEqual(fit["inlier_count"], 7)
        np.testing.assert_allclose(fit["affine"], [1, 0, 12, 0, 1, -7], atol=1e-6)
        with self.assertRaises(ValueError):
            fit_registration(np.zeros((8,2)), np.zeros((8,2)))

    def test_persistence_checksums_and_array_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build(graph([(0,0,10,10)]), graph([(0,0,10,10)],year=2015), output_directory=directory)
            loaded = load_stage5_result(directory)
            self.assertEqual(loaded["graph"], result["graph"])
            self.assertTrue((Path(directory)/"outcome_graph.png").is_file())
            self.assertEqual([e["relation"] for e in loaded["semantic_temporal_edges"]],
                             ["rangeland_unchanged"])
            with np.load(Path(directory)/"graph_arrays.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["temporal_edge_index"].shape, (2,1))
                self.assertEqual(arrays["semantic_edge_index"].shape, (2,1))
                self.assertLess(arrays["years"][arrays["temporal_edge_index"][0,0]], arrays["years"][arrays["temporal_edge_index"][1,0]])
            (Path(directory)/"graph.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                load_stage5_result(directory)

    def test_registration_requires_at_least_six_inliers(self):
        triangle = [[0,0],[100,0],[0,100]]
        with self.assertRaisesRegex(ValueError, "min_inliers"):
            fit_registration(triangle,triangle,min_inliers=3)

    def test_registration_rejects_invalid_policy_values(self):
        points = [[0,0],[100,0],[0,100],[100,100],[50,30],[20,80]]
        invalid = [
            {"max_residual_m":float("nan")}, {"max_residual_m":float("inf")},
            {"max_residual_m":True}, {"min_inlier_ratio":float("nan")},
            {"min_inlier_ratio":0}, {"min_inlier_ratio":1.1},
            {"min_inliers":6.5}, {"min_inliers":True},
        ]
        for policy in invalid:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError,"Registration policy"):
                    fit_registration(points,points,**policy)

    def test_repeated_control_points_do_not_count_as_independent_support(self):
        points = [[0,0],[100,0],[0,100]] * 3
        with self.assertRaisesRegex(ValueError,"distinct"):
            fit_registration(points,points)

    def test_series_file_builds_from_checksum_bound_stage4_directories(self):
        from src.stage4.schema import MATCH_SCHEMA_VERSION
        from src.stage2.artifacts import atomic_json, checksum
        from src.stage5 import run_stage5
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = []
            for year in [2005,2015]:
                directory = root/str(year)
                directory.mkdir()
                atomic_json(directory/"graph.json", graph([(0,0,10,10)],year=year))
                atomic_json(directory/"overlap_matches.json", {"schema_version":MATCH_SCHEMA_VERSION,"matches":[]})
                atomic_json(directory/"report.json", {})
                artifacts = {name:str(directory/name) for name in ["graph.json","overlap_matches.json","report.json"]}
                atomic_json(directory/"manifest.json", {"schema_version":STAGE4_SCHEMA,"status":"complete",
                            "artifacts":artifacts,"artifact_checksums":{key:checksum(Path(path)) for key,path in artifacts.items()}})
                snapshots.append({"year":year,"graph_directory":str(year)})
            atomic_json(root/"series.json", {"name":"integration", "snapshots":snapshots})
            summary = run_stage5("series.json",project_root=root)
            self.assertEqual(summary["temporal_edge_count"],1)
            loaded = load_stage5_result(summary["output_directory"])
            self.assertEqual(loaded["graph"]["years"],[2005,2015])
            (root/"2005"/"graph.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError,"upstream input"):
                load_stage5_result(summary["output_directory"])

    def test_sift_raster_registration_world_coordinates(self):
        import cv2
        import rasterio
        from rasterio.transform import from_origin
        from src.stage5.registration import image_control_points
        image = np.zeros((256,256),dtype=np.uint8)
        rng = np.random.default_rng(42)
        for x,y,radius,color in rng.integers([10,10,2,70],[245,245,8,255],size=(100,4)):
            cv2.circle(image,(int(x),int(y)),int(radius),int(color),-1)
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary)/name for name in ["reference.tif","moving.tif"]]
            for path, shift in zip(paths,[0,12]):
                with rasterio.open(path,"w",driver="GTiff",width=256,height=256,count=1,dtype="uint8",
                                   crs="EPSG:32636",transform=from_origin(500000+shift,3700000,1,1)) as raster:
                    raster.write(image,1)
            moving,reference = image_control_points(paths[1],paths[0],"EPSG:32636")
            result = fit_registration(moving,reference,max_residual_m=0.5)
            np.testing.assert_allclose(result["affine"],[1,0,-12,0,1,0],atol=1e-5)


if __name__ == "__main__":
    unittest.main()
