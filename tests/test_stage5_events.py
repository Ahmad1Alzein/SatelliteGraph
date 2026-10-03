"""Scientific behavior tests for temporal change edges; no pytest required."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

from test_stage5 import graph, build
from src.stage5 import build_temporal_graph, load_stage5_result
from src.stage5.events import ChangePolicy, CLASS_EVENTS, detect_change_events


POLICY = {"boundary_tolerance_m":0.0, "minimum_area_m2":1.0, "minimum_relative_change":0.05}


def events(a,b,**kwargs):
    return build(a,b,change_policy={**POLICY, **kwargs})


def kinds(result, class_id):
    return {e["relation"] for e in result["change_events"] if e["class_id"] == class_id}


class ChangeEventTests(unittest.TestCase):
    def test_appearance_with_mixed_dimension_coverage_does_not_crash(self):
        earlier_polygons = [box(0, 0, 2, 2), box(4, 0, 5, 1)]
        later_polygons = [box(1, 0, 2, 2), box(3, 0, 4, 1)]
        earlier = [
            {"node_id": f"earlier-{index}", "year": 2013,
             "segment_type": {"id": class_id}, "mean_semantic_confidence": 1.0}
            for index, class_id in enumerate((1, 2))
        ]
        later = [
            {"node_id": f"later-{index}", "year": 2023,
             "segment_type": {"id": class_id}, "mean_semantic_confidence": 1.0}
            for index, class_id in enumerate((0, 2))
        ]
        result = detect_change_events(
            earlier, later, earlier_polygons, later_polygons,
            ChangePolicy(minimum_area_m2=0.5, boundary_tolerance_m=0.0),
        )
        self.assertIn("appeared", {event["relation"] for event in result["events"]})

    def test_each_known_class_appearance_and_disappearance(self):
        for class_id in range(8):
            other = (class_id+1)%8
            with self.subTest(class_id=class_id):
                a = graph([(0,0,10,10)],classes=[other])
                b = graph([(0,0,10,10)],classes=[class_id],year=2015)
                forward = events(a,b)
                backward = events(b,a)
                self.assertIn(CLASS_EVENTS[class_id]["appeared"], kinds(forward,class_id))
                self.assertIn(CLASS_EVENTS[class_id]["disappeared"], kinds(backward,class_id))
                self.assertTrue(forward["change_edges"])
                ids = {n["node_id"]:n for n in forward["nodes"]}
                for edge in forward["change_edges"]:
                    self.assertLess(ids[edge["source"]]["year"],ids[edge["target"]]["year"])

    def test_construction_in_large_background_without_identity_match(self):
        a = graph([(0,0,100,100)],classes=[6])
        b = graph([(0,0,10,10),(10,0,100,10),(0,10,100,100)],classes=[7,6,6],year=2015)
        result = events(a,b)
        building = next(n for n in result["nodes"] if n["year"]==2015 and n["segment_type"]["id"]==7)
        self.assertFalse(any(e["target"]==building["node_id"] for e in result["temporal_edges"]))
        self.assertTrue(any(e["target"]==building["node_id"] and e["relation"]=="constructed"
                            for e in result["change_edges"]))

    def test_all_known_classes_expansion_and_contraction(self):
        for cid in range(8):
            other = (cid+1)%8
            a = graph([(0,0,10,10),(10,0,20,10)],classes=[cid,other])
            b = graph([(0,0,15,10),(15,0,20,10)],classes=[cid,other],year=2015)
            with self.subTest(class_id=cid):
                expansion = events(a,b)
                contraction = events(b,a)
                self.assertIn("expanded",kinds(expansion,cid))
                self.assertIn("contracted",kinds(contraction,cid))
                event = next(e for e in expansion["change_events"] if e["class_id"]==cid)
                self.assertAlmostEqual(event["measurements"]["area_delta_m2"],50)

    def test_segmentation_split_is_stable_extent(self):
        a = graph([(0,0,10,10)],classes=[7])
        b = graph([(0,0,5,10),(5,0,10,10)],classes=[7,7],year=2015)
        result = events(a,b)
        self.assertEqual(kinds(result,7),{"persisted"})
        self.assertEqual({e["relation"] for e in result["temporal_edges"]},{"split"})

    def test_equal_net_area_with_gain_and_loss_is_reconfigured(self):
        a = graph([(0,0,10,10),(10,0,20,10)],classes=[5,0])
        b = graph([(0,0,5,10),(5,0,15,10),(15,0,20,10)],classes=[0,5,0],year=2015)
        result = events(a,b)
        self.assertEqual(kinds(result,5),{"reconfigured"})
        water = next(e for e in result["change_events"] if e["class_id"]==5)
        self.assertEqual(water["measurements"]["area_delta_m2"],0)
        self.assertEqual(water["measurements"]["gain_area_m2"],50)
        self.assertEqual(water["measurements"]["loss_area_m2"],50)

    def test_cropped_water_not_disappeared_or_contracted(self):
        result = events(graph([(0,0,10,10)],classes=[5]),
                        graph([(5,0,10,10)],classes=[5],year=2015))
        self.assertFalse(kinds(result,5))
        self.assertIn("insufficient_coverage", {a["status"] for a in result["change_assessments"]})

    def test_unknown_transitions_are_not_construction_or_demolition(self):
        for old,new,expected in [(8,7,"identified"),(7,8,"became_unknown"),(8,8,"unknown_persisted")]:
            result = events(graph([(0,0,10,10)],classes=[old]),
                            graph([(0,0,10,10)],classes=[new],year=2015))
            relations = {e["relation"] for e in result["change_edges"]}
            self.assertIn(expected,relations)
            self.assertFalse(relations & {"constructed","demolished","appeared","disappeared"})

    def test_low_confidence_cannot_establish_change(self):
        a = graph([(0,0,10,10)],classes=[0])
        a["nodes"][0]["mean_semantic_confidence"] = 0.1
        result = events(a,graph([(0,0,10,10)],classes=[7],year=2015))
        self.assertNotIn("constructed",kinds(result,7))

    def test_boundary_tolerance_suppresses_small_misalignment(self):
        a = graph([(0,0,10,10),(10,0,20,10)],classes=[7,0])
        b = graph([(0,0,10.2,10),(10.2,0,20,10)],classes=[7,0],year=2015)
        result = events(a,b,boundary_tolerance_m=0.5)
        self.assertEqual(kinds(result,7),{"persisted"})

    def test_change_event_exports_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build(graph([(0,0,10,10)],classes=[0]),
                           graph([(0,0,10,10)],classes=[7],year=2015),
                           change_policy=POLICY,output_directory=directory)
            loaded = load_stage5_result(directory)
            self.assertEqual(loaded["change_events"],result["change_events"])
            with np.load(Path(directory)/"graph_arrays.npz",allow_pickle=False) as arrays:
                self.assertEqual(arrays["change_edge_index"].shape[1],len(result["change_edges"]))
                self.assertIn("constructed",arrays["change_relations"])
            self.assertTrue((Path(directory)/"changes.geojson").is_file())

    def test_invalid_change_policy_rejected(self):
        for kwargs in [{"boundary_tolerance_m":-1},{"minimum_area_m2":float("nan")},
                       {"minimum_confidence":1.1},{"minimum_relative_change":0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ChangePolicy(**kwargs)

    def test_all_81_class_transitions_have_expected_semantics(self):
        for old in range(9):
            for new in range(9):
                with self.subTest(old=old,new=new):
                    result = events(graph([(0,0,10,10)],classes=[old]),
                                    graph([(0,0,10,10)],classes=[new],year=2015))
                    if old==8 or new==8:
                        expected = "unknown_persisted" if old==new else "identified" if old==8 else "became_unknown"
                        self.assertEqual({e["relation"] for e in result["change_events"]},{expected})
                    elif old==new:
                        self.assertEqual(kinds(result,old),{"persisted"})
                    else:
                        self.assertEqual(kinds(result,old),{CLASS_EVENTS[old]["disappeared"]})
                        self.assertEqual(kinds(result,new),{CLASS_EVENTS[new]["appeared"]})

    def test_pure_merge_does_not_infer_expansion(self):
        result = events(graph([(0,0,5,10),(5,0,10,10)],classes=[5,5]),
                        graph([(0,0,10,10)],classes=[5],year=2015))
        self.assertEqual(kinds(result,5),{"persisted"})

    def test_outside_coverage_does_not_infer_construction(self):
        result = events(graph([(0,0,10,10)],classes=[0]),
                        graph([(20,0,30,10)],classes=[7],year=2015))
        self.assertEqual(result["change_edges"],[])

    def test_unknown_in_replacement_blocks_demolition(self):
        a = graph([(0,0,10,10)],classes=[7])
        b = graph([(0,0,5,10),(5,0,10,10)],classes=[0,8],year=2015)
        result = events(a,b)
        self.assertNotIn("demolished",kinds(result,7))
        self.assertIn("became_unknown",kinds(result,8))

    def test_small_unknown_replacement_never_gets_demolition_edge(self):
        result = events(graph([(0,0,100,10)],classes=[7]),
                        graph([(0,0,96,10),(96,0,100,10)],classes=[0,8],year=2015))
        physical = [e for e in result["change_edges"] if e["relation"]=="demolished"]
        self.assertTrue(physical)
        self.assertTrue(all(e["semantic_to"]!=8 for e in physical))
        event = next(e for e in result["change_events"] if e["relation"]=="demolished")
        self.assertEqual(event["measurements"]["loss_area_m2"],960)
        self.assertEqual(event["measurements"]["unassessed_change_area_m2"],40)
        self.assertAlmostEqual(sum(e["change_support_area_m2"] for e in physical),960)

    def test_small_low_confidence_source_never_gets_construction_edge(self):
        a = graph([(0,0,96,10),(96,0,100,10)],classes=[0,1])
        a["nodes"][1]["mean_semantic_confidence"] = 0.1
        result = events(a,graph([(0,0,100,10)],classes=[7],year=2015))
        physical = [e for e in result["change_edges"] if e["relation"]=="constructed"]
        self.assertTrue(physical)
        self.assertTrue(all(e["semantic_from"]==0 for e in physical))

    def test_schema_rejects_backward_change_edge(self):
        from src.stage5.schema import validate_temporal_graph
        result = events(graph([(0,0,10,10)],classes=[0]),graph([(0,0,10,10)],classes=[7],year=2015))
        edge = result["change_edges"][0]
        edge["source"],edge["target"] = edge["target"],edge["source"]
        with self.assertRaises(ValueError):
            validate_temporal_graph(result)

    def test_legacy_v1_schema_remains_accepted(self):
        from src.stage5.schema import validate_temporal_graph,LEGACY_SCHEMA_VERSION
        result = build(graph([(0,0,10,10)]),graph([(0,0,10,10)],year=2015))
        result["schema_version"] = LEGACY_SCHEMA_VERSION
        for key in ("change_edges","change_events","change_assessments"):
            result.pop(key)
        validate_temporal_graph(result)


if __name__ == "__main__":
    unittest.main()
