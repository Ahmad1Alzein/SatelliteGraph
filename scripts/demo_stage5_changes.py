"""Synthetic example only: python -m scripts.demo_stage5_changes.

Creates no claims about the user's imagery. Rectangular regions illustrate the
event vocabulary and produce inspectable graph/GeoJSON/PNG artifacts.
"""
import json
from pathlib import Path

from shapely.geometry import box, mapping

from src.labels import FUSED_CLASS_NAMES
from src.stage4.schema import SCHEMA_VERSION, make_combined_edge_id
from src.stage5 import build_temporal_graph, load_stage5_result


def synthetic_graph(year, regions):
    polygons = [box(left,0,right,20) for left,right,_ in regions]
    nodes = [{"node_id":f"synthetic-{year}-{i}","instance_id":i+1,
              "member_node_ids":[f"synthetic-member-{year}-{i}"],
              "geometry_source_pixel":mapping(polygon),"area_pixels":int(polygon.area),
              "segment_type":{"id":region[2],"name":FUSED_CLASS_NAMES[region[2]]},
              "mean_semantic_confidence":0.99}
             for i,(region,polygon) in enumerate(zip(regions,polygons))]
    edges = []
    for i,a in enumerate(polygons):
        for j in range(i+1,len(polygons)):
            contact = a.boundary.intersection(polygons[j].boundary)
            if contact.length>0:
                source,target = nodes[i]["node_id"],nodes[j]["node_id"]
                edges.append({"edge_id":make_combined_edge_id(source,target),"source":source,"target":target,
                              "same_semantic_class":regions[i][2]==regions[j][2],
                              "shared_boundary_length_pixels":contact.length,
                              "contact_centroid_source_pixel_xy":list(contact.centroid.coords[0])})
    return {"schema_version":SCHEMA_VERSION,"nodes":nodes,"edges":edges,
            "graph":{"id":f"synthetic-combined-{year}","source_image_id":f"SYNTHETIC-{year}",
                     "selected_tile_ids":[f"synthetic-tile-{i}" for i in range(4)],
                     "directed":False,"multigraph":False,"node_count":len(nodes),"edge_count":len(edges),
                     "crs":"EPSG:32636","source_affine_transform":[1,0,500000,0,-1,3700000]}}


def main():
    before = [(0,20,6),(20,40,5),(40,60,7),(60,80,3),(80,100,0),(100,110,5),(110,120,0)]
    after = [(0,10,6),(10,20,7),(20,30,5),(30,40,6),(40,60,0),(60,90,3),(90,110,0),(110,120,5)]
    directory = Path(__file__).resolve().parents[1]/"artifacts/stage5/SYNTHETIC_change_demo/01_temporal_graph"
    build_temporal_graph([
        {"year":2005,"document":synthetic_graph(2005,before)},
        {"year":2015,"document":synthetic_graph(2015,after)},
    ],output_directory=directory)
    document = load_stage5_result(directory)
    print(json.dumps({"data_kind":"SYNTHETIC DEMONSTRATION, not observed imagery",
                     "output_directory":str(directory),
                     "events":[{"class":e["class_name"],"relation":e["relation"],
                                "gain_m2":e["measurements"].get("gain_area_m2"),
                                "loss_m2":e["measurements"].get("loss_area_m2")}
                               for e in document["change_events"]]},indent=2))


if __name__ == "__main__":
    main()
