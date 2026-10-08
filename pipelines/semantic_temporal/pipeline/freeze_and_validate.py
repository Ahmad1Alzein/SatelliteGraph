"""Freeze the final segmentation of one tile/year and run the node-readiness / topology validation.

Steps
  1. Reproducibility: rerun segment_final.py into a scratch folder; the segment raster must be
     bit-identical to the working result.
  2. Build the final node table (segment id, class, area, centroid, geometry, confidence, source).
  3. Freeze: copy the release files into final_segmentation/FROZEN_<year>_<tile>_v<k>/ together with
     a code snapshot and a SHA-256 manifest of data, code, input image and model checkpoints.
  4. Validate the FROZEN copy independently of the pipeline (checks C1-C15), write
     validation_report.json/.md into the frozen folder, then mark every file read-only.

Checks
  C1  full pixel coverage (no unlabelled pixel)        C9  georeferencing (CRS, 0.75 m grid, extent)
  C2  ids contiguous; raster = nodes = polygons        C10 class minimum segment sizes
  C3  no overlaps / gaps between polygons              C11 adjacency graph consistent and connected
  C4  every geometry valid, single Polygon             C12 merge step is at a fixed point
  C5  raster and vector agree pixel for pixel          C13 reproducible (bit-identical rerun)
  C6  every segment one 4-connected region             C14 node position inside its own segment
  C7  exactly one semantic class per segment           C15 node table complete and consistent
  C8  areas consistent (pixels, polygon, table)
"""
from __future__ import annotations

from osgeo import gdal, ogr, osr  # noqa: F401  (GDAL DLLs before torch)
import argparse
import hashlib
import json
import shutil
import stat
import time
from collections import Counter
from pathlib import Path

import paths  # package: input/checkpoint locations

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree, from_wkb, unary_union
from shapely.geometry import Point, box

import refine_merge
import segment_final as sf
import segment_v2 as v2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE_FILES = ["segment_final.py", "refine_merge.py", "segment_v2.py", "semantic_ms.py", "harmonize.py",
              "segment_final_viz.py", "freeze_and_validate.py"]
RELEASE_FILES = ["segment_ids.tif", "segment_classes.tif", "segments.gpkg", "nodes.csv", "edges.csv",
                 "merge_log.csv", "run_log.json", "viz_full.png", "viz_zooms.png", "viz_merge_decisions.png",
                 "segment_ids_before_merge.tif", "README.md"]
MODELS = {"segformer": paths.SEGFORMER_DIR / "model.safetensors",   # package: checkpoint locations from paths.py
          "delineate_anything_v2": paths.DELANY_CKPT}
PX_AREA, PX_M = 0.5625, 0.75


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- 1 reproducibility
def rerun(tile, year, image_path, scratch: Path):
    with rasterio.open(image_path) as s:
        image = s.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
    log = {}
    final, *_ = sf.segment(image, PX_AREA, log)
    return final


# ----------------------------------------------------------------------------- 2 final node table
def source_primary(src: str) -> str:
    for key, name in (("building_v3", "building_instance"), ("road_segment", "road_segment"),
                      ("delany_parcel", "delany_parcel"), ("water", "water_body"), ("class_region", "class_region")):
        if src.startswith(key):
            return name
    return src.split("+")[0]


def build_node_table(work: Path, tile: str, year: str):
    nodes = pd.read_csv(work / "nodes.csv")
    ds = ogr.Open(str(work / "segments.gpkg"))
    lyr = ds.GetLayer(0)
    geoms = {f.GetField("node_id"): from_wkb(bytes(f.GetGeometryRef().ExportToWkb())) for f in lyr}
    ds = None
    rows = []
    for r in nodes.to_dict("records"):
        g = geoms[r["node_id"]]
        c = g.centroid
        ip = g.representative_point()
        rule = "rgb_rule" in r["source"]
        rows.append({
            "segment_id": int(r["node_id"]), "node_key": r["node_key"], "tile_id": tile, "year": year,
            "class_id": int(r["class_id"]), "class": r["class"],
            "area_m2": round(float(g.area), 2), "perimeter_m": r["perimeter_m"],
            "centroid_x": round(c.x, 3), "centroid_y": round(c.y, 3),
            "centroid_inside": bool(g.contains(c)),
            "node_x": round(ip.x, 3), "node_y": round(ip.y, 3),     # point guaranteed inside the segment
            "confidence": float(r["class_confidence"]),
            "confidence_type": "rgb_rule_support" if rule else "model_class_probability",
            "source": source_primary(r["source"]),
            "source_lineage": r["source"],
            "edge_split": "edge_split" in r["source"], "merged": "merged" in r["source"], "rgb_rule": rule,
            "geometry_wkt": g.wkt,
        })
    return pd.DataFrame(rows), geoms


def write_node_gpkg(path: Path, table: pd.DataFrame, geoms, crs_wkt: str):
    if path.exists():
        path.unlink()
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    srs = osr.SpatialReference(); srs.ImportFromWkt(crs_wkt)
    lyr = ds.CreateLayer("nodes", srs, ogr.wkbPolygon)
    types = {"segment_id": ogr.OFTInteger, "class_id": ogr.OFTInteger, "area_m2": ogr.OFTReal, "perimeter_m": ogr.OFTReal,
             "centroid_x": ogr.OFTReal, "centroid_y": ogr.OFTReal, "node_x": ogr.OFTReal, "node_y": ogr.OFTReal,
             "confidence": ogr.OFTReal, "centroid_inside": ogr.OFTInteger, "edge_split": ogr.OFTInteger,
             "merged": ogr.OFTInteger, "rgb_rule": ogr.OFTInteger}
    cols = [c for c in table.columns if c != "geometry_wkt"]
    for c in cols:
        lyr.CreateField(ogr.FieldDefn(c[:30], types.get(c, ogr.OFTString)))
    lyr.StartTransaction()
    for r in table.to_dict("records"):
        f = ogr.Feature(lyr.GetLayerDefn())
        for c in cols:
            v = r[c]
            f.SetField(c[:30], int(v) if isinstance(v, (bool, np.bool_)) else v)
        f.SetGeometry(ogr.CreateGeometryFromWkb(geoms[r["segment_id"]].wkb))
        lyr.CreateFeature(f)
    lyr.CommitTransaction()
    ds = None


# ----------------------------------------------------------------------------- 4 validation
def validate(frozen: Path, image_path: Path, repro: dict):
    R = []

    def check(cid, name, ok, detail, level="FAIL"):
        R.append({"id": cid, "check": name, "result": "PASS" if ok else level, "detail": detail})

    with rasterio.open(frozen / "segment_ids.tif") as s:
        seg = s.read(1); seg_tr, seg_crs = s.transform, s.crs
    with rasterio.open(frozen / "segment_classes.tif") as s:
        cls_map = s.read(1)
    with rasterio.open(image_path) as s:
        img_tr, img_crs, img_shape = s.transform, s.crs, (s.height, s.width)
    nodes = pd.read_csv(frozen / "nodes_final.csv")
    edges = pd.read_csv(frozen / "edges.csv")
    ds = ogr.Open(str(frozen / "nodes_final.gpkg"))
    lyr = ds.GetLayer(0)
    polys = {f.GetField("segment_id"): from_wkb(bytes(f.GetGeometryRef().ExportToWkb())) for f in lyr}
    ds = None
    N = int(seg.max())
    ids = np.unique(seg)

    # C1 coverage
    zero = int((seg == 0).sum())
    check("C1", "Full pixel coverage", zero == 0, f"{seg.size:,} pixels, {zero} unlabelled")
    # C2 id consistency
    contiguous = np.array_equal(ids, np.arange(1, N + 1))
    same_sets = set(ids.tolist()) == set(nodes.segment_id) == set(polys)
    check("C2", "Segment ids consistent", contiguous and same_sets and len(nodes) == N,
          f"{N} segments; ids 1..{N} contiguous={contiguous}; raster/table/polygon id sets equal={same_sets}")
    # C3 overlaps and gaps (vector)
    geoms = [polys[i] for i in sorted(polys)]
    tree = STRtree(geoms)
    overlap = 0.0; pairs = 0
    for i, g in enumerate(geoms):
        for j in tree.query(g, predicate="intersects"):
            if j > i:
                a = g.intersection(geoms[j]).area
                if a > 0:
                    overlap += a; pairs += 1
    footprint = box(seg_tr.c, seg_tr.f + seg_tr.e * seg.shape[0], seg_tr.c + seg_tr.a * seg.shape[1], seg_tr.f)
    union = unary_union(geoms)
    gap = footprint.difference(union).area
    outside = union.difference(footprint).area
    check("C3", "No overlaps or gaps", overlap < 1e-6 and gap < 1e-6 and outside < 1e-6,
          f"overlap area {overlap:.6f} m2 ({pairs} pairs with area); gap area {gap:.6f} m2; area outside tile {outside:.6f} m2; "
          f"sum of polygon areas {sum(g.area for g in geoms):,.2f} m2 vs tile {footprint.area:,.2f} m2")
    # C4 validity, single polygon
    invalid = [i for i, g in polys.items() if not g.is_valid]
    multi = [i for i, g in polys.items() if g.geom_type != "Polygon"]
    empty = [i for i, g in polys.items() if g.is_empty]
    holes = sum(len(g.interiors) > 0 for g in polys.values())
    check("C4", "Valid single-part geometry", not invalid and not multi and not empty,
          f"invalid {len(invalid)}, multi-part {len(multi)}, empty {len(empty)}; {holes} segments have holes (they enclose other segments - allowed)")
    # C5 raster/vector agreement
    rr = features.rasterize(((polys[i], i) for i in polys), out_shape=seg.shape, transform=seg_tr, fill=0, dtype=np.int32)
    mismatch = int((rr != seg).sum())
    check("C5", "Raster and polygons agree", mismatch == 0, f"{mismatch} pixels differ after re-rasterizing the polygons")
    # C6 connectivity
    disconnected = []
    for i, sl in enumerate(ndimage.find_objects(seg), 1):
        if sl is not None and ndimage.label(seg[sl] == i, structure=v2.FOUR)[1] != 1:
            disconnected.append(i)
    check("C6", "Every segment is one connected region", not disconnected,
          f"{len(disconnected)} segments with more than one 4-connected component")
    # C7 one class per segment
    lo = ndimage.minimum(cls_map, seg, ids); hi = ndimage.maximum(cls_map, seg, ids)
    mixed = int((np.array(lo) != np.array(hi)).sum())
    tab = nodes.set_index("segment_id").class_id
    mism = int(sum(int(l) != int(tab[i]) for i, l in zip(ids, lo)))
    valid_cls = set(nodes.class_id) <= set(sf.CLASSES)
    check("C7", "Exactly one semantic class per segment", mixed == 0 and mism == 0 and valid_cls,
          f"{mixed} segments with mixed class pixels; {mism} raster/table class mismatches; classes used {sorted(set(nodes['class']))}")
    # C8 areas
    pix = np.bincount(seg.ravel())[1:] * PX_AREA
    d_pix = float(np.abs(pix - nodes.sort_values("segment_id").area_m2.values).max())
    d_poly = float(max(abs(polys[r.segment_id].area - r.area_m2) for r in nodes.itertuples()))
    check("C8", "Areas consistent", d_pix < 0.01 and d_poly < 0.01,
          f"max |pixel area - table| {d_pix:.4f} m2; max |polygon area - table| {d_poly:.4f} m2")
    # C9 georeferencing
    exp_tr = img_tr * rasterio.Affine.scale(0.5)
    geo_ok = seg_crs == img_crs and seg_tr == exp_tr and seg.shape == (img_shape[0] * 2, img_shape[1] * 2)
    check("C9", "Georeferencing", geo_ok,
          f"CRS {seg_crs.to_string()}; pixel {seg_tr.a} m; origin ({seg_tr.c}, {seg_tr.f}) = source tile origin; {seg.shape[1]}x{seg.shape[0]} px = {seg.shape[1]*PX_M:.0f} x {seg.shape[0]*PX_M:.0f} m")
    # C10 minimum sizes
    below = [(r.segment_id, r["class"], r.area_m2) for _, r in nodes.iterrows()
             if r.area_m2 + 1e-6 < sf.P["min_segment_m2"][int(r.class_id)]]
    check("C10", "Class minimum segment size", not below, f"{len(below)} segments below their class minimum {sf.P['min_segment_m2']}")
    # C11 adjacency
    recomputed = set(sf.shared_boundaries(seg.astype(np.int64)).keys())
    listed = set(zip(edges.source, edges.target))
    listed = {(min(a, b), max(a, b)) for a, b in listed}
    self_loops = int((edges.source == edges.target).sum())
    dup = len(edges) - len(listed)
    bad_ref = int((~edges.source.isin(nodes.segment_id) | ~edges.target.isin(nodes.segment_id)).sum())
    a = np.array([e[0] for e in listed]); b = np.array([e[1] for e in listed])
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(N + 1, N + 1))
    ncomp, comp = connected_components(g, directed=False)
    isolated = N - len(set(a) | set(b))
    deg = Counter(list(a) + list(b))
    check("C11", "Adjacency graph consistent and connected",
          listed == recomputed and self_loops == 0 and dup == 0 and bad_ref == 0 and ncomp - 1 == 1 and isolated == 0,
          f"{len(listed)} undirected edges; equal to raster adjacency={listed == recomputed}; self-loops {self_loops}; duplicates {dup}; "
          f"dangling references {bad_ref}; connected components {ncomp - 1}; isolated nodes {isolated}; degree median {int(np.median(list(deg.values())))}, max {max(deg.values())}")
    # C12 merge fixed point
    with rasterio.open(image_path) as s:
        image = s.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
    image2x = cv2.resize(image, seg.shape[::-1], interpolation=cv2.INTER_CUBIC)
    _, cp = sf.semantics(image)
    grad = sf.gradient_image(image2x)
    rag = refine_merge.RAG(seg.astype(np.int64), tab.to_dict(), nodes.set_index("segment_id").source_lineage.to_dict(),
                           image2x, cp, grad, PX_M)
    still = 0; cand = 0
    for (p, q) in rag.pair:
        if tab[p] == tab[q] and tab[p] not in (1, 2):
            cand += 1
            f = rag.features(p, q)
            parcel = "delany_parcel" in rag.src[p] or "delany_parcel" in rag.src[q]
            if refine_merge.decide(f, parcel)[0]:
                still += 1
    check("C12", "Merge refinement at a fixed point", still == 0,
          f"{cand} remaining same-class land/paved/water neighbour pairs; {still} would still qualify for merging")
    # C13 reproducibility
    check("C13", "Reproducible", repro["identical"],
          f"rerun of segment_final.py from the frozen code: {repro['differing_pixels']} differing pixels, {repro['segments_rerun']} segments")
    # C14 node position inside segment
    inside = nodes.apply(lambda r: polys[r.segment_id].contains(Point(r.node_x, r.node_y)), axis=1)
    check("C14", "Node position inside its segment", bool(inside.all()),
          f"node_x/node_y inside own polygon for {int(inside.sum())}/{len(nodes)}; plain centroid falls outside for {int((~nodes.centroid_inside).sum())} concave segments (use node_x/node_y)")
    # C15 table completeness
    req = ["segment_id", "class_id", "class", "area_m2", "centroid_x", "centroid_y", "node_x", "node_y",
           "confidence", "confidence_type", "source", "geometry_wkt"]
    nan = int(nodes[req].isna().sum().sum())
    conf_ok = bool(nodes.confidence.between(0, 1).all())
    uniq = nodes.segment_id.is_unique and nodes.node_key.is_unique
    check("C15", "Node table complete", nan == 0 and conf_ok and uniq,
          f"{len(nodes)} rows; missing values in required columns {nan}; confidence in [0,1] {conf_ok}; ids/keys unique {uniq}")
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="tile_00000")
    ap.add_argument("--year", default="2023")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    work = sf.OUT_ROOT / f"{args.year}_{args.tile}"
    run_log = json.loads((work / "run_log.json").read_text())
    image_path = Path(run_log["image"])
    frozen = sf.OUT_ROOT / f"FROZEN_{args.year}_{args.tile}_{args.version}"
    if frozen.exists():
        raise SystemExit(f"{frozen} already exists: frozen releases are never overwritten; use a new --version")

    # 1 reproducibility
    t0 = time.time()
    with rasterio.open(work / "segment_ids.tif") as s:
        current = s.read(1)
    again = rerun(args.tile, args.year, image_path, None)
    repro = {"identical": bool(np.array_equal(again, current)), "differing_pixels": int((again != current).sum()),
             "segments_rerun": int(again.max())}
    print("reproducibility:", repro, flush=True)
    if not repro["identical"]:
        raise SystemExit("Working result is not reproducible from the current code; not freezing.")

    # 2 node table
    table, geoms = build_node_table(work, args.tile, args.year)

    # 3 freeze
    frozen.mkdir(parents=True)
    for f in RELEASE_FILES:
        if (work / f).exists():
            shutil.copy2(work / f, frozen / f)
    table.to_csv(frozen / "nodes_final.csv", index=False)
    with rasterio.open(work / "segment_ids.tif") as s:
        crs_wkt = s.crs.to_wkt()
    write_node_gpkg(frozen / "nodes_final.gpkg", table, geoms, crs_wkt)
    (frozen / "code").mkdir()
    for f in CODE_FILES:
        shutil.copy2(HERE / f, frozen / "code" / f)

    # 4 validate the frozen copy
    results = validate(frozen, image_path, repro)
    status = "PASS" if all(r["result"] == "PASS" for r in results) else "FAIL"
    report = {"frozen_release": frozen.name, "tile": args.tile, "year": args.year, "status": status,
              "segments": int(len(table)), "checks": results,
              "segments_by_class": table["class"].value_counts().to_dict(),
              "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    (frozen / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [f"# Validation: {frozen.name}", "", f"**Status: {status}** ({len(table)} segments)", "",
          "| Check | Result | Detail |", "|---|---|---|"]
    md += [f"| {r['id']} {r['check']} | {r['result']} | {r['detail']} |" for r in results]
    (frozen / "validation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # manifest + read-only
    manifest = {"release": frozen.name, "created": report["validated_at"], "status": status,
                "input_image": {"path": str(image_path), "sha256": sha256(image_path)},
                "models": {k: {"path": str(p), "sha256": sha256(p)} for k, p in MODELS.items()},
                "parameters": {"segment_final": sf.P, "refine_merge": refine_merge.M},
                "files": {str(p.relative_to(frozen)).replace("\\", "/"): sha256(p) for p in sorted(frozen.rglob("*")) if p.is_file()}}
    (frozen / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for p in frozen.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IREAD)
    print(json.dumps({"status": status, "frozen": str(frozen)}, indent=1))
    for r in results:
        print(f"{r['id']:4s} {r['result']:5s} {r['check']:42s} {r['detail']}")


if __name__ == "__main__":
    main()
