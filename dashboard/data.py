"""Read-only loaders for the thesis dashboard. Every number shown comes from files on disk; nothing is recomputed
into a new 'result'. Paths can be overridden with environment variables:
  SG_LEGACY_BUNDLE   frozen study inputs (default data/legacy_building_stgnn)
  SG_LEGACY_RUN      a verified legacy building ST-GNN run (default: newest artifacts/legacy_building_stgnn/*/report.json)
  SG_RUNS            folder of multi-class pipeline runs (default artifacts/shouf3)
  SG_EXTERNAL_EVAL   optional folder with the OSM building comparison CSVs (default data/external_eval, else
                     D:/new/OriginalPipeline_Improved/improve/out/compare_repo)
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {1: "building", 2: "road", 3: "tree", 4: "grass_shrub", 5: "cropland", 6: "bare_soil", 7: "paved_developed", 8: "water"}
LABELS = {"building": "Buildings", "road": "Roads", "tree": "Trees", "grass_shrub": "Grass / shrubs", "cropland": "Cropland",
          "bare_soil": "Bare soil", "paved_developed": "Paved areas", "water": "Water"}
COLORS = {1: (230, 57, 70), 2: (90, 90, 100), 3: (27, 120, 55), 4: (150, 200, 90), 5: (238, 200, 70),
          6: (196, 150, 100), 7: (170, 140, 200), 8: (40, 120, 230)}           # original study palette (segment_final.py)
CLASS_ID = {v: k for k, v in CLASSES.items()}
PIXEL_M = 0.75
TILES = [f"tile_{i:05d}" for i in range(7)]


def hexcolor(name_or_id) -> str:
    cid = name_or_id if isinstance(name_or_id, int) else CLASS_ID.get(str(name_or_id), 0)
    return "#%02x%02x%02x" % COLORS.get(cid, (128, 128, 128))


def paths() -> dict:
    legacy_root = ROOT / "artifacts" / "legacy_building_stgnn"
    runs = sorted(legacy_root.glob("*/report.json"), key=lambda p: p.stat().st_mtime)
    return {
        "bundle": Path(os.environ.get("SG_LEGACY_BUNDLE", ROOT / "data" / "legacy_building_stgnn")),
        "legacy_run": Path(os.environ["SG_LEGACY_RUN"]) if os.environ.get("SG_LEGACY_RUN") else (runs[-1].parent if runs else None),
        "runs": Path(os.environ.get("SG_RUNS", ROOT / "artifacts" / "shouf3")),
        "external_eval": Path(os.environ["SG_EXTERNAL_EVAL"]) if os.environ.get("SG_EXTERNAL_EVAL") else next(
            (p for p in (ROOT / "data" / "external_eval", Path(r"D:\new\OriginalPipeline_Improved\improve\out\compare_repo"))
             if (p / "buildings_osm_all.csv").is_file()), ROOT / "data" / "external_eval"),
    }


def available() -> dict:
    p = paths()
    return {"bundle": (p["bundle"] / "strip" / "2013" / "segment_ids.tif").is_file(),
            "legacy_run": bool(p["legacy_run"]) and (p["legacy_run"] / "report.json").is_file(),
            "runs": bool(multiclass_runs()),
            "external_eval": (p["external_eval"] / "buildings_osm_all.csv").is_file()}


# ----------------------------------------------------------------------------- frozen study (strip, 7 tiles)
@st.cache_data(show_spinner=False)
def tiles() -> pd.DataFrame:
    m = pd.read_csv(paths()["bundle"] / "dataset" / "2023" / "metadata.csv")
    m = m[m.tile_id.isin(TILES)].sort_values("tile_id")
    out = pd.DataFrame({"tile": m.tile_id, "col0": m.x0.astype(int) * 2, "width": m.width.astype(int) * 2,
                        "left": m.bounds_left, "bottom": m.bounds_bottom, "right": m.bounds_right, "top": m.bounds_top,
                        "lon": (m.wgs84_left + m.wgs84_right) / 2, "lat": (m.wgs84_bottom + m.wgs84_top) / 2})
    return out.reset_index(drop=True)


def tile_window(tile: str) -> tuple[int, int]:
    r = tiles().set_index("tile").loc[tile]
    return int(r.col0), int(r.col0 + r.width)


@st.cache_resource(show_spinner="Loading the frozen strip imagery…")
def strip_image(year: str) -> np.ndarray:
    import rasterio
    with rasterio.open(paths()["bundle"] / "strip" / year / "image_2x.tif") as s:
        return s.read([1, 2, 3]).transpose(1, 2, 0)


@st.cache_resource(show_spinner=False)
def strip_segments(year: str) -> np.ndarray:
    import rasterio
    with rasterio.open(paths()["bundle"] / "strip" / year / "segment_ids.tif") as s:
        return s.read(1)


@st.cache_data(show_spinner=False)
def strip_transform() -> tuple:
    import rasterio
    with rasterio.open(paths()["bundle"] / "strip" / "2013" / "segment_ids.tif") as s:
        return tuple(s.transform)[:6]


@st.cache_data(show_spinner=False)
def nodes(year: str) -> pd.DataFrame:
    n = pd.read_csv(paths()["bundle"] / "strip" / year / "nodes.csv")
    a, _, c, _, e, f = strip_transform()
    n["px"] = (n.node_x - c) / a
    n["py"] = (n.node_y - f) / e
    starts = tiles().col0.values                     # a node belongs to the last tile starting left of it
    n["tile"] = np.array(TILES)[np.clip(np.searchsorted(starts, n.px.values, side="right") - 1, 0, len(TILES) - 1)]
    return n


@st.cache_data(show_spinner=False)
def class_raster(year: str) -> np.ndarray:
    n = nodes(year)
    lut = np.zeros(int(n.segment_id.max()) + 1, np.uint8)
    lut[n.segment_id.values] = n["class_id"].values
    return lut[strip_segments(year)]


@st.cache_data(show_spinner=False)
def edges(year: str) -> pd.DataFrame:
    return pd.read_csv(paths()["bundle"] / "strip" / year / "edges.csv")


@st.cache_data(show_spinner=False)
def temporal_edges() -> pd.DataFrame:
    return pd.read_csv(paths()["bundle"] / "strip" / "temporal_edges.csv")


@st.cache_data(show_spinner=False)
def temporal_components() -> pd.DataFrame:
    return pd.read_csv(paths()["bundle"] / "strip" / "temporal_components.csv")


@st.cache_data(show_spinner=False)
def strip_validation() -> dict:
    return json.loads((paths()["bundle"] / "strip" / "stgraph_validation.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def strip_summary(year: str) -> dict:
    return json.loads((paths()["bundle"] / "strip" / year / "summary.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- legacy building ST-GNN (verified run)
@st.cache_data(show_spinner=False)
def legacy_report() -> dict:
    run = paths()["legacy_run"]
    return json.loads((run / "report.json").read_text(encoding="utf-8"))


def legacy_figure(name: str) -> Path:
    return paths()["legacy_run"] / "figures" / name


def legacy_tile_dir(tile: str) -> Path:
    return paths()["legacy_run"] / "work" / f"gnn_viz_{tile}"


@st.cache_data(show_spinner=False)
def legacy_predictions() -> pd.DataFrame:
    return pd.read_csv(paths()["legacy_run"] / "work" / "graphs" / "strip_tile_00000-00006" / "gnn" / "predictions.csv")


@st.cache_data(show_spinner=False)
def legacy_audit() -> pd.DataFrame:
    return pd.read_csv(paths()["legacy_run"] / "work" / "graphs" / "strip_tile_00000-00006" / "gnn" / "audit_agreement.csv")


# ----------------------------------------------------------------------------- multi-class pipeline runs
def multiclass_runs() -> dict:
    root = paths()["runs"]
    out = {}
    if not root.is_dir():
        return out
    for run in sorted(root.iterdir()):
        if (run / "temporal" / "classical_changes.json").is_file() and (run / "run.json").is_file():
            out[run.name] = run
    return out


def run_label(name: str) -> str:
    seg = {"geo": "SegFormer + Delineate Anything (new)", "old": "SAM 2.1 + FasterSeg (old)"}
    tile, _, kind = name.rpartition("_")
    return f"{tile} · {seg.get(kind, kind)}"


@st.cache_data(show_spinner=False)
def run_state(name: str) -> dict:
    return json.loads((multiclass_runs()[name] / "run.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Reading change events…")
def run_events(name: str) -> pd.DataFrame:
    doc = json.loads((multiclass_runs()[name] / "temporal" / "classical_changes.json").read_text(encoding="utf-8"))
    rows = []
    for e in doc["events"]:
        g = e["geometry"] if isinstance(e["geometry"], dict) else ast.literal_eval(e["geometry"])
        rows.append({"event_id": e["event_id"], "source_class": CLASSES.get(int(e["source_class"]), str(e["source_class"])),
                     "target_class": CLASSES.get(int(e["target_class"]), str(e["target_class"])), "transition": e.get("display_name"),
                     "status": e["status"], "season_sensitive": bool(e["season_sensitive"]), "area_m2": float(e["area"]),
                     "related_tasks": ", ".join(e["related_tasks"] if isinstance(e["related_tasks"], list) else ast.literal_eval(e["related_tasks"])),
                     "source_nodes": len(e["source_nodes"] if isinstance(e["source_nodes"], list) else ast.literal_eval(e["source_nodes"])),
                     "target_nodes": len(e["target_nodes"] if isinstance(e["target_nodes"], list) else ast.literal_eval(e["target_nodes"])),
                     "geometry": g})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_gnn(name: str) -> dict:
    return json.loads((multiclass_runs()[name] / "temporal" / "gnn" / "report.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def run_graph_counts(name: str) -> dict:
    g = json.loads((multiclass_runs()[name] / "temporal" / "graph.json").read_text(encoding="utf-8"))
    cls = pd.Series([n["class"] for n in g["nodes"]])
    yrs = pd.Series([str(n["year"]) for n in g["nodes"]])
    return {"nodes": len(g["nodes"]), "temporal_edges": len(g["temporal_edges"]), "spatial_edges": len(g["spatial_edges"]),
            "by_year_class": pd.crosstab(cls, yrs)}


def run_tile(name: str) -> str:
    return name[:10]


# ----------------------------------------------------------------------------- optional external comparison
@st.cache_data(show_spinner=False)
def external_eval() -> dict:
    d = paths()["external_eval"]
    out = {}
    for key, f in (("buildings", "buildings_osm_all.csv"), ("points", "reference_points_all.csv"), ("counts", "change_counts_all.csv")):
        if (d / f).is_file():
            out[key] = pd.read_csv(d / f)
    return out


def geometry_to_pixels(geom: dict, col0: int = 0) -> list[np.ndarray]:
    """Projected polygon(s) -> lists of (x, y) pixel rings on the strip grid, shifted to a tile window."""
    a, _, c, _, e, f = strip_transform()
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else list(geom["coordinates"])
    rings = []
    for poly in polys:
        ring = np.asarray(poly[0], float)
        rings.append(np.c_[(ring[:, 0] - c) / a - col0, (ring[:, 1] - f) / e])
    return rings
