"""Run annual processing for folders or individual PNGs, then compare years."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

from src.config import DEFAULT_CONFIG, load_config, validate_config
from src.stage2.artifacts import atomic_json, now


ANNUAL_PIPELINE_REVISION = "micro_unknown_omission_node_adjacency_v3"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inspect_folder(folder, *, raster=None, metadata=None, image=None):
    """Inspect georeferenced tiles, or one unreferenced PNG in local pixels."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Tile folder not found: {folder}")
    if image is None:
        tiles = sorted(folder.glob("*.png"))
    else:
        selected = Path(image).expanduser().resolve()
        if selected.parent != folder or selected.suffix.lower() != ".png" or not selected.is_file():
            raise ValueError(f"Selected image must be one PNG directly inside {folder}: {selected}")
        tiles = [selected]
    if not tiles:
        raise ValueError(f"No PNG tiles found directly in {folder}")
    if raster is None:
        options = sorted([*folder.glob("*.tif"),*folder.glob("*.tiff")])
        preferred = [item for item in options if item.stem=="source"]
        raster = preferred[0] if len(preferred)==1 else options[0] if len(options)==1 else None
    raster = Path(raster).expanduser().resolve() if raster is not None else None
    metadata = Path(metadata).expanduser().resolve() if metadata is not None else folder/"metadata.csv"
    if image is not None and raster is None and not metadata.exists():
        tile = tiles[0]
        pixels = cv2.imread(str(tile), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError(f"Unreadable RGB/RGBA tile: {tile}")
        height, width = pixels.shape[:2]
        alpha = 4 if pixels.shape[2] == 4 else ""
        valid = int(np.count_nonzero(pixels[:, :, 3])) if alpha else width * height
        row = dict(tile_id=tile.stem.removeprefix("tile_"), tile_name=tile.name,
                   x0=0, y0=0, x1=width, y1=height, width=width, height=height,
                   valid_width=width, valid_height=height, alpha_band=alpha,
                   valid_pixel_count=valid)
        return {"folder": folder, "tiles": tiles, "raster": None, "metadata": None,
                "rows": [row], "coordinate_frame": "local_pixel"}
    if raster is None or not raster.is_file():
        raise ValueError("A georeferenced source raster is required: put one .tif/.tiff in the folder or pass --source-raster")
    if not metadata.is_file():
        raise ValueError("metadata.csv with exact x0/y0 offsets is required (or pass --metadata)")
    with metadata.open(newline="",encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"tile_name","x0","y0"}.issubset(reader.fieldnames or []):
            raise ValueError("Tile metadata needs tile_name, x0, and y0 columns")
        rows = list(reader)
    names = [p.name for p in tiles]
    metadata_names = [row["tile_name"] for row in rows]
    if (len(set(metadata_names)) != len(rows)
            or (image is None and set(metadata_names) != set(names))
            or (image is not None and not set(names).issubset(metadata_names))):
        raise ValueError("Tile metadata must contain each selected PNG exactly once")
    by_name = {row["tile_name"]:row for row in rows}
    normalized = []
    with rasterio.open(raster) as source:
        if source.crs is None or source.transform is None or source.count<3:
            raise ValueError("Source raster needs a CRS, affine transform, and three RGB bands")
        for tile in tiles:
            image = cv2.imread(str(tile),cv2.IMREAD_UNCHANGED)
            if image is None or image.ndim!=3 or image.shape[2] not in (3,4):
                raise ValueError(f"Unreadable RGB/RGBA tile: {tile}")
            row = by_name[tile.name]
            try:
                x,y = int(row["x0"]),int(row["y0"])
            except (TypeError,ValueError) as exc:
                raise ValueError(f"Noninteger x0/y0 for {tile.name}") from exc
            h,w = image.shape[:2]
            try:
                valid_w = int(row["valid_width"]) if row.get("valid_width") else w
                valid_h = int(row["valid_height"]) if row.get("valid_height") else h
                declared_w = int(row["width"]) if row.get("width") else w
                declared_h = int(row["height"]) if row.get("height") else h
            except (TypeError,ValueError) as exc:
                raise ValueError(f"Invalid tile dimensions for {tile.name}") from exc
            if (declared_w,declared_h)!=(w,h) or not (0<valid_w<=w and 0<valid_h<=h):
                raise ValueError(f"Tile {tile.name} dimensions disagree with metadata")
            if x<0 or y<0 or x+valid_w>source.width or y+valid_h>source.height:
                raise ValueError(f"Tile {tile.name} valid pixels lie outside its source raster")
            if image.shape[2]==4:
                if row.get("alpha_band") and row["alpha_band"]!="4":
                    raise ValueError(f"Tile {tile.name} alpha band disagrees with metadata")
                mask = image[:,:,3]>0
                if mask[valid_h:,:].any() or mask[:,valid_w:].any():
                    raise ValueError(f"Tile {tile.name} has nontransparent pixels outside its valid extent")
                valid = int(mask.sum())
                alpha = 4
            else:
                if valid_w!=w or valid_h!=h or row.get("alpha_band"):
                    raise ValueError(f"Tile {tile.name} needs an alpha band for padded pixels")
                mask = np.ones((h,w),dtype=bool)
                valid = w*h
                alpha = ""
            if row.get("valid_pixel_count") and int(row["valid_pixel_count"])!=valid:
                raise ValueError(f"Tile {tile.name} valid-pixel count disagrees with metadata")
            source_rgb = source.read([1,2,3],window=Window(x,y,valid_w,valid_h)).transpose(1,2,0)
            tile_rgb = image[:valid_h,:valid_w,2::-1]
            if not np.array_equal(tile_rgb[mask[:valid_h,:valid_w]],source_rgb[mask[:valid_h,:valid_w]]):
                raise ValueError(f"Tile {tile.name} pixels do not match its source raster window at x0/y0")
            normalized.append(dict(tile_id=tile.stem.removeprefix("tile_"),tile_name=tile.name,
                x0=x,y0=y,x1=x+w,y1=y+h,width=w,height=h,
                valid_width=valid_w,valid_height=valid_h,alpha_band=alpha,valid_pixel_count=valid))
    return {"folder":folder,"tiles":tiles,"raster":raster,"metadata":metadata,
            "rows":normalized,"coordinate_frame":"georeferenced"}


def _annual_config(config, output, source, image_name):
    cfg = load_config(config) if not isinstance(config,dict) else copy.deepcopy(config)
    paths=cfg["project"]["paths"]
    paths.update(raw_directory=str(source.parent),tiles_directory=str(output/"inputs/tiles"),
                 metadata_directory=str(output/"inputs/metadata"),data_directory=str(output/"inputs"))
    cfg["stage2"]["output"]["root_directory"]=str(output/"stage2")
    cfg["stage3"]["output"]["root_directory"]=str(output/"stage3")
    cfg["stage4"]["input"]["root_directory"]=str(output/"stage3")
    cfg["stage4"]["output"]["root_directory"]=str(output/"stage4")
    cfg["refinement"]["mode"]="automatic"
    validate_config(cfg)
    return cfg


def _require_micro_unknown_omission(refined):
    omission = refined.report.get("micro_unknown_omission", {})
    if (omission.get("enabled") is not True
            or omission.get("maximum_area_pixels") != 40):
        raise RuntimeError(
            "Automatic refinement did not record the required sub-41-pixel "
            "unknown cleanup"
        )


def run_folder(folder, year, output, *, source_raster=None, metadata=None,
               config=DEFAULT_CONFIG, project_root=".", show_progress=True,
               selected_image=None):
    """Complete Stage 2, automatic Stage 2R, Stage 3, and Stage 4 for every PNG."""
    if isinstance(year,bool) or not isinstance(year,int) or not 1<=year<=9999:
        raise ValueError("year must be an integer in 1..9999")
    root=Path(project_root).resolve()
    path=lambda value: Path(value) if Path(value).is_absolute() else root/value
    inspected=inspect_folder(path(folder),raster=path(source_raster) if source_raster else None,
                             metadata=path(metadata) if metadata else None,
                             image=path(selected_image) if selected_image else None)
    output=path(output).resolve()
    if (output==inspected["folder"] or output in inspected["folder"].parents
            or inspected["folder"] in output.parents):
        raise ValueError("Output must be separate from the input tile folder")
    image_name=(inspected["raster"] or inspected["tiles"][0]).stem
    selected=_annual_config(path(config) if not isinstance(config,dict) else config,
                            output,inspected["raster"] or inspected["tiles"][0],image_name)
    signature={str(p):_digest(p) for p in [inspected["raster"],inspected["metadata"],*inspected["tiles"]] if p is not None}
    cfg_hash=hashlib.sha256(json.dumps(selected,sort_keys=True,default=str).encode()).hexdigest()
    ledger=output/"run.json"
    if ledger.exists():
        state=json.loads(ledger.read_text(encoding="utf-8"))
        if state.get("pipeline_revision") != ANNUAL_PIPELINE_REVISION:
            raise ValueError(
                "Existing annual run predates the current refinement and graph pipeline; "
                "choose another output folder"
            )
        if state.get("year")!=year or state.get("input_sha256")!=signature or state.get("config_sha256")!=cfg_hash:
            raise ValueError("Existing run has a different year, input content, or configuration; choose another output folder")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder already has data but no run.json: {output}")
    else:
        output.mkdir(parents=True,exist_ok=True)
        state={"status":"running","year":year,
               "pipeline_revision":ANNUAL_PIPELINE_REVISION,
               "source_folder":str(inspected["folder"]),
               "source_raster":str(inspected["raster"]) if inspected["raster"] else None,
               "source_metadata":str(inspected["metadata"]) if inspected["metadata"] else None,
               "coordinate_frame":inspected["coordinate_frame"],
               "tile_count":len(inspected["tiles"]),"tile_ids":[],"input_sha256":signature,
               "config_sha256":cfg_hash,"tiles":{},"started_at":now()}
        atomic_json(ledger,state)
    try:
        staged=output/"inputs/tiles"/image_name
        staged.mkdir(parents=True,exist_ok=True)
        csv_path=output/"inputs/metadata"/f"{image_name}.csv"
        csv_path.parent.mkdir(parents=True,exist_ok=True)
        with csv_path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(inspected["rows"][0]))
            writer.writeheader();writer.writerows(inspected["rows"])
        for tile in inspected["tiles"]:
            target=staged/tile.name
            if not target.exists() or _digest(target)!=signature[str(tile)]:
                shutil.copy2(tile,target)
        ids=[f"{image_name}__{tile.stem}" for tile in inspected["tiles"]]
        state["tile_ids"]=ids
        atomic_json(ledger,state)
    except Exception as exc:
        state.update(status="failed",error=f"{type(exc).__name__}: {exc}",updated_at=now())
        atomic_json(ledger,state)
        raise
    from src.stage2 import run_stage2
    from src.stage2.backend import create_backend
    from src.stage2.semantic import create_semantic_backend
    from src.config import instance_config, semantic_config
    from src.refinement import apply_refinement, load_refinement_result, validate_refinement
    from src.stage3 import run_stage3, load_stage3_result
    from src.stage4 import combine_tile_graphs, load_stage4_result
    sam=semantic=None
    try:
        for index,tile_id in enumerate(ids,1):
            entry=state["tiles"].get(tile_id,{})
            if entry.get("status")=="complete":
                _require_micro_unknown_omission(
                    load_refinement_result(output/"stage2"/tile_id/"13_refinement")
                )
                load_stage3_result(output/"stage3"/tile_id/"01_graph")
                validate_refinement(tile_id,project_root=root,config=selected,show_progress=False)
                continue
            state["current_step"]=f"{index}/{len(ids)} {tile_id}: segmentation"
            atomic_json(ledger,state)
            print(state["current_step"],flush=True)
            if sam is None:
                sam=create_backend(instance_config(selected),root);sam.load()
                semantic=create_semantic_backend(semantic_config(selected),root);semantic.load()
            segmented=run_stage2(tile_id,project_root=root,config=selected,
                                 sam_backend=sam,semantic_backend=semantic,show_progress=show_progress)
            entry["segmentation"]={"status":"complete","segment_count":len(segmented.segments)}
            state["current_step"]=f"{tile_id}: automatic refinement and noise cleanup"
            atomic_json(ledger,state)
            refined=apply_refinement(tile_id,project_root=root,config=selected,
                                     workflow_mode="automatic",show_progress=show_progress)
            _require_micro_unknown_omission(refined)
            entry["refinement"]={"status":"complete","output_directory":str(refined.output_directory)}
            validated=validate_refinement(tile_id,project_root=root,config=selected,show_progress=False)
            entry["refinement_validation"]={"status":validated["status"]}
            state["current_step"]=f"{tile_id}: tile graph"
            atomic_json(ledger,state)
            graph=run_stage3(tile_id,project_root=root,config=selected,show_progress=show_progress)
            entry["stage3"]={"status":"complete","node_count":len(graph.nodes),"edge_count":len(graph.edges),
                             "output_directory":str(graph.output_directory)}
            entry["status"]="complete"
            state["tiles"][tile_id]=entry
            atomic_json(ledger,state)
        if state.get("status")=="complete":
            combined=load_stage4_result(state["combined_graph_directory"])
        else:
            state["current_step"]="Combine all tile graphs"
            atomic_json(ledger,state)
            combined=combine_tile_graphs(ids,project_root=root,config=selected,output_name=f"year_{year}")
        overlay=output/"full_image_with_graph.png"
        analysis_overlay=output/"full_image_with_analysis_graph.png"
        shutil.copy2(combined.artifacts["graph_overlay_audit"],overlay)
        shutil.copy2(combined.artifacts["graph_overlay"],analysis_overlay)
        for key in ("error","current_step","updated_at"):
            state.pop(key,None)
        state.update(status="complete",completed_at=now(),combined_graph_directory=str(combined.output_directory),
                     full_image_with_graph=str(overlay),full_image_with_analysis_graph=str(analysis_overlay),
                     combined_node_count=len(combined.nodes),
                     combined_edge_count=len(combined.edges))
        atomic_json(ledger,state)
        return {key:state[key] for key in ("status","year","tile_count","combined_graph_directory",
                                           "full_image_with_graph","full_image_with_analysis_graph",
                                           "combined_node_count","combined_edge_count")}
    except Exception as exc:
        state.update(status="failed",error=f"{type(exc).__name__}: {exc}",updated_at=now())
        atomic_json(ledger,state)
        raise


def run_image(image, year, output, *, source_raster=None, metadata=None,
              config=DEFAULT_CONFIG, project_root=".", show_progress=True):
    """Run all annual stages for one PNG, with optional geospatial sidecars."""
    root = Path(project_root).resolve()
    selected = Path(image).expanduser()
    selected = (selected if selected.is_absolute() else root/selected).resolve()
    if selected.suffix.lower() != ".png" or not selected.is_file():
        raise ValueError(f"Input image must be an existing PNG: {selected}")
    return run_folder(selected.parent, year, output, source_raster=source_raster,
                      metadata=metadata, config=config, project_root=root,
                      show_progress=show_progress, selected_image=selected)


def _staged_preview_tiles(folder, annual, source_image_id):
    """Find the exact PNGs and source-pixel windows saved by an annual run."""
    metadata = folder/"inputs"/"metadata"/f"{source_image_id}.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"Saved tile metadata is missing: {metadata}")
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {f"{source_image_id}__{Path(row['tile_name']).stem}": row for row in rows}
    if len(by_id) != len(rows) or not set(annual["tile_ids"]).issubset(by_id):
        raise ValueError("Saved tile metadata does not match this annual run")
    tiles = {}
    for tile_id in annual["tile_ids"]:
        row = by_id[tile_id]
        image = folder/"inputs"/"tiles"/source_image_id/row["tile_name"]
        if not image.is_file():
            raise FileNotFoundError(f"Saved tile PNG is missing: {image}")
        expected = {digest for original, digest in annual.get("input_sha256", {}).items()
                    if Path(original).name == image.name}
        if expected and _digest(image) not in expected:
            raise ValueError(f"Saved tile PNG changed since graph creation: {image}")
        width, height = int(row["width"]), int(row["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid saved tile dimensions for {image.name}")
        tiles[tile_id] = {"path": str(image), "tile_id": tile_id,
                          "tile_name": image.name, "x0": int(row["x0"]),
                          "y0": int(row["y0"]), "width": width, "height": height}
    return metadata, tiles


def compare_folders(first, second, output, *, registration_json=None,
                    register_images=False, boundary_tolerance=None,
                    config=DEFAULT_CONFIG, project_root=".", show_progress=True):
    """Read two completed annual runs and build the final Stage 5 graph."""
    if registration_json and register_images:
        raise ValueError("Choose either registration control points or automatic image registration")
    root=Path(project_root).resolve()
    path=lambda value: (Path(value) if Path(value).is_absolute() else root/value).resolve()
    sources=[]
    run_inputs=[]
    from src.stage4 import load_stage4_result
    for folder in (first,second):
        folder=path(folder)
        record=folder/"run.json"
        if not record.is_file():
            raise ValueError(f"Only a complete folder run can be compared: {folder}")
        annual=json.loads(record.read_text(encoding="utf-8"))
        tile_ids=annual.get("tile_ids")
        tiles=annual.get("tiles")
        if (annual.get("status")!="complete" or not isinstance(tile_ids,list) or not tile_ids
                or len(tile_ids)!=annual.get("tile_count") or not isinstance(tiles,dict)
                or set(tile_ids)!=set(tiles)
                or any(tiles[tile_id].get("status")!="complete" for tile_id in tile_ids)):
            raise ValueError(f"Only a complete folder run can be compared: {folder}")
        if annual.get("coordinate_frame") == "local_pixel" or not annual.get("source_raster"):
            raise ValueError(f"Temporal comparison needs a real georeference for {folder}; "
                             "a PNG-only run has local pixel coordinates")
        graph_dir=folder/"stage4"/f"year_{annual['year']}"/"01_combined_graph"
        combined=load_stage4_result(graph_dir)
        if (graph_dir.parent.parent.parent!=folder
                or combined.graph["selected_tile_ids"]!=sorted(tile_ids)
                or combined.graph["source_image_id"]!=Path(annual["source_raster"]).stem):
            raise ValueError(f"Combined graph does not match the complete folder run: {folder}")
        if register_images:
            image=Path(annual["source_raster"])
            expected=annual.get("input_sha256",{}).get(str(image))
            if not expected or not image.is_file() or _digest(image)!=expected:
                raise ValueError(f"Annual source raster changed since graph creation: {image}")
        saved_metadata, preview_tiles = _staged_preview_tiles(folder, annual, combined.graph["source_image_id"])
        sources.append({"year":int(annual["year"]),"graph_directory":str(graph_dir),
                        "image_path":str(annual["source_raster"]),
                        "preview_tiles":list(preview_tiles.values())})
        run_inputs.extend((record,graph_dir/"manifest.json",graph_dir/"graph.json",
                           saved_metadata,*(Path(tile["path"]) for tile in preview_tiles.values())))
    if sources[0]["year"]>=sources[1]["year"]:
        raise ValueError("--first must be the earlier year and --second the later year")
    if registration_json:
        registration=json.loads(path(registration_json).read_text(encoding="utf-8"))
        sources[1]["registration"]=registration
    elif register_images:
        sources[1]["register_to_reference"]=True
    cfg=load_config(path(config)) if not isinstance(config,dict) else copy.deepcopy(config)
    if boundary_tolerance is not None:
        cfg["stage5"]["changes"]["boundary_tolerance_m"]=boundary_tolerance
    output=path(output)
    series={"name":f"{sources[0]['year']}_{sources[1]['year']}","snapshots":sources}
    if cfg.get("stage5",{}).get("target_crs"):
        series["target_crs"]=cfg["stage5"]["target_crs"]
    signature=hashlib.sha256(json.dumps({
        "series":series,"stage5_config":cfg.get("stage5",{}),
        "run_inputs":{str(item):_digest(item) for item in run_inputs},
    },sort_keys=True,default=str).encode()).hexdigest()
    manifest=output/"run.json"
    if manifest.exists():
        previous=json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("status") not in {"failed","running"} or previous.get("input_sha256")!=signature:
            raise FileExistsError(f"Comparison output already exists with different inputs or is complete: {output}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Comparison output directory is not empty: {output}")
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/"series.json",series)
    state={"status":"running","years":[s["year"] for s in sources],"series":str(output/"series.json"),
           "input_sha256":signature,"started_at":now()}
    atomic_json(manifest,state)
    try:
        from src.stage5 import build_temporal_graph
        graph=build_temporal_graph(sources,target_crs=series.get("target_crs"),
            policy=cfg.get("stage5",{}).get("matching",{}),
            change_policy=cfg.get("stage5",{}).get("changes",{}),
            output_directory=output/"01_temporal_graph", show_progress=show_progress)
        if graph["comparisons"][0]["common_coverage_m2"]<=0:
            raise ValueError("The yearly graphs have no common observed coverage after registration")
        from src.stage5 import load_stage5_result
        load_stage5_result(output/"01_temporal_graph")
        state.update(status="complete",completed_at=now(),
                     graph_directory=str(output/"01_temporal_graph"),
                     temporal_graph_image=str(output/"01_temporal_graph"/"temporal_graph.png"),
                     report_path=str(output/"01_temporal_graph"/"report.json"),
                     common_coverage_m2=graph["comparisons"][0]["common_coverage_m2"],
                     node_count=graph["graph"]["node_count"],
                     spatial_edge_count=graph["graph"]["spatial_edge_count"],
                     temporal_edge_count=graph["graph"]["temporal_edge_count"],
                     change_edge_count=graph["graph"]["change_edge_count"],
                     change_event_count=graph["graph"]["change_event_count"])
        atomic_json(manifest,state)
        return state
    except Exception as exc:
        state.update(status="failed",error=f"{type(exc).__name__}: {exc}",updated_at=now())
        atomic_json(manifest,state)
        raise


def compare_images(first, second, output, *, tile_name=None, registration_json=None,
                   boundary_tolerance=None, config=DEFAULT_CONFIG, project_root=".",
                   show_progress=True):
    """Compare one completed PNG graph per year, without rerunning segmentation.

    Single-image run folders need no selector. For annual folders containing
    several tiles, --tile selects the same saved PNG name in each year.
    """
    from src.stage4 import load_stage4_result
    from src.stage5 import build_temporal_graph, load_stage5_result
    from src.stage5.subset import clip_combined_graph_to_tile

    root = Path(project_root).resolve()
    path = lambda value: (Path(value) if Path(value).is_absolute() else root/value).resolve()
    output = path(output)
    if tile_name is not None:
        tile_name = Path(tile_name).name
        if not tile_name.lower().endswith(".png"):
            tile_name += ".png"
    prepared, run_inputs = [], []
    for source in (first, second):
        folder = path(source)
        if output == folder or folder in output.parents or output in folder.parents:
            raise ValueError("Comparison output must be separate from annual inputs")
        record = folder/"run.json"
        if not record.is_file():
            raise FileNotFoundError(f"Completed image output has no run.json: {folder}")
        annual = json.loads(record.read_text(encoding="utf-8"))
        ids, tiles = annual.get("tile_ids"), annual.get("tiles")
        if (annual.get("status") != "complete" or not isinstance(ids, list) or not ids
                or len(ids) != annual.get("tile_count") or not isinstance(tiles, dict)
                or set(ids) != set(tiles)
                or any(tiles[tile_id].get("status") != "complete" for tile_id in ids)):
            raise ValueError(f"Only completed image or annual outputs can be compared: {folder}")
        if annual.get("coordinate_frame") == "local_pixel" or not annual.get("source_raster"):
            raise ValueError("Single-image temporal comparison needs a real georeference")
        graph_dir = folder/"stage4"/f"year_{annual['year']}"/"01_combined_graph"
        combined = load_stage4_result(graph_dir)
        if combined.graph["selected_tile_ids"] != sorted(ids):
            raise ValueError(f"Combined graph does not match the completed run: {folder}")
        saved_metadata, preview_tiles = _staged_preview_tiles(folder, annual, combined.graph["source_image_id"])
        if tile_name is None:
            if len(preview_tiles) != 1:
                raise ValueError("Multi-tile annual outputs require --tile tile_00000.png")
            selected_id = next(iter(preview_tiles))
        else:
            matching = [key for key, value in preview_tiles.items() if value["tile_name"] == tile_name]
            if len(matching) != 1:
                raise ValueError(f"Saved tile {tile_name!r} is absent from {folder}")
            selected_id = matching[0]
        tile = preview_tiles[selected_id]
        extent = (tile["x0"], tile["y0"], tile["x0"]+tile["width"], tile["y0"]+tile["height"])
        document = {"schema_version":"satellitegraph.stage4.combined-graph.v1",
                    "graph":combined.graph,"nodes":combined.nodes,"edges":combined.edges}
        selected = clip_combined_graph_to_tile(document, selected_id, extent)
        prepared.append({"year":int(annual["year"]), "document":selected,
                         "preview_tiles":[tile], "tile_name":tile["tile_name"]})
        run_inputs.extend((record,graph_dir/"graph.json",graph_dir/"manifest.json",
                           saved_metadata,Path(tile["path"])))
    if prepared[0]["year"] >= prepared[1]["year"]:
        raise ValueError("--first must be the earlier year and --second the later year")
    if prepared[0]["tile_name"] != prepared[1]["tile_name"]:
        raise ValueError("The selected yearly PNGs must have the same tile name")
    cfg = load_config(path(config)) if not isinstance(config, dict) else copy.deepcopy(config)
    if boundary_tolerance is not None:
        cfg["stage5"]["changes"]["boundary_tolerance_m"] = boundary_tolerance
    if registration_json:
        registration_path = path(registration_json)
        prepared[1]["registration"] = json.loads(registration_path.read_text(encoding="utf-8"))
        run_inputs.append(registration_path)
    signature = hashlib.sha256(json.dumps({
        "years":[item["year"] for item in prepared], "tile_name":prepared[0]["tile_name"],
        "stage5_config":cfg.get("stage5",{}),
        "run_inputs":{str(item):_digest(item) for item in run_inputs},
    },sort_keys=True,default=str).encode()).hexdigest()
    manifest = output/"run.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("status") not in {"failed","running"} or previous.get("input_sha256") != signature:
            raise FileExistsError(f"Comparison output already exists with different inputs or is complete: {output}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Comparison output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state = {"status":"running","years":[item["year"] for item in prepared],
             "tile_name":prepared[0]["tile_name"],"input_sha256":signature,
             "started_at":now()}
    atomic_json(manifest,state)
    try:
        for item in prepared:
            graph_path = output/"inputs"/f"year_{item['year']}"/"graph.json"
            atomic_json(graph_path,item["document"])
            item["input_paths"] = [str(graph_path), item["preview_tiles"][0]["path"]]
        graph = build_temporal_graph(prepared,
            target_crs=cfg.get("stage5",{}).get("target_crs"),
            policy=cfg.get("stage5",{}).get("matching",{}),
            change_policy=cfg.get("stage5",{}).get("changes",{}),
            output_directory=output/"01_temporal_graph",show_progress=show_progress)
        if graph["comparisons"][0]["common_coverage_m2"] <= 0:
            raise ValueError("The selected tiles have no common observed coverage")
        load_stage5_result(output/"01_temporal_graph")
        state.update(status="complete",completed_at=now(),
                     graph_directory=str(output/"01_temporal_graph"),
                     temporal_graph_image=str(output/"01_temporal_graph"/"temporal_graph.png"),
                     report_path=str(output/"01_temporal_graph"/"report.json"),
                     common_coverage_m2=graph["comparisons"][0]["common_coverage_m2"],
                     node_count=graph["graph"]["node_count"],
                     spatial_edge_count=graph["graph"]["spatial_edge_count"],
                     temporal_edge_count=graph["graph"]["temporal_edge_count"],
                     change_edge_count=graph["graph"]["change_edge_count"],
                     change_event_count=graph["graph"]["change_event_count"])
        atomic_json(manifest,state)
        return state
    except Exception as exc:
        state.update(status="failed",error=f"{type(exc).__name__}: {exc}",updated_at=now())
        atomic_json(manifest,state)
        raise
