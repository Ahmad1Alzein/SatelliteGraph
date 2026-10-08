"""Command-line entry point for SatelliteGraph processing stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DEFAULT_CONFIG, instance_config, load_config
from src.stage2.phases import PHASES
from src.stage2.pipeline import run_stage2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m src")
    commands = root.add_subparsers(dest="stage", required=True)
    stage1 = commands.add_parser("stage1",help="Optionally resample a GeoTIFF, then generate fixed-size tiles")
    stage1_phases = stage1.add_subparsers(dest="phase",required=True)
    tile = stage1_phases.add_parser("tile")
    tile.add_argument("--source",required=True)
    tile.add_argument("--data-directory",default=None)
    resolution = tile.add_mutually_exclusive_group()
    resolution.add_argument("--resolution",type=float,default=None,help="Target pixel size in metres")
    resolution.add_argument("--keep-resolution",action="store_true",help="Override config and retain the source grid")
    tile.add_argument("--target-crs",default=None,help="Projected metre CRS; defaults to source metre CRS or local UTM")
    tile.add_argument("--tile-size",type=int,default=None)
    tile.add_argument("--overlap",type=int,default=None)
    tile.add_argument("--name",default=None)
    tile.add_argument("--resampling",choices=["average","bilinear","cubic","nearest"],default=None)
    tile.add_argument("--resolution-mode",choices=["preserve","projected"],default=None,
                      help="Preserve the full footprint (default), or explicitly reproject")
    tile.add_argument("--edge-mode",choices=["shift","pad"],default=None,
                      help="Shift final tiles inward (default), or add transparent padding")
    tile.add_argument("--config",default=str(DEFAULT_CONFIG))
    tile.add_argument("--project-root",default=".")
    stage2 = commands.add_parser(
        "stage2",
        help="SAM 2 + OpenEarthMap full semantic instance segmentation",
    )
    phases = stage2.add_subparsers(dest="phase", required=True)
    for name in [*PHASES, "run"]:
        command = phases.add_parser(name)
        if name != "check-model":
            command.add_argument("--tile-id", required=True)
        command.add_argument(
            "--config",
            default=str(DEFAULT_CONFIG),
            help="Single project configuration (default: config.yaml)",
        )
        command.add_argument("--project-root", default=".")
        if name == "run":
            command.add_argument(
                "--no-progress",
                action="store_true",
                help="Disable the tqdm phase progress bar",
            )

    refinement = commands.add_parser(
        "refine",
        help="Propose, review, apply, and validate reversible Stage 2 refinements",
    )
    refinement_phases = refinement.add_subparsers(dest="phase", required=True)
    refinement_help = {
        "propose": "Generate the checksum-bound human-review bundle",
        "apply": "Apply human decisions or the configured workflow mode",
        "auto": "Run one tile unattended with selective automatic decisions",
        "validate": "Validate a completed refinement result",
    }
    for name in ("propose", "apply", "auto", "validate"):
        command = refinement_phases.add_parser(name, help=refinement_help[name])
        command.add_argument("--tile-id", required=True)
        command.add_argument(
            "--config",
            default=str(DEFAULT_CONFIG),
            help="Single project configuration (default: config.yaml)",
        )
        command.add_argument("--project-root", default=".")
        command.add_argument(
            "--no-progress",
            action="store_true",
            help="Disable the refinement progress bar",
        )
        if name == "apply":
            command.add_argument(
                "--decisions",
                default=None,
                help=(
                    "Optional reviewed decisions JSON. When omitted, deterministic "
                    "connectivity normalization and configured safe micro-unknown cleanup "
                    "are applied."
                ),
            )
            command.add_argument(
                "--mode",
                choices=("human_review", "automatic"),
                default=None,
                help="Decision provider (default: refinement.mode from config)",
            )

    refinement_batch = refinement_phases.add_parser(
        "batch",
        help="Generate review bundles or automatically refine many Stage 2 tiles",
    )
    refinement_batch.add_argument(
        "--tile-id",
        action="append",
        dest="tile_ids",
        default=None,
        help="Tile to process; repeat for several. Omit to discover all complete tiles.",
    )
    refinement_batch.add_argument(
        "--mode",
        choices=("human_review", "automatic"),
        default="automatic",
        help="Automatic publishes results; human_review generates proposal bundles",
    )
    refinement_batch.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded number of tiles processed concurrently (default: 1)",
    )
    refinement_batch.add_argument(
        "--force",
        action="store_true",
        help="Reprocess compatible completed automatic results instead of resuming",
    )
    refinement_batch.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed tile instead of recording and continuing",
    )
    refinement_batch.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Single project configuration (default: config.yaml)",
    )
    refinement_batch.add_argument("--project-root", default=".")
    refinement_batch.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the batch progress bar",
    )

    stage3 = commands.add_parser(
        "stage3",
        help="Build one tile's segment-adjacency graph from the final Stage 2 result",
    )
    stage3_phases = stage3.add_subparsers(dest="phase", required=True)
    stage3_run = stage3_phases.add_parser("run")
    stage3_run.add_argument("--tile-id", required=True)
    stage3_run.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Single project configuration (default: config.yaml)",
    )
    stage3_run.add_argument("--project-root", default=".")
    stage3_run.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm phase progress bar",
    )
    stage3_run.add_argument(
        "--refine-first",
        action="store_true",
        help="Apply configured Stage 2 refinement immediately before building the graph",
    )
    stage3_run.add_argument(
        "--refinement-decisions",
        default=None,
        help="Optional reviewed decisions JSON used with --refine-first",
    )
    stage3_run.add_argument(
        "--refinement-mode",
        choices=("human_review", "automatic"),
        default=None,
        help="Decision provider used with --refine-first (default: refinement.mode)",
    )

    stage4 = commands.add_parser(
        "stage4",
        help="Combine selected Stage 3 tile graphs and reconcile their overlaps",
    )
    stage4_phases = stage4.add_subparsers(dest="phase", required=True)
    stage4_combine = stage4_phases.add_parser("combine")
    stage4_combine.add_argument(
        "--tile-id",
        action="append",
        dest="tile_ids",
        required=True,
        help="Completed Stage 3 tile graph to include; repeat for any number of tiles",
    )
    stage4_combine.add_argument(
        "--name",
        default=None,
        help="Optional safe output collection name (default: deterministic tile-set hash)",
    )
    stage4_combine.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Single project configuration (default: config.yaml)",
    )
    stage4_combine.add_argument("--project-root", default=".")
    stage5 = commands.add_parser("stage5", help="Connect yearly Stage 4 combined graphs in world coordinates")
    stage5_phases = stage5.add_subparsers(dest="phase", required=True)
    stage5_build = stage5_phases.add_parser("build")
    stage5_build.add_argument("--series", required=True, help="JSON listing one completed Stage 4 graph per year")
    stage5_build.add_argument("--name", default=None)
    stage5_build.add_argument("--config", default=str(DEFAULT_CONFIG))
    stage5_build.add_argument("--project-root", default=".")
    image = commands.add_parser("image", help="Run the annual pipeline for exactly one PNG")
    image_phases = image.add_subparsers(dest="phase", required=True)
    image_run = image_phases.add_parser("run", help="Segment, refine, and graph one PNG")
    image_run.add_argument("--image", required=True, help="The one PNG image to process")
    image_run.add_argument("--year", required=True, type=int)
    image_run.add_argument("--output", required=True)
    image_run.add_argument("--source-raster", default=None, help="Georeferenced GeoTIFF, if not beside the PNG")
    image_run.add_argument("--metadata", default=None, help="Tile offset CSV, if not beside the PNG")
    image_run.add_argument("--config", default=str(DEFAULT_CONFIG))
    image_run.add_argument("--project-root", default=".")
    image_run.add_argument("--no-progress", action="store_true")
    image_compare = image_phases.add_parser("compare", help="Compare one completed PNG output in each year")
    image_compare.add_argument("--first", required=True, help="Earlier completed image or annual output folder")
    image_compare.add_argument("--second", required=True, help="Later completed image or annual output folder")
    image_compare.add_argument("--tile", default=None,
                               help="PNG name when each output contains several tiles, e.g. tile_00000.png")
    image_compare.add_argument("--output", required=True)
    image_compare.add_argument("--registration-json", default=None)
    image_compare.add_argument("--boundary-tolerance", type=float, default=None)
    image_compare.add_argument("--config", default=str(DEFAULT_CONFIG))
    image_compare.add_argument("--project-root", default=".")
    image_compare.add_argument("--no-progress", action="store_true")
    folder = commands.add_parser("folder", help="Run all stages for one tile folder, then compare yearly results")
    folder_phases = folder.add_subparsers(dest="phase", required=True)
    folder_run = folder_phases.add_parser("run", help="Segment, refine, graph, and combine every tile in one folder")
    folder_run.add_argument("--folder", required=True, help="Folder of PNG tiles with metadata.csv and a source GeoTIFF")
    folder_run.add_argument("--year", required=True, type=int)
    folder_run.add_argument("--output", required=True)
    folder_run.add_argument("--source-raster", default=None, help="GeoTIFF for tile coordinates, if outside the folder")
    folder_run.add_argument("--metadata", default=None, help="Tile offset CSV, if outside the folder")
    folder_run.add_argument("--config", default=str(DEFAULT_CONFIG))
    folder_run.add_argument("--project-root", default=".")
    folder_run.add_argument("--no-progress", action="store_true")
    folder_compare = folder_phases.add_parser("compare", help="Make a temporal graph from two completed folder runs")
    folder_compare.add_argument("--first", required=True, help="Completed annual output for the earlier year")
    folder_compare.add_argument("--second", required=True, help="Completed annual output for the later year")
    folder_compare.add_argument("--output", required=True)
    registration = folder_compare.add_mutually_exclusive_group()
    registration.add_argument("--registration-json", default=None, help="Measured cross-year control points JSON")
    registration.add_argument("--register-images", action="store_true",
                              help="Align geographically overlapping saved PNG tiles with guided image matching")
    folder_compare.add_argument("--boundary-tolerance", type=float, default=None,
                                help="Ignore change within this spatial distance in metres")
    folder_compare.add_argument("--config", default=str(DEFAULT_CONFIG))
    folder_compare.add_argument("--project-root", default=".")
    folder_compare.add_argument("--no-progress", action="store_true")
    _add_pipeline_parser(commands)
    return root


def _add_pipeline_parser(commands) -> None:
    """`pipeline`: selectable segmentation (old|new) -> graph (old|new) -> change (classical|gnn)."""
    pipeline = commands.add_parser(
        "pipeline",
        help="Selectable pipeline: segmentation old|new -> graph old|new -> change detection classical|gnn",
    )
    phases = pipeline.add_subparsers(dest="phase", required=True)

    def common(p):
        p.add_argument("--output", required=True, help="Run folder, e.g. artifacts/runs/<name> (one folder per run)")
        p.add_argument("--project-root", default=None)
        new = p.add_argument_group("semantic temporal model inputs")
        new.add_argument("--data", default=None, help="Folder with dataset/ and aligned/")
        new.add_argument("--checkpoints", default=None,
                         help="Folder with satlens-segformer/ and DelineateAnythingv2.pt")
        new.add_argument("--skip-input-check", action=argparse.BooleanOptionalAction, default=None,
                         help="Allow inputs other than the reference dataset")
        old = p.add_argument_group("shared inputs for both methods")
        inputs=old.add_mutually_exclusive_group()
        inputs.add_argument("--folders", nargs='+', default=None, help="One or two metadata-based image folders, earlier year first")
        inputs.add_argument("--images", nargs='+', default=None, help="One or two individual PNGs with placement metadata, or georeferenced RGB GeoTIFFs")
        old.add_argument("--years", nargs='+', type=int, default=None, help="One year per input (default: folder names)")
        old.add_argument("--source-rasters", nargs='+', default=None, help="One source GeoTIFF per input")
        old.add_argument("--metadata", nargs='+', default=None, help="One placement CSV per input")
        old.add_argument("--pixel-size", nargs='+', type=float, default=None, help="Physical metres per original pixel when no CRS is available")
        old.add_argument("--device", choices=('cpu','cuda','auto'),default=None)
        old.add_argument("--semantic-models",default=None)
        old.add_argument("--gnn-checkpoint",default=None)
        old.add_argument("--target-crs",default=None)
        old.add_argument("--target-resolution",type=float,default=None)
        old.add_argument("--inference-window",type=int,default=None)
        old.add_argument("--inference-overlap",type=int,default=None)
        old.add_argument("--tta",action=argparse.BooleanOptionalAction,default=None)
        old.add_argument("--max-pixels",type=int,default=None)
        old.add_argument("--max-comparison-pixels",type=int,default=None)
        old.add_argument("--gnn-epochs",type=int,default=None)
        old.add_argument("--gnn-hidden",type=int,default=None)
        reg = old.add_mutually_exclusive_group()
        reg.add_argument("--registration-json", default=None, help="Cross-year control points (old classical)")
        reg.add_argument("--register-images", action=argparse.BooleanOptionalAction, default=None,
                         help="Estimate cross-year registration")
        old.add_argument("--boundary-tolerance", type=float, default=None)
        old.add_argument("--config", default=None)
        progress = old.add_mutually_exclusive_group()
        progress.add_argument("--no-progress", dest="no_progress", action="store_true", default=None)
        progress.add_argument("--progress", dest="no_progress", action="store_false")

    for name, help_text, choices in (
        ("segment", "Segment both years", ("old", "new")),
        ("graph", "Build the graphs from a completed segmentation in --output", ("old", "new")),
        ("change", "Detect change from completed graphs in --output", ("classical", "gnn")),
    ):
        p = phases.add_parser(name, help=help_text)
        p.add_argument("--method", required=True, choices=choices)
        common(p)
    run = phases.add_parser("run", help="End to end: segmentation -> graph -> change detection")
    run.add_argument("--segmentation", required=True, choices=("old", "new"))
    run.add_argument("--graph", required=True, choices=("old", "new"))
    run.add_argument("--change", required=True, choices=("classical", "gnn"))
    run.add_argument('--until',choices=('segmentation','graph','change'),default='change')
    run.add_argument('--resume',action='store_true')
    common(run)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    project_root = Path(args.project_root or '.').resolve()
    try:
        if args.stage == "stage1":
            from src.io.tile_image import tile_image
            cfg = load_config(args.config)["project"]
            tiling = cfg["tiling"]
            data = Path(args.data_directory or cfg["paths"]["data_directory"])
            source = Path(args.source)
            result = tile_image(source_path=source if source.is_absolute() else project_root/source,
                data_directory=data if data.is_absolute() else project_root/data,
                target_resolution=None if args.keep_resolution else (args.resolution if args.resolution is not None else tiling.get("target_resolution_m")),
                target_crs=None if args.keep_resolution else (args.target_crs or tiling.get("target_crs")),
                tile_size=args.tile_size if args.tile_size is not None else tiling["tile_size"],
                overlap=args.overlap if args.overlap is not None else tiling["overlap"],image_name=args.name,
                resampling=args.resampling or tiling.get("resampling","average"),progress=True,
                resolution_mode=args.resolution_mode or tiling.get("resolution_mode","preserve"),
                edge_mode=args.edge_mode or tiling.get("edge_mode","shift"))
        elif args.stage == "stage2":
            if args.phase == "run":
                result = run_stage2(
                    args.tile_id,
                    project_root=project_root,
                    config=args.config,
                    show_progress=not args.no_progress,
                ).summary()
            else:
                cfg = instance_config(load_config(args.config))
                if args.phase == "check-model":
                    result = PHASES[args.phase](cfg, project_root)
                else:
                    result = PHASES[args.phase](cfg, project_root, args.tile_id)
        elif args.stage == "refine":
            # Import lazily so Stage 1/2 commands do not initialize refinement.
            from src.refinement import (
                apply_refinement,
                propose_refinement,
                validate_refinement,
            )

            common = {
                "project_root": project_root,
                "config": args.config,
                "show_progress": not args.no_progress,
            }
            if args.phase == "batch":
                from src.refinement import run_refinement_batch

                result = run_refinement_batch(
                    args.tile_ids,
                    workflow_mode=args.mode,
                    workers=args.workers,
                    resume=not args.force,
                    continue_on_error=not args.fail_fast,
                    **common,
                ).summary()
            elif args.phase == "propose":
                result = propose_refinement(args.tile_id, **common).summary()
            elif args.phase == "apply":
                result = apply_refinement(
                    args.tile_id,
                    decisions=args.decisions,
                    workflow_mode=args.mode,
                    **common,
                ).summary()
            elif args.phase == "auto":
                result = apply_refinement(
                    args.tile_id,
                    workflow_mode="automatic",
                    **common,
                ).summary()
            else:
                result = validate_refinement(args.tile_id, **common)
        elif args.stage == "stage3":
            # Import lazily so Stage 1/2 commands do not initialize Stage 3.
            from src.stage3 import run_stage3

            refinement_summary = None
            if args.refinement_decisions is not None and not args.refine_first:
                raise ValueError("--refinement-decisions requires --refine-first")
            if args.refinement_mode is not None and not args.refine_first:
                raise ValueError("--refinement-mode requires --refine-first")
            if args.refine_first:
                from src.refinement import apply_refinement

                refinement_summary = apply_refinement(
                    args.tile_id,
                    project_root=project_root,
                    config=args.config,
                    decisions=args.refinement_decisions,
                    workflow_mode=args.refinement_mode,
                    show_progress=not args.no_progress,
                ).summary()
            graph_summary = run_stage3(
                args.tile_id,
                project_root=project_root,
                config=args.config,
                show_progress=not args.no_progress,
            ).summary()
            result = (
                {
                    "status": "complete",
                    "refinement": refinement_summary,
                    "stage3": graph_summary,
                }
                if refinement_summary is not None
                else graph_summary
            )
        elif args.stage == "stage4":
            from src.stage4 import combine_tile_graphs

            result = combine_tile_graphs(
                args.tile_ids,
                project_root=project_root,
                config=args.config,
                output_name=args.name,
            ).summary()
        elif args.stage == "stage5":
            from src.stage5 import run_stage5
            result = run_stage5(args.series, project_root=project_root,
                                config=args.config, output_name=args.name)
        elif args.stage == "image":
            from src.folder_pipeline import compare_images, run_image

            if args.phase == "run":
                result = run_image(args.image, args.year, args.output,
                                   source_raster=args.source_raster, metadata=args.metadata,
                                   config=args.config, project_root=project_root,
                                   show_progress=not args.no_progress)
            else:
                result = compare_images(args.first,args.second,args.output,
                                        tile_name=args.tile,
                                        registration_json=args.registration_json,
                                        boundary_tolerance=args.boundary_tolerance,
                                        config=args.config,project_root=project_root,
                                        show_progress=not args.no_progress)
        elif args.stage == "folder":
            from src.folder_pipeline import compare_folders, run_folder
            if args.phase == "run":
                result = run_folder(args.folder,args.year,args.output,source_raster=args.source_raster,
                                    metadata=args.metadata,config=args.config,project_root=project_root,
                                    show_progress=not args.no_progress)
            else:
                result = compare_folders(args.first,args.second,args.output,
                                         registration_json=args.registration_json,
                                         register_images=args.register_images,
                                         boundary_tolerance=args.boundary_tolerance,
                                         config=args.config,project_root=project_root,
                                         show_progress=not args.no_progress)
        elif args.stage == "pipeline":
            from src import pipeline_select as ps

            def resolve(value):
                if value is None:
                    return None
                path = Path(value)
                return path if path.is_absolute() else project_root / path

            options = {
                "project_root": project_root if args.project_root is not None else None,
                "data": resolve(args.data), "checkpoints": resolve(args.checkpoints),
                "skip_input_check": args.skip_input_check,
                "folders": [resolve(f) for f in args.folders] if args.folders else None, "years": args.years,
                "images": [resolve(f) for f in args.images] if args.images else None,
                "source_rasters": [resolve(f) for f in args.source_rasters] if args.source_rasters else None,
                "metadata": [resolve(f) for f in args.metadata] if args.metadata else None,
                "registration_json": resolve(args.registration_json), "register_images": args.register_images,
                "boundary_tolerance": args.boundary_tolerance, "config": resolve(args.config),
                "no_progress": args.no_progress,
            }
            for key in ('pixel_size','device','semantic_models','gnn_checkpoint','target_crs','target_resolution','inference_window','inference_overlap','tta','max_pixels','max_comparison_pixels','gnn_epochs','gnn_hidden'):
                options[key]=getattr(args,key)
            output = resolve(args.output)
            if args.phase == "run":
                result = ps.run(args.segmentation, args.graph, args.change, output, until=args.until,resume=args.resume,**options)
            else:
                result = getattr(ps, args.phase)(args.method, output, **options)
        else:  # pragma: no cover - argparse restricts the available stages.
            raise ValueError(f"Unsupported stage: {args.stage}")
    except Exception as exc:
        raise SystemExit(
            f"{args.stage} {args.phase} failed: {type(exc).__name__}: {exc}"
        ) from exc
    print(json.dumps(result, indent=2, default=str))
