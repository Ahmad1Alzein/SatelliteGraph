"""Command-line entry point for the unified Stage 2 pipeline."""

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
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    try:
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
    except Exception as exc:
        raise SystemExit(f"stage2 {args.phase} failed: {type(exc).__name__}: {exc}") from exc
    print(json.dumps(result, indent=2, default=str))
