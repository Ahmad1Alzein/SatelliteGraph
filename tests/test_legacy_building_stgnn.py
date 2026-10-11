"""Optional legacy building spatio-temporal GNN: integrity, staging, verification and CLI wiring.

The end-to-end test needs the hash-verified input bundle (data/legacy_building_stgnn, built with
`python -m src legacy-building-gnn bundle`) and is skipped without it."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipelines.legacy_building_stgnn import runner as R
from src.cli import parser

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("LEGACY_BUILDING_STGNN_BUNDLE", ROOT / "data" / "legacy_building_stgnn"))


def test_original_code_is_byte_identical_to_frozen_release():
    assert R.check_code() == []
    manifest = json.loads((R.REFERENCE / "code_manifest.json").read_text())["files"]
    assert {"gnn_change.py", "thesis_figures.py", "validation/gnn_predictions_tile_viz.py"} <= set(manifest)


def test_legacy_gnn_matches_frozen_study_package_copy():
    package_copy = ROOT / "pipelines" / "semantic_temporal" / "pipeline" / "gnn_change.py"
    assert R.sha256(R.ORIGINAL / "gnn_change.py") == R.sha256(package_copy)


def test_reference_records_frozen_metrics_and_all_tiles():
    ref = json.loads((R.REFERENCE / "output_reference.json").read_text())
    full = ref["summary_without_timings"]["models"]["gnn_full"]
    assert (full["ap"], full["precision"], full["recall"], full["f1"]) == (0.53, 0.507, 0.6, 0.549)
    assert set(ref["tile_figures"]) == set(R.TILES)
    assert ref["tile_figures"]["tile_00002"]["reference"].startswith("frozen original")
    assert set(ref["strip_figures"]) == set(R.STRIP_FIGURES)


def _fake_source(root: Path) -> tuple[Path, Path]:
    strip = root / "graphs" / R.FROZEN_STRIP
    for rel in ("stgraph_nodes.csv", "2013/segment_ids.tif", "2023/image_2x.tif", "MANIFEST.json"):
        (strip / rel).parent.mkdir(parents=True, exist_ok=True); (strip / rel).write_text(rel)
    for rel in ("code/gnn_change.py", "thesis_figures/fig5c.png", "labels/x.csv"):
        (strip / rel).parent.mkdir(parents=True, exist_ok=True); (strip / rel).write_text("excluded")
    (strip / "gnn").mkdir()
    for name in R.GNN_FILES:
        (strip / "gnn" / name).write_text(name)
    for tile in R.TILES:
        d = root / "graphs" / f"change_stgraph_2013_2023_{tile}"
        (d / "audit").mkdir(parents=True)
        (d / "nodes_change.csv").write_text(tile); (d / "audit" / "A_building_changes.csv").write_text("a"); (d / "audit" / "A.jpg").write_text("j")
    meta = root / "metadata.csv"; meta.write_text("tile_id")
    return root, meta


def test_bundle_copies_only_inputs_and_detects_tampering(tmp_path):
    source, meta = _fake_source(tmp_path / "src")
    bundle = tmp_path / "bundle"
    R.bundle_inputs(source, meta, bundle, manifest={})
    files = R._files(bundle)
    assert "strip/stgraph_nodes.csv" in files and "strip/2013/segment_ids.tif" in files
    assert not any(k.startswith(("strip/gnn", "strip/code", "strip/thesis_figures", "strip/labels")) for k in files)
    assert "frozen_gnn/predictions.csv" in files and "labels/change_stgraph_2013_2023_tile_00006/audit/A_building_changes.csv" in files
    assert not any(k.endswith(".jpg") for k in files)
    assert R.verify_bundle(bundle, files)["status"] == "verified"
    (bundle / "strip" / "stgraph_nodes.csv").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        R.verify_bundle(bundle, files)
    with pytest.raises(FileExistsError):
        R.bundle_inputs(source, meta, bundle, manifest={})


def test_workspace_has_original_layout_and_is_never_overwritten(tmp_path):
    source, meta = _fake_source(tmp_path / "src")
    bundle = tmp_path / "bundle"; R.bundle_inputs(source, meta, bundle, manifest={})
    work = R.prepare_workspace(bundle, tmp_path / "run")
    assert R.sha256(work / "gnn_change.py") == R.sha256(R.ORIGINAL / "gnn_change.py")
    assert (work / "validation" / "gnn_predictions_tile_viz.py").is_file()
    assert (work / "graphs" / R.STRIP_NAME / "stgraph_nodes.csv").is_file()
    assert (work / "graphs" / "change_stgraph_2013_2023_tile_00000" / "nodes_change.csv").read_text() == "tile_00000"
    assert (tmp_path / "run" / "dataset" / "2023" / "metadata.csv").is_file()   # original HERE.parent/dataset layout
    with pytest.raises(FileExistsError):
        R.prepare_workspace(bundle, tmp_path / "run")


def test_pixel_hash_ignores_png_encoding_but_not_pixels(tmp_path):
    a = (np.arange(48 * 32 * 3) % 251).astype(np.uint8).reshape(48, 32, 3)
    Image.fromarray(a).save(tmp_path / "a.png", optimize=True)
    Image.fromarray(a).save(tmp_path / "b.png", compress_level=0)
    a[5, 5, 0] ^= 1
    Image.fromarray(a).save(tmp_path / "c.png")
    assert R.sha256(tmp_path / "a.png") != R.sha256(tmp_path / "b.png")
    assert R.pixel_sha256(tmp_path / "a.png") == R.pixel_sha256(tmp_path / "b.png")
    assert R.pixel_sha256(tmp_path / "a.png") != R.pixel_sha256(tmp_path / "c.png")


def test_summary_comparison_ignores_only_timings(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"labelled": 3, "models": {"gnn_full": {"ap": 0.5, "seconds": 9.0}}}))
    assert R._summary_core(p) == {"labelled": 3, "models": {"gnn_full": {"ap": 0.5}}}


def test_verify_reports_missing_outputs(tmp_path):
    (tmp_path / "work").mkdir()
    result = R.verify(tmp_path)
    assert result["status"] == "differs" and "gnn/predictions.csv" in result["failed"]


def test_run_rejects_unknown_tiles(tmp_path):
    with pytest.raises(ValueError, match="Unknown tiles"):
        R.run(tmp_path, tmp_path / "out", tiles=["tile_00009"])


def test_cli_exposes_optional_legacy_pipeline_without_changing_pipeline_command():
    args = parser().parse_args(["legacy-building-gnn", "run", "--output", "x", "--tiles", "tile_00002", "--reuse-frozen-predictions"])
    assert (args.stage, args.phase, args.tiles, args.reuse_frozen_predictions) == ("legacy-building-gnn", "run", ["tile_00002"], True)
    assert args.bundle == "data/legacy_building_stgnn"
    args = parser().parse_args(["pipeline", "run", "--segmentation", "new", "--graph", "new", "--change", "gnn", "--output", "y"])
    assert (args.stage, args.change) == ("pipeline", "gnn")


@pytest.mark.skipif(not (BUNDLE / "strip").is_dir(), reason="legacy input bundle not available")
def test_frozen_figures_regenerate_pixel_identical(tmp_path):
    R.verify_bundle(BUNDLE)
    report = R.run(BUNDLE, tmp_path / "run", tiles=["tile_00002"], reuse_frozen_predictions=True)
    assert report["verification"]["status"] == "identical_to_frozen", report["verification"]["failed"]
    assert (tmp_path / "run" / "figures" / "fig5c_gnn_change_map.png").is_file()
    assert (tmp_path / "run" / "figures" / "tile_00002_predictions_overlay.png").is_file()


def test_cli_help_runs_as_module():
    out = subprocess.run([sys.executable, "-m", "src", "legacy-building-gnn", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "verify" in out.stdout
