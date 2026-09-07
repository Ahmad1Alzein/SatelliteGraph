"""Write the real four-tile before/after evidence and visual comparison."""
import json
from pathlib import Path

from src.stage4 import load_stage4_result
from src.stage2.artifacts import atomic_json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def main():
    root = Path.cwd()
    directory = root / "artifacts/unknown_cleanup_comparison"
    rows = []
    for index in range(4):
        tile_id = f"IKONOS_2005-002__tile_{index:05d}"
        before = json.loads((directory / "before" / f"{tile_id}-report.json").read_text())
        after = json.loads((root / "artifacts/stage2" / tile_id / "13_refinement/report.json").read_text())
        old, new = before["comparison"], after["comparison"]
        rows.append({
            "tile_id": tile_id,
            "objects_before": old["refined_segment_count"],
            "objects_after": new["refined_segment_count"],
            "unknown_objects_before": old["refined_unknown_segment_count"],
            "unknown_objects_after": new["refined_unknown_segment_count"],
            "unknown_pixels_before": old["refined_unknown_pixel_count"],
            "unknown_pixels_after": new["refined_unknown_pixel_count"],
            "additional_unknown_pixels_absorbed": old["refined_unknown_pixel_count"] - new["refined_unknown_pixel_count"],
            "validation": after["validation"],
        })
    combined = load_stage4_result(root / "artifacts/stage4/IKONOS_tiles_00000_00003/01_combined_graph")
    report = {"tiles": rows, "combined": combined.summary()}
    atomic_json(directory / "comparison.json", report)
    panels = [
        (directory / "before/four_tile_overlay.png", "Before: original cleanup, full graph"),
        (Path(combined.artifacts["graph_overlay_audit"]), "After semantic cleanup: full graph, all unresolved regions retained"),
        (Path(combined.artifacts["graph_overlay"]), "After semantic cleanup: analysis view, unresolved regions below 41 pixels omitted"),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(20, 21), constrained_layout=True)
    for axis, (path, title) in zip(axes, panels):
        with Image.open(path) as image:
            axis.imshow(np.asarray(image.convert("RGB")))
        axis.set_title(title)
        axis.set_axis_off()
    figure.savefig(directory / "four_tile_comparison.png", dpi=150)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
