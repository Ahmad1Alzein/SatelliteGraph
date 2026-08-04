"""Stable class and provenance identifiers for the fused Stage 2 product."""

from __future__ import annotations

from src.labels import OPENEARTHMAP_CLASS_NAMES


CLASS_NAMES = OPENEARTHMAP_CLASS_NAMES

UNKNOWN_CLASS_ID = len(CLASS_NAMES)
NODATA_CLASS_ID = 255

SOURCE_CODES = {
    "sam2_geometry_openearthmap_label": 1,
    "openearthmap_geometry_and_label": 2,
    "fusion_unknown": 3,
    "nodata": 255,
}


def class_registry() -> list[dict[str, object]]:
    """Return the versioned semantic values stored in the final raster."""

    rows = [
        {
            "id": class_id,
            "name": name,
            "source": "openearthmap_fasterseg",
            "description": "OpenEarthMap FasterSeg semantic class.",
        }
        for class_id, name in enumerate(CLASS_NAMES)
    ]
    rows.extend(
        [
            {
                "id": UNKNOWN_CLASS_ID,
                "name": "unknown",
                "source": "stage2_fusion_rule",
                "description": "Valid pixels without a semantic prediction above the configured confidence threshold.",
            },
            {
                "id": NODATA_CLASS_ID,
                "name": "nodata",
                "source": "input_validity_mask",
                "description": "Pixels outside the valid input-data footprint.",
            },
        ]
    )
    return rows
