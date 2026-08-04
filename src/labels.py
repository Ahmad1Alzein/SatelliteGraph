"""Model label IDs shared by configuration and Stage 2 runtime code."""

from __future__ import annotations


# Published channel order of the OpenEarthMap FasterSeg checkpoint.
OPENEARTHMAP_CLASS_NAMES = (
    "bareland",
    "rangeland",
    "developed space",
    "road",
    "tree",
    "water",
    "agriculture land",
    "buildings",
)

FUSED_CLASS_NAMES = (*OPENEARTHMAP_CLASS_NAMES, "unknown")
