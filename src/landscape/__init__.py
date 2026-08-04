"""Semantic landscape completion after Stage 2 instance segmentation."""

from .backend import LandcoverBackend, OpenEarthMapBackend, create_landcover_backend
from .completion import complete_non_tree_landscape

__all__ = [
    "LandcoverBackend",
    "OpenEarthMapBackend",
    "complete_non_tree_landscape",
    "create_landcover_backend",
]
