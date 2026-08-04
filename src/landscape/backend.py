"""Compatibility aliases for the semantic backend now owned by Stage 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.stage2.semantic import (
    FakeSemanticBackend,
    OpenEarthMapBackend,
    SemanticBackend,
    create_semantic_backend,
    timed_semantic_inference,
)


LandcoverBackend = SemanticBackend
FakeLandcoverBackend = FakeSemanticBackend


def create_landcover_backend(cfg: dict[str, Any]) -> LandcoverBackend:
    """Legacy name; the unified runner calls ``create_semantic_backend``."""

    return create_semantic_backend(cfg, Path.cwd())


def timed_inference(
    backend: LandcoverBackend,
    image: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return timed_semantic_inference(backend, image)


__all__ = [
    "FakeLandcoverBackend",
    "LandcoverBackend",
    "OpenEarthMapBackend",
    "create_landcover_backend",
    "timed_inference",
]
