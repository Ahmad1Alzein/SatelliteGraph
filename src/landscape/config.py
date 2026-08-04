"""Legacy landscape selectors backed by the single root ``config.yaml``.

The public full pipeline now lives in :mod:`src.stage2.pipeline`. These names
remain so existing phase-level imports continue to read the same settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import (
    DEFAULT_CONFIG,
    landscape_config_hash,
    load_landscape_config,
    relevant_landscape_config,
    validate_landscape_config,
)
from src.stage2.classes import CLASS_NAMES


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return load_landscape_config(path)


def validate_config(cfg: dict[str, Any]) -> None:
    validate_landscape_config(cfg)


def relevant_config(cfg: dict[str, Any], phase: str) -> dict[str, Any]:
    return relevant_landscape_config(cfg, phase)


def config_hash(cfg: dict[str, Any], phase: str) -> str:
    return landscape_config_hash(cfg, phase)


__all__ = ["CLASS_NAMES", "DEFAULT_CONFIG", "config_hash", "load_config", "relevant_config", "validate_config"]
