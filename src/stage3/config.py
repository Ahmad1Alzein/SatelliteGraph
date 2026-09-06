"""Stage 3 configuration selectors backed by the root ``config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import (
    DEFAULT_CONFIG,
    load_stage3_config,
    select_stage3_config,
    stable_hash,
    validate_stage3_config,
)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return load_stage3_config(path)


def validate_config(config: dict[str, Any]) -> None:
    validate_stage3_config(config)


def config_hash(config: dict[str, Any]) -> str:
    """Hash a selected Stage 3 mapping, excluding internal derived keys."""
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    return stable_hash(public)


__all__ = [
    "DEFAULT_CONFIG",
    "config_hash",
    "load_config",
    "select_stage3_config",
    "stable_hash",
    "validate_config",
]
