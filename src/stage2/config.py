"""Compatibility selectors backed by the single root ``config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import (
    DEFAULT_CONFIG,
    load_stage2_config,
    relevant_stage2_config,
    stable_hash,
    validate_instance_config,
)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return load_stage2_config(path)


def validate_config(cfg: dict[str, Any]) -> None:
    validate_instance_config(cfg)


def relevant_config(cfg: dict[str, Any], phase: str) -> dict[str, Any]:
    return relevant_stage2_config(cfg, phase)


__all__ = ["DEFAULT_CONFIG", "load_config", "relevant_config", "stable_hash", "validate_config"]
