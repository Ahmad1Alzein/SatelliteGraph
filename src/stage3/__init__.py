"""Tile-local region-adjacency graph public API."""

from __future__ import annotations

from typing import Any


__all__ = ["Stage3Result", "load_stage3_result", "run_stage3"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .pipeline import Stage3Result, load_stage3_result, run_stage3

        return {
            "Stage3Result": Stage3Result,
            "load_stage3_result": load_stage3_result,
            "run_stage3": run_stage3,
        }[name]
    raise AttributeError(name)
