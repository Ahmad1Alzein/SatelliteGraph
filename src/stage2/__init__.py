"""Unified SAM 2 + OpenEarthMap Stage 2 public API."""

from __future__ import annotations

from typing import Any


__all__ = ["Stage2Result", "load_stage2_result", "run_stage2"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .pipeline import Stage2Result, load_stage2_result, run_stage2

        return {
            "Stage2Result": Stage2Result,
            "load_stage2_result": load_stage2_result,
            "run_stage2": run_stage2,
        }[name]
    raise AttributeError(name)
