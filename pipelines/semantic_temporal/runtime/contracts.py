"""Immutable coordinate, input and scientific artifact contracts."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class GridSpec:
    crs: str | None
    affine: tuple[float, float, float, float, float, float]
    units: Literal['metres', 'pixels']
    physical_pixel_size: tuple[float, float] | None = None


@dataclass(frozen=True)
class TileInput:
    key: str
    year: int
    image: Path
    metadata: Path | None
    source_id: str
    offset_xy: tuple[int, int]
    shape_hw: tuple[int, int]
    grid: GridSpec
    valid_extent_hw: tuple[int, int]


@dataclass(frozen=True)
class YearInput:
    year: int
    tiles: tuple[TileInput, ...]
    source_raster: Path | None = None


@dataclass(frozen=True)
class RunContext:
    output: Path
    years: tuple[YearInput, ...]
    options: dict
    models: dict
    fingerprint: str


@dataclass(frozen=True)
class SegmentationArtifact:
    year: int
    tile_key: str
    directory: Path
    grid: GridSpec
    shape_hw: tuple[int, int]
    paths: dict[str, Path]
    lineage: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphArtifact:
    year: int
    method: str
    directory: Path
    grid: GridSpec
    paths: dict[str, Path]


@dataclass(frozen=True)
class TemporalArtifact:
    years: tuple[int, int]
    directory: Path
    paths: dict[str, Path]


CLASSES = {0:'unknown', 1:'building', 2:'road', 3:'tree', 4:'grass_shrub',
           5:'cropland', 6:'bare_soil', 7:'paved_developed', 8:'water'}
SCHEMA = 'satellitegraph.semantic-temporal.v1'
