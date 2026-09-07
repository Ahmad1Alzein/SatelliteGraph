"""Single configuration source for every SatelliteGraph stage."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.labels import FUSED_CLASS_NAMES as FUSED_LABEL_NAMES
from src.labels import OPENEARTHMAP_CLASS_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# OpenEarthMap FasterSeg emits these IDs. Keep this order in code: changing it
# in a user-editable YAML file would silently relabel model output pixels.
CLASS_NAMES = list(OPENEARTHMAP_CLASS_NAMES)
FUSED_CLASS_NAMES = list(FUSED_LABEL_NAMES)

NORMALIZATIONS = {"none", "fixed_range", "min_max", "percentile", "per_band_percentile"}
BACKGROUNDS = {"leave_unassigned", "single_background", "connected_components"}
DUPE_POLICIES = {
    "keep_higher_predicted_iou",
    "keep_higher_stability",
    "keep_larger",
    "keep_smaller",
    "keep_all",
    "merge_union",
}
NESTED_POLICIES = {"keep_all", "keep_parent", "keep_child", "remove_near_identical_only"}
STAGE3_ADJACENCY_METHODS = {"shared_edge"}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the one project configuration document."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Project configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(document)
    result = copy.deepcopy(document)
    result["_config_path"] = str(config_path.resolve())
    return result


def validate_config(config: dict[str, Any]) -> None:
    """Validate the unified project configuration document.

    Later stages remain optional here so existing programmatic Stage 2
    configurations keep working. Their entry points require the relevant
    section through the stage-specific selector.
    """
    _require_mapping(config, "configuration")
    _require_keys(config, {"project", "stage2"}, "configuration")
    validate_project_config(config["project"])
    stage2 = _require_mapping(config["stage2"], "stage2")
    _require_keys(
        stage2,
        {
            "instance_model",
            "semantic_model",
            "preprocessing",
            "filtering",
            "deduplication",
            "coverage",
            "fusion",
            "visualization",
            "output",
        },
        "stage2",
    )
    validate_instance_config(_instance_view(stage2))
    validate_semantic_config(stage2["semantic_model"])
    validate_fusion_config(stage2["fusion"])
    _validate_visualization(stage2["visualization"])
    _validate_output(stage2["output"])
    if "refinement" in config:
        validate_refinement_config(config["refinement"])
    if "stage3" in config:
        validate_stage3_config(config["stage3"])
    if "stage4" in config:
        validate_stage4_config(config["stage4"])


def validate_project_config(config: dict[str, Any]) -> None:
    config = _require_mapping(config, "project")
    _require_keys(config, {"paths", "tiling"}, "project")
    paths = _require_mapping(config["paths"], "project.paths")
    _require_keys(
        paths,
        {"data_directory", "raw_directory", "tiles_directory", "metadata_directory"},
        "project.paths",
    )
    for name in ("data_directory", "raw_directory", "tiles_directory", "metadata_directory"):
        if not isinstance(paths[name], str) or not paths[name].strip():
            raise ValueError(f"project.paths.{name} must be a non-empty path string")
    tiling = _require_mapping(config["tiling"], "project.tiling")
    _require_keys(tiling, {"tile_size", "overlap"}, "project.tiling")
    tile_size = _positive_int(tiling["tile_size"], "project.tiling.tile_size")
    overlap = _nonnegative_int(tiling["overlap"], "project.tiling.overlap")
    if overlap >= tile_size:
        raise ValueError("project.tiling.overlap must be smaller than tile_size")


def validate_instance_config(config: dict[str, Any]) -> None:
    """Validate the flattened configuration consumed by instance phases."""
    config = _require_mapping(config, "stage2 instance configuration")
    _require_keys(
        config,
        {
            "backend",
            "preprocessing",
            "automatic_mask_generator",
            "filtering",
            "deduplication",
            "coverage",
            "visualization",
            "output",
        },
        "stage2 instance configuration",
    )
    backend = config["backend"]
    if backend not in {"sam2", "fake"}:
        raise ValueError("stage2.instance_model.backend must be 'sam2' or 'fake'")
    if backend == "sam2" and not config.get("model_config"):
        raise ValueError("stage2.instance_model.model_config is required for SAM 2")

    preprocessing = _require_mapping(config["preprocessing"], "stage2.preprocessing")
    _require_keys(
        preprocessing,
        {"rgb_bands", "normalization", "lower_percentile", "upper_percentile"},
        "stage2.preprocessing",
    )
    if preprocessing["normalization"] not in NORMALIZATIONS:
        raise ValueError(f"Unsupported normalization: {preprocessing['normalization']}")
    bands = preprocessing["rgb_bands"]
    if (
        not isinstance(bands, list)
        or len(bands) != 3
        or any(isinstance(band, bool) or not isinstance(band, int) or band < 1 for band in bands)
    ):
        raise ValueError("stage2.preprocessing.rgb_bands must contain three positive 1-based integers")
    lower = _number(preprocessing["lower_percentile"], "stage2.preprocessing.lower_percentile")
    upper = _number(preprocessing["upper_percentile"], "stage2.preprocessing.upper_percentile")
    if not 0 <= lower < upper <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    generator = _require_mapping(config["automatic_mask_generator"], "stage2.instance_model.automatic_mask_generator")
    common_parameters = {
        "points_per_side",
        "points_per_batch",
        "pred_iou_thresh",
        "stability_score_thresh",
        "stability_score_offset",
        "box_nms_thresh",
        "crop_n_layers",
        "crop_nms_thresh",
        "crop_overlap_ratio",
        "crop_n_points_downscale_factor",
        "min_mask_region_area",
        "output_mode",
    }
    sam2_parameters = {"mask_threshold", "use_m2m", "multimask_output"}
    supported = common_parameters | (sam2_parameters if backend in {"sam2", "fake"} else set())
    unknown = set(generator) - supported
    if unknown:
        raise ValueError(f"Unsupported automatic-mask parameters: {sorted(unknown)}")
    _require_keys(generator, common_parameters, "stage2.instance_model.automatic_mask_generator")
    if generator["output_mode"] not in {"binary_mask", "uncompressed_rle", "coco_rle"}:
        raise ValueError("Unsupported automatic_mask_generator.output_mode")

    filtering = _require_mapping(config["filtering"], "stage2.filtering")
    _positive_int(filtering["minimum_area"], "stage2.filtering.minimum_area", allow_zero=True)
    maximum_area_ratio = _number(filtering["maximum_area_ratio"], "stage2.filtering.maximum_area_ratio")
    if not 0 < maximum_area_ratio <= 1:
        raise ValueError("stage2.filtering.maximum_area_ratio must be in (0, 1]")
    if filtering.get("border_policy") not in {"keep", "reject"}:
        raise ValueError("stage2.filtering.border_policy must be 'keep' or 'reject'")

    deduplication = _require_mapping(config["deduplication"], "stage2.deduplication")
    if deduplication.get("duplicate_policy") not in DUPE_POLICIES:
        raise ValueError("Unsupported duplicate policy")
    if deduplication.get("nested_policy") not in NESTED_POLICIES:
        raise ValueError("Unsupported nested policy")

    coverage = _require_mapping(config["coverage"], "stage2.coverage")
    if coverage.get("background_policy") not in BACKGROUNDS:
        raise ValueError("Unsupported background policy")
    _validate_visualization(config["visualization"])
    _validate_output(config["output"])


def validate_semantic_config(config: dict[str, Any]) -> None:
    """Validate the semantic-model portion of the unified document."""
    config = _require_mapping(config, "stage2.semantic_model")
    _require_keys(config, {"backend", "device", "inference"}, "stage2.semantic_model")
    if config["backend"] not in {"openearthmap_fasterseg", "fake"}:
        raise ValueError("stage2.semantic_model.backend must be 'openearthmap_fasterseg' or 'fake'")
    if config["backend"] == "openearthmap_fasterseg":
        for key in ("repository", "architecture", "checkpoint"):
            if not config.get(key):
                raise ValueError(f"stage2.semantic_model.{key} is required for OpenEarthMap FasterSeg")
    inference = _require_mapping(config["inference"], "stage2.semantic_model.inference")
    _positive_int(inference.get("input_size"), "stage2.semantic_model.inference.input_size")


def validate_fusion_config(config: dict[str, Any]) -> None:
    """Validate fusion thresholds and require complete final-label coverage."""
    config = _require_mapping(config, "stage2.fusion")
    _require_keys(
        config,
        {
            "minimum_sam_component_area",
            "semantic_confidence_threshold",
            "minimum_component_area",
            "fill_low_confidence_as_unknown",
        },
        "stage2.fusion",
    )
    threshold = _number(config["semantic_confidence_threshold"], "stage2.fusion.semantic_confidence_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("stage2.fusion.semantic_confidence_threshold must be between 0 and 1")
    _positive_int(config["minimum_sam_component_area"], "stage2.fusion.minimum_sam_component_area", allow_zero=True)
    if config["fill_low_confidence_as_unknown"] is not True:
        raise ValueError("stage2.fusion.fill_low_confidence_as_unknown must be true for full coverage")
    minimums = _require_mapping(config["minimum_component_area"], "stage2.fusion.minimum_component_area")
    missing_minimums = set(CLASS_NAMES) - set(minimums)
    unknown_minimums = set(minimums) - set(FUSED_CLASS_NAMES)
    if missing_minimums or unknown_minimums:
        raise ValueError(
            "stage2.fusion.minimum_component_area must define every final label; "
            f"missing={sorted(missing_minimums)}, unknown={sorted(unknown_minimums)}"
        )
    for label, value in minimums.items():
        _positive_int(value, f"stage2.fusion.minimum_component_area.{label}", allow_zero=True)


def validate_stage3_config(config: dict[str, Any]) -> None:
    """Validate exact per-tile graph construction settings."""
    config = _require_mapping(config, "stage3")
    _require_keys(
        config,
        {"input", "adjacency", "geometry", "visualization", "output"},
        "stage3",
    )

    input_config = _require_mapping(config["input"], "stage3.input")
    _require_keys(input_config, {"source"}, "stage3.input")
    if input_config["source"] != "refinement":
        raise ValueError("stage3.input.source must be 'refinement'")

    adjacency = _require_mapping(config["adjacency"], "stage3.adjacency")
    _require_keys(
        adjacency,
        {"method", "connectivity", "minimum_shared_boundary_pixels"},
        "stage3.adjacency",
    )
    if adjacency["method"] not in STAGE3_ADJACENCY_METHODS:
        raise ValueError("stage3.adjacency.method must be 'shared_edge'")
    connectivity = _positive_int(adjacency["connectivity"], "stage3.adjacency.connectivity")
    if connectivity != 4:
        raise ValueError("stage3.adjacency.connectivity must be 4 for shared-edge adjacency")
    minimum_shared_boundary = _positive_int(
        adjacency["minimum_shared_boundary_pixels"],
        "stage3.adjacency.minimum_shared_boundary_pixels",
    )
    if minimum_shared_boundary != 1:
        raise ValueError(
            "stage3.adjacency.minimum_shared_boundary_pixels must be 1; "
            "Stage 3 preserves every shared-edge adjacency"
        )

    geometry = _require_mapping(config["geometry"], "stage3.geometry")
    _require_keys(geometry, {"simplification_tolerance"}, "stage3.geometry")
    tolerance = _number(
        geometry["simplification_tolerance"],
        "stage3.geometry.simplification_tolerance",
    )
    if tolerance != 0:
        raise ValueError(
            "stage3.geometry.simplification_tolerance must be 0.0; "
            "Stage 3 preserves exact pixel-edge polygons"
        )

    visualization = _require_mapping(config["visualization"], "stage3.visualization")
    _require_keys(
        visualization,
        {
            "random_seed",
            "show_ids",
            "node_size_min",
            "node_size_max",
            "edge_width_min",
            "edge_width_max",
            "edge_alpha",
        },
        "stage3.visualization",
    )
    _nonnegative_int(visualization["random_seed"], "stage3.visualization.random_seed")
    if not isinstance(visualization["show_ids"], bool):
        raise ValueError("stage3.visualization.show_ids must be a boolean")
    node_size_min = _positive_number(
        visualization["node_size_min"], "stage3.visualization.node_size_min"
    )
    node_size_max = _positive_number(
        visualization["node_size_max"], "stage3.visualization.node_size_max"
    )
    if node_size_min > node_size_max:
        raise ValueError("stage3.visualization.node_size_min must not exceed node_size_max")
    edge_width_min = _positive_number(
        visualization["edge_width_min"], "stage3.visualization.edge_width_min"
    )
    edge_width_max = _positive_number(
        visualization["edge_width_max"], "stage3.visualization.edge_width_max"
    )
    if edge_width_min > edge_width_max:
        raise ValueError("stage3.visualization.edge_width_min must not exceed edge_width_max")
    edge_alpha = _number(visualization["edge_alpha"], "stage3.visualization.edge_alpha")
    if not 0 <= edge_alpha <= 1:
        raise ValueError("stage3.visualization.edge_alpha must be between 0 and 1")
    _nonnegative_int(
        visualization.get("minimum_display_node_area_pixels", 0),
        "stage3.visualization.minimum_display_node_area_pixels",
    )
    _nonnegative_int(
        visualization.get("minimum_display_unknown_area_pixels", 0),
        "stage3.visualization.minimum_display_unknown_area_pixels",
    )
    _nonnegative_int(
        visualization.get("minimum_display_topology_fragment_area_pixels", 0),
        "stage3.visualization.minimum_display_topology_fragment_area_pixels",
    )
    _nonnegative_int(
        visualization.get("minimum_display_shared_boundary_pixels", 0),
        "stage3.visualization.minimum_display_shared_boundary_pixels",
    )
    for name in ("route_edges_through_contacts", "show_contact_anchors"):
        if not isinstance(visualization.get(name, True), bool):
            raise ValueError(f"stage3.visualization.{name} must be a boolean")

    _validate_stage3_output(config["output"])


def validate_refinement_config(config: dict[str, Any]) -> None:
    """Validate the reversible Stage 2R refinement configuration."""

    # Kept as a lazy import because :mod:`src.config` validates the root YAML
    # while it is itself being imported by the processing stages.
    from src.refinement.config import select_refinement_config as select

    select(config)


def validate_stage4_config(config: dict[str, Any]) -> None:
    """Validate multi-tile graph stitching and overlap reconciliation."""

    config = _require_mapping(config, "stage4")
    _require_keys(config, {"input", "overlap", "ownership", "adjacency", "output"}, "stage4")
    input_config = _require_mapping(config["input"], "stage4.input")
    _require_keys(input_config, {"root_directory", "phase_directory"}, "stage4.input")
    for name in ("root_directory", "phase_directory"):
        if not isinstance(input_config[name], str) or not input_config[name].strip():
            raise ValueError(f"stage4.input.{name} must be a non-empty path")
    overlap = _require_mapping(config["overlap"], "stage4.overlap")
    _require_keys(
        overlap,
        {
            "method",
            "minimum_intersection_pixels",
            "minimum_intersection_over_smaller",
            "minimum_intersection_over_union",
            "minimum_bidirectional_coverage",
            "unknown_minimum_intersection_over_smaller",
            "require_same_class",
            "require_mutual_best",
        },
        "stage4.overlap",
    )
    if overlap["method"] != "mutual_best_mask_overlap":
        raise ValueError("stage4.overlap.method must be 'mutual_best_mask_overlap'")
    _positive_int(overlap["minimum_intersection_pixels"], "stage4.overlap.minimum_intersection_pixels")
    for name in (
        "minimum_intersection_over_smaller",
        "minimum_intersection_over_union",
        "minimum_bidirectional_coverage",
        "unknown_minimum_intersection_over_smaller",
    ):
        value = _number(overlap[name], f"stage4.overlap.{name}")
        if not 0 <= value <= 1:
            raise ValueError(f"stage4.overlap.{name} must be between zero and one")
    for name in ("require_same_class", "require_mutual_best"):
        if not isinstance(overlap[name], bool):
            raise ValueError(f"stage4.overlap.{name} must be a boolean")
    ownership = _require_mapping(config["ownership"], "stage4.ownership")
    _require_keys(ownership, {"method"}, "stage4.ownership")
    if ownership["method"] != "maximum_distance_to_tile_edge_then_confidence":
        raise ValueError(
            "stage4.ownership.method must be "
            "'maximum_distance_to_tile_edge_then_confidence'"
        )
    adjacency = _require_mapping(config["adjacency"], "stage4.adjacency")
    _require_keys(adjacency, {"minimum_shared_boundary_pixels"}, "stage4.adjacency")
    if _positive_int(
        adjacency["minimum_shared_boundary_pixels"],
        "stage4.adjacency.minimum_shared_boundary_pixels",
    ) != 1:
        raise ValueError("Stage 4 must preserve every shared pixel-side adjacency")
    output = _require_mapping(config["output"], "stage4.output")
    _require_keys(output, {"root_directory"}, "stage4.output")
    if not isinstance(output["root_directory"], str) or not output["root_directory"].strip():
        raise ValueError("stage4.output.root_directory must be a non-empty path")
    analysis = config.get("analysis", {})
    for key, default in (("minimum_unknown_area_pixels", 41), ("minimum_shared_boundary_pixels", 2)):
        _positive_int(analysis.get(key, default), f"stage4.analysis.{key}")


def select_project_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only project paths and tiling settings."""
    project = copy.deepcopy(_require_mapping(config.get("project"), "project"))
    validate_project_config(project)
    return project


def select_stage2_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the complete unified Stage 2 mapping."""
    stage2 = copy.deepcopy(_require_mapping(config.get("stage2"), "stage2"))
    candidate = {"project": select_project_config(config), "stage2": stage2}
    validate_config(candidate)
    return stage2


def select_stage3_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return Stage 3 settings with the Stage 2 artifact root derived once."""
    stage3 = copy.deepcopy(_require_mapping(config.get("stage3"), "stage3"))
    validate_stage3_config(stage3)
    stage2 = select_stage2_config(config)
    stage3["stage2_root"] = stage2["output"]["root_directory"]
    if config.get("_config_path"):
        stage3["_config_path"] = config["_config_path"]
    return stage3


def select_stage4_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return validated Stage 4 settings."""

    stage4 = copy.deepcopy(_require_mapping(config.get("stage4"), "stage4"))
    validate_stage4_config(stage4)
    if config.get("_config_path"):
        stage4["_config_path"] = config["_config_path"]
    return stage4


def select_refinement_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return validated Stage 2R settings with the Stage 2 root derived once."""

    from src.refinement.config import select_refinement_config as select

    return select(config)


def select_instance_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten unified settings for the existing SAM/SAM 2 phase API."""
    stage2 = select_stage2_config(config)
    result = _instance_view(stage2)
    result["_project_paths"] = copy.deepcopy(select_project_config(config)["paths"])
    if config.get("_config_path"):
        result["_config_path"] = config["_config_path"]
    validate_instance_config(result)
    return result


def select_landscape_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten unified settings for the existing semantic/fusion phase API."""
    stage2 = select_stage2_config(config)
    result = copy.deepcopy(stage2["semantic_model"])
    result["stage2_root"] = stage2["output"]["root_directory"]
    fusion = copy.deepcopy(stage2["fusion"])
    # Compatibility values for the retired non-tree phase API. They are
    # derived here and are not a second user-editable configuration source.
    result["completion"] = {
        **fusion,
        "tree_confidence_threshold": fusion["semantic_confidence_threshold"],
        "tree_dilation_pixels": 0,
        "sam_tree_overlap_threshold": 0.60,
        "priority": [name for name in CLASS_NAMES if name != "tree"],
        "minimum_component_area": {**fusion["minimum_component_area"], "unknown": 1},
        "morphology_pixels": {},
    }
    result["visualization"] = copy.deepcopy(stage2["visualization"])
    result["output"] = copy.deepcopy(stage2["output"])
    if config.get("_config_path"):
        result["_config_path"] = config["_config_path"]
    validate_landscape_config(result)
    return result


def instance_config(project_config: dict[str, Any]) -> dict[str, Any]:
    """Public selector for the instance-model pipeline."""
    return select_instance_config(project_config)


def semantic_config(project_config: dict[str, Any]) -> dict[str, Any]:
    """Public selector for the semantic-model and fusion pipeline."""
    return select_landscape_config(project_config)


def stage3_config(project_config: dict[str, Any]) -> dict[str, Any]:
    """Public selector for the per-tile graph-building pipeline."""
    return select_stage3_config(project_config)


def stage4_config(project_config: dict[str, Any]) -> dict[str, Any]:
    """Public selector for multi-tile graph-combination settings."""

    return select_stage4_config(project_config)


def refinement_config(project_config: dict[str, Any]) -> dict[str, Any]:
    """Public selector for reversible Stage 2R refinement settings."""

    return select_refinement_config(project_config)


def load_stage2_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the unified YAML and select its legacy instance-phase view."""
    return select_instance_config(load_config(path))


def load_landscape_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the unified YAML and select its legacy semantic/fusion view."""
    return select_landscape_config(load_config(path))


def load_stage3_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the unified YAML and select Stage 3 graph settings."""
    return select_stage3_config(load_config(path))


def load_stage4_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the unified YAML and select Stage 4 graph-combination settings."""

    return select_stage4_config(load_config(path))


def load_refinement_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the unified YAML and select Stage 2R refinement settings."""

    return select_refinement_config(load_config(path))


def validate_landscape_config(config: dict[str, Any]) -> None:
    """Validate the flattened configuration consumed by legacy landscape phases."""
    config = _require_mapping(config, "legacy landscape configuration")
    _require_keys(
        config,
        {"backend", "stage2_root", "device", "inference", "completion", "visualization", "output"},
        "legacy landscape configuration",
    )
    semantic = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key in {"backend", "repository", "architecture", "checkpoint", "device", "inference"}
    }
    validate_semantic_config(semantic)
    completion = _require_mapping(config["completion"], "legacy landscape completion")
    for key in ("tree_confidence_threshold", "semantic_confidence_threshold", "sam_tree_overlap_threshold"):
        value = _number(completion[key], f"legacy landscape completion.{key}")
        if not 0 <= value <= 1:
            raise ValueError(f"legacy landscape completion.{key} must be between 0 and 1")
    unknown = set(completion["priority"]) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown OpenEarthMap classes in completion.priority: {sorted(unknown)}")
    if "tree" in completion["priority"]:
        raise ValueError("Trees must not appear in legacy completion.priority")
    _validate_visualization(config["visualization"])
    _validate_output(config["output"])


def relevant_stage2_config(config: dict[str, Any], phase: str) -> dict[str, Any]:
    sections = {
        "01_validation": [],
        "02_preprocessing": ["preprocessing"],
        "03_model": ["backend", "checkpoint", "model_type", "model_config", "device", "mixed_precision"],
        "04_raw_inference": [
            "backend",
            "checkpoint",
            "model_type",
            "model_config",
            "device",
            "mixed_precision",
            "automatic_mask_generator",
        ],
        "05_normalized": [],
        "06_filtering": ["filtering"],
        "07_deduplication": ["deduplication"],
        "08_coverage": ["coverage"],
        "09_visualization": ["visualization"],
        "10_final": [
            "backend",
            "checkpoint",
            "model_type",
            "model_config",
            "preprocessing",
            "automatic_mask_generator",
            "filtering",
            "deduplication",
            "coverage",
        ],
    }
    if phase not in sections:
        raise KeyError(f"Unknown instance phase: {phase}")
    return {key: copy.deepcopy(config.get(key)) for key in sections[phase]}


def relevant_landscape_config(config: dict[str, Any], phase: str) -> dict[str, Any]:
    sections = {
        "01_model": ["backend", "repository", "architecture", "checkpoint", "device"],
        "02_landcover": ["backend", "repository", "architecture", "checkpoint", "device", "inference"],
        "03_completion": ["completion"],
        "04_visualization": ["visualization"],
        "05_final": ["backend", "architecture", "checkpoint", "inference", "completion"],
    }
    if phase not in sections:
        raise KeyError(f"Unknown semantic/fusion phase: {phase}")
    return {key: copy.deepcopy(config.get(key)) for key in sections[phase]}


def select_stage2_sections(config: dict[str, Any], sections: Iterable[str]) -> dict[str, Any]:
    """Select unified Stage 2 sections for manifests and cache hashes."""
    stage2 = config["stage2"] if "stage2" in config else config
    requested = list(sections)
    unknown = set(requested) - set(stage2)
    if unknown:
        raise KeyError(f"Unknown Stage 2 configuration sections: {sorted(unknown)}")
    return {name: copy.deepcopy(stage2[name]) for name in requested}


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-compatible configuration."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_hash(config: dict[str, Any], sections: Iterable[str] | None = None) -> str:
    """Hash the complete unified config or a selected set of Stage 2 sections."""
    value = select_stage2_sections(config, sections) if sections is not None else _without_internal_keys(config)
    return stable_hash(value)


def stage2_config_hash(config: dict[str, Any], phase: str) -> str:
    return stable_hash(relevant_stage2_config(config, phase))


def landscape_config_hash(config: dict[str, Any], phase: str) -> str:
    return stable_hash(relevant_landscape_config(config, phase))


def stage3_config_hash(config: dict[str, Any]) -> str:
    """Hash graph behavior plus the derived Stage 2 artifact location."""
    return stable_hash(select_stage3_config(config))


def stage4_config_hash(config: dict[str, Any]) -> str:
    """Hash multi-tile graph-combination behavior and input location."""

    return stable_hash(select_stage4_config(config))


def refinement_config_hash(config: dict[str, Any]) -> str:
    """Hash refinement behavior plus its derived Stage 2 artifact location."""

    from src.refinement.config import config_hash as refinement_hash

    return refinement_hash(select_refinement_config(config))


def _instance_view(stage2: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_require_mapping(stage2["instance_model"], "stage2.instance_model"))
    for name in ("preprocessing", "filtering", "deduplication", "coverage", "visualization", "output"):
        result[name] = copy.deepcopy(stage2[name])
    return result


def _validate_visualization(config: dict[str, Any]) -> None:
    config = _require_mapping(config, "stage2.visualization")
    _require_keys(config, {"alpha", "random_seed"}, "stage2.visualization")
    alpha = _number(config["alpha"], "stage2.visualization.alpha")
    if not 0 <= alpha <= 1:
        raise ValueError("stage2.visualization.alpha must be between 0 and 1")
    _nonnegative_int(config["random_seed"], "stage2.visualization.random_seed")


def _validate_output(config: dict[str, Any]) -> None:
    config = _require_mapping(config, "stage2.output")
    _require_keys(config, {"root_directory"}, "stage2.output")
    if not isinstance(config["root_directory"], str) or not config["root_directory"].strip():
        raise ValueError("stage2.output.root_directory must be a non-empty path string")


def _validate_stage3_output(config: dict[str, Any]) -> None:
    config = _require_mapping(config, "stage3.output")
    _require_keys(config, {"root_directory"}, "stage3.output")
    if not isinstance(config["root_directory"], str) or not config["root_directory"].strip():
        raise ValueError("stage3.output.root_directory must be a non-empty path string")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_keys(config: dict[str, Any], required: set[str], name: str) -> None:
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing {name} settings: {sorted(missing)}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _positive_number(value: Any, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    return _positive_int(value, name, allow_zero=True)


def _without_internal_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_internal_keys(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_without_internal_keys(item) for item in value]
    return value


# Stage 1 compatibility constants are derived from the unified YAML. Existing
# imports such as ``from src.config import TILE_SIZE`` therefore keep working.
_LOADED_CONFIG = load_config()
_PROJECT_CONFIG = _LOADED_CONFIG["project"]
_PATH_CONFIG = _PROJECT_CONFIG["paths"]


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


DATA_DIR = _project_path(_PATH_CONFIG["data_directory"])
RAW_DIR = _project_path(_PATH_CONFIG["raw_directory"])
TILES_DIR = _project_path(_PATH_CONFIG["tiles_directory"])
METADATA_DIR = _project_path(_PATH_CONFIG["metadata_directory"])
TILE_SIZE = int(_PROJECT_CONFIG["tiling"]["tile_size"])
OVERLAP = int(_PROJECT_CONFIG["tiling"]["overlap"])
STRIDE = TILE_SIZE - OVERLAP
