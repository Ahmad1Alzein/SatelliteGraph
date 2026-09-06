"""Configuration defaults and validation for segmentation refinement.

The refinement phase deliberately defaults to human review.  Automatic mode
uses the same immutable proposals and replay validators, but supplies
conservative, versioned decisions and abstains when its evidence is incomplete.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.labels import FUSED_CLASS_NAMES

from .automatic import default_automatic_config, validate_automatic_config


CONFIG_VERSION = "1.0"

DEFAULT_ONTOLOGY: dict[str, dict[str, Any]] = {
    "bareland": {
        "entity_type": "continuous_homogeneous_region",
        "policy": "connected_homogeneous_region",
        "allow_merge": True,
        "allow_split": True,
    },
    "rangeland": {
        "entity_type": "continuous_homogeneous_region",
        "policy": "connected_homogeneous_region",
        "allow_merge": True,
        "allow_split": True,
    },
    "developed space": {
        "entity_type": "continuous_homogeneous_region",
        "policy": "connected_homogeneous_region",
        "allow_merge": True,
        "allow_split": True,
    },
    "road": {
        "entity_type": "network_reach",
        "policy": "junction_to_junction",
        "allow_merge": True,
        "allow_split": True,
    },
    "tree": {
        "entity_type": "continuous_canopy",
        "policy": "split_only_across_non_tree_gap",
        "allow_merge": True,
        "allow_split": True,
    },
    "water": {
        "entity_type": "continuous_homogeneous_region",
        "policy": "connected_homogeneous_region",
        "allow_merge": True,
        "allow_split": True,
    },
    "agriculture land": {
        "entity_type": "field",
        "policy": "preserve_individual_field_boundaries",
        "allow_merge": True,
        "allow_split": True,
    },
    "buildings": {
        "entity_type": "physical_building",
        "policy": "one_physical_building",
        "allow_merge": True,
        "allow_split": True,
    },
    "unknown": {
        "entity_type": "unresolved_region",
        "policy": "retain_until_supported",
        "allow_merge": False,
        "allow_split": False,
    },
}


DEFAULT_REFINEMENT_CONFIG: dict[str, Any] = {
    "configuration_version": CONFIG_VERSION,
    "mode": "human_review",
    "connectivity": 4,
    "ontology": DEFAULT_ONTOLOGY,
    "operations": {
        "connectivity_split": {
            "enabled": True,
            "auto_apply": True,
            # Components stay separate internally for evidence/replay, but a
            # secondary component must reach this area to become an object node.
            "minimum_component_area_pixels": 41,
        },
        "semantic_split": {
            "enabled": True,
            "auto_apply": False,
            "minimum_segment_area_pixels": 64,
            "minimum_child_area_pixels": 16,
            "score_threshold": 0.98,
            "stability_trials": 3,
        },
        "absorb": {
            "enabled": True,
            "auto_apply": False,
            "maximum_segment_area_pixels": 40,
            "score_threshold": 0.98,
            "score_margin_threshold": 0.25,
            "minimum_shared_boundary_fraction": 0.80,
        },
        "unknown_cleanup": {
            "enabled": True,
            "auto_apply": True,
            "maximum_segment_area_pixels": 40,
            "allowed_target_classes": [
                "bareland",
                "rangeland",
                "developed space",
                "tree",
                "water",
            ],
            "minimum_mean_target_probability": 0.60,
            "minimum_probability_margin": 0.20,
            "minimum_dominant_pixel_fraction": 0.90,
            "minimum_shared_boundary_fraction": 0.50,
            "minimum_weak_color_boundary_score": 0.70,
            "minimum_region_mean_color_similarity": 0.55,
            "require_not_tile_boundary": True,
        },
        "contextual_cleanup": {
            "enabled": True,
            "auto_apply": True,
            "maximum_segment_area_pixels": 40,
            "semantic_consensus_minimum_probability": 0.55,
            "semantic_consensus_minimum_margin": 0.15,
            "semantic_consensus_minimum_purity": 0.80,
            "maximum_selected_energy": 0.40,
            "minimum_energy_margin": 0.02,
            "standalone_tree_minimum_area_pixels": 20,
            "standalone_tree_minimum_probability": 0.55,
            "standalone_tree_minimum_margin": 0.15,
            "standalone_tree_minimum_purity": 0.80,
            "standalone_building_minimum_area_pixels": 20,
            "standalone_building_minimum_probability": 0.65,
            "standalone_building_minimum_margin": 0.20,
            "standalone_building_minimum_purity": 0.90,
            "agriculture_minimum_boundary_fraction": 0.75,
            "building_minimum_probability": 0.35,
            "road_minimum_probability": 0.30,
        },
        "merge": {
            "enabled": True,
            "auto_apply": False,
            "score_threshold": 0.98,
            "minimum_shared_boundary_pixels": 8,
            "minimum_shared_boundary_fraction": 0.15,
        },
    },
    "evidence": {
        "semantic_weight": 0.35,
        "boundary_weight": 0.30,
        "appearance_weight": 0.25,
        "shape_weight": 0.10,
    },
    "review": {
        "crop_padding_pixels": 32,
        "maximum_candidates_per_tile": 5000,
    },
    "automatic": default_automatic_config(),
    "visualization": {
        "random_seed": 42,
        "alpha": 0.45,
    },
    "output": {
        # Resolved by ``select_refinement_config`` from Stage 2 unless the
        # refinement section explicitly uses a different root.
        "root_directory": None,
        "phase_directory": "13_refinement",
    },
}


def default_refinement_config() -> dict[str, Any]:
    """Return a mutable, independent copy of the conservative defaults."""

    return copy.deepcopy(DEFAULT_REFINEMENT_CONFIG)


def select_refinement_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Select and validate refinement settings from a project configuration.

    ``config`` may be the full project mapping, a selected refinement mapping,
    or ``None``.  A full project mapping may use either ``refinement`` or the
    compatibility name ``stage2_refinement`` (but not both).  The Stage 2
    output root is inherited so that the result lands at
    ``<stage2_root>/<tile_id>/13_refinement`` by default.
    """

    source: Mapping[str, Any] = {} if config is None else _mapping(config, "config")
    if "refinement" in source and "stage2_refinement" in source:
        raise ValueError("Use only one of 'refinement' or 'stage2_refinement'")

    is_project = any(key in source for key in ("stage1", "stage2", "stage3", "refinement", "stage2_refinement"))
    if is_project:
        override_value = source.get("refinement", source.get("stage2_refinement", {}))
        override = _mapping(override_value, "refinement")
        stage2_root = _project_stage2_root(source)
    else:
        override = source
        selected_root = override.get("stage2_root")
        stage2_root = str(selected_root) if selected_root is not None else None
        # ``stage2_root`` is derived state, not a user-overridable default key.
        override = {key: value for key, value in override.items() if key not in {"stage2_root", "_config_path"}}

    selected = default_refinement_config()
    _deep_merge(selected, override, "refinement")
    explicit_output_root = selected["output"].get("root_directory")
    resolved_root = explicit_output_root or stage2_root or "artifacts/stage2"
    selected["output"]["root_directory"] = str(resolved_root)
    selected["stage2_root"] = str(stage2_root or resolved_root)
    if source.get("_config_path") is not None:
        selected["_config_path"] = str(source["_config_path"])
    validate_refinement_config(selected)
    return selected


def validate_refinement_config(config: Mapping[str, Any]) -> None:
    """Validate a selected refinement configuration or raise ``ValueError``."""

    cfg = _mapping(config, "refinement")
    required = {
        "configuration_version",
        "mode",
        "connectivity",
        "ontology",
        "operations",
        "evidence",
        "review",
        "automatic",
        "visualization",
        "output",
        "stage2_root",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"refinement is missing required keys: {', '.join(missing)}")
    unexpected = sorted(set(cfg) - required - {"_config_path"})
    if unexpected:
        raise ValueError(f"refinement has unexpected keys: {', '.join(unexpected)}")
    if cfg["configuration_version"] != CONFIG_VERSION:
        raise ValueError(f"refinement.configuration_version must be {CONFIG_VERSION!r}")
    normalize_refinement_mode(cfg["mode"])
    if _integer(cfg["connectivity"], "refinement.connectivity", minimum=1) != 4:
        raise ValueError("refinement.connectivity must be 4")

    ontology = _mapping(cfg["ontology"], "refinement.ontology")
    expected_classes = set(FUSED_CLASS_NAMES)
    actual_classes = set(ontology)
    if actual_classes != expected_classes:
        missing_classes = sorted(expected_classes - actual_classes)
        extra_classes = sorted(actual_classes - expected_classes)
        detail = []
        if missing_classes:
            detail.append(f"missing {missing_classes}")
        if extra_classes:
            detail.append(f"unexpected {extra_classes}")
        raise ValueError("refinement.ontology must use exact Stage 2 class names: " + "; ".join(detail))
    for class_name in FUSED_CLASS_NAMES:
        rule = _mapping(ontology[class_name], f"refinement.ontology.{class_name}")
        _exact_keys(rule, {"entity_type", "policy", "allow_merge", "allow_split"}, f"refinement.ontology.{class_name}")
        _nonempty_string(rule["entity_type"], f"refinement.ontology.{class_name}.entity_type")
        _nonempty_string(rule["policy"], f"refinement.ontology.{class_name}.policy")
        _boolean(rule["allow_merge"], f"refinement.ontology.{class_name}.allow_merge")
        _boolean(rule["allow_split"], f"refinement.ontology.{class_name}.allow_split")

    operations = _mapping(cfg["operations"], "refinement.operations")
    _exact_keys(
        operations,
        {
            "connectivity_split",
            "semantic_split",
            "absorb",
            "unknown_cleanup",
            "contextual_cleanup",
            "merge",
        },
        "refinement.operations",
    )
    _validate_connectivity_split(operations["connectivity_split"])
    _validate_semantic_split(operations["semantic_split"])
    _validate_absorb(operations["absorb"])
    _validate_unknown_cleanup(operations["unknown_cleanup"])
    _validate_contextual_cleanup(operations["contextual_cleanup"])
    _validate_merge(operations["merge"])

    evidence = _mapping(cfg["evidence"], "refinement.evidence")
    weight_names = {"semantic_weight", "boundary_weight", "appearance_weight", "shape_weight"}
    _exact_keys(evidence, weight_names, "refinement.evidence")
    total = sum(_fraction(evidence[name], f"refinement.evidence.{name}") for name in weight_names)
    if abs(total - 1.0) > 1e-9:
        raise ValueError("refinement evidence weights must sum to 1.0")

    review = _mapping(cfg["review"], "refinement.review")
    _exact_keys(review, {"crop_padding_pixels", "maximum_candidates_per_tile"}, "refinement.review")
    _integer(review["crop_padding_pixels"], "refinement.review.crop_padding_pixels", minimum=0)
    _integer(review["maximum_candidates_per_tile"], "refinement.review.maximum_candidates_per_tile", minimum=1)

    validate_automatic_config(
        _mapping(cfg["automatic"], "refinement.automatic")
    )
    protected_classes = set(cfg["automatic"]["protected_classes"])
    unknown_protected = sorted(protected_classes - set(FUSED_CLASS_NAMES))
    if unknown_protected:
        raise ValueError(
            "refinement.automatic.protected_classes contains unknown Stage 2 "
            f"classes: {unknown_protected}"
        )

    visualization = _mapping(cfg["visualization"], "refinement.visualization")
    _exact_keys(visualization, {"random_seed", "alpha"}, "refinement.visualization")
    _integer(visualization["random_seed"], "refinement.visualization.random_seed", minimum=0)
    _fraction(visualization["alpha"], "refinement.visualization.alpha")

    output = _mapping(cfg["output"], "refinement.output")
    _exact_keys(output, {"root_directory", "phase_directory"}, "refinement.output")
    _nonempty_string(output["root_directory"], "refinement.output.root_directory")
    phase = _nonempty_string(output["phase_directory"], "refinement.output.phase_directory")
    if phase != "13_refinement":
        raise ValueError("refinement.output.phase_directory must be '13_refinement'")
    _nonempty_string(cfg["stage2_root"], "refinement.stage2_root")


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable hash of the selected, public refinement settings."""

    public = {key: value for key, value in config.items() if not str(key).startswith("_")}
    validate_refinement_config(public)
    encoded = json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_refinement_mode(value: Any) -> str:
    """Return the stable workflow name, accepting the two legacy aliases.

    ``proposal_only`` and ``apply_approved`` were configuration values in v1
    but were never consumed by the pipeline.  Treating both as human review
    preserves existing programmatic configurations without silently enabling
    automatic semantic decisions.
    """

    mode = _nonempty_string(value, "refinement.mode")
    aliases = {
        "human": "human_review",
        "review": "human_review",
        "human_review": "human_review",
        "proposal_only": "human_review",
        "apply_approved": "human_review",
        "auto": "automatic",
        "automatic": "automatic",
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError(
            "refinement.mode must be 'human_review' or 'automatic'"
        ) from exc


def ontology_for_class(config: Mapping[str, Any], class_name: str) -> dict[str, Any]:
    """Return an independent ontology rule for an exact Stage 2 class name."""

    validate_refinement_config(config)
    if class_name not in config["ontology"]:
        raise KeyError(f"Unknown Stage 2 class: {class_name!r}")
    return copy.deepcopy(config["ontology"][class_name])


def refinement_directory(config: Mapping[str, Any], tile_id: str) -> Path:
    """Resolve the configured output directory for one safe tile identifier."""

    validate_refinement_config(config)
    tile = _nonempty_string(tile_id, "tile_id")
    if Path(tile).name != tile or tile in {".", ".."}:
        raise ValueError("tile_id must be one path-safe name")
    return Path(config["output"]["root_directory"]).expanduser() / tile / "13_refinement"


def _project_stage2_root(config: Mapping[str, Any]) -> str | None:
    stage2 = config.get("stage2")
    if stage2 is None:
        return None
    stage2 = _mapping(stage2, "stage2")
    output = stage2.get("output")
    if output is None:
        return None
    output = _mapping(output, "stage2.output")
    value = output.get("root_directory")
    return None if value is None else _nonempty_string(value, "stage2.output.root_directory")


def _validate_connectivity_split(value: Any) -> None:
    path = "refinement.operations.connectivity_split"
    config = _mapping(value, path)
    _exact_keys(config, {"enabled", "auto_apply", "minimum_component_area_pixels"}, path)
    _operation_flags(config, path)
    if config["enabled"] is not True or config["auto_apply"] is not True:
        raise ValueError(
            f"{path} must remain enabled and auto_apply=true so component-level "
            "unknown cleanup and proposal replay remain exact"
        )
    _integer(config["minimum_component_area_pixels"], f"{path}.minimum_component_area_pixels", minimum=1)


def _validate_semantic_split(value: Any) -> None:
    path = "refinement.operations.semantic_split"
    config = _mapping(value, path)
    _exact_keys(
        config,
        {
            "enabled",
            "auto_apply",
            "minimum_segment_area_pixels",
            "minimum_child_area_pixels",
            "score_threshold",
            "stability_trials",
        },
        path,
    )
    _operation_flags(config, path)
    if config["auto_apply"] is not False:
        raise ValueError(
            f"{path}.auto_apply must remain false; semantic splits require human review"
        )
    parent_area = _integer(config["minimum_segment_area_pixels"], f"{path}.minimum_segment_area_pixels", minimum=2)
    child_area = _integer(config["minimum_child_area_pixels"], f"{path}.minimum_child_area_pixels", minimum=1)
    if child_area * 2 > parent_area:
        raise ValueError(f"{path}.minimum_child_area_pixels cannot exceed half the parent minimum")
    _fraction(config["score_threshold"], f"{path}.score_threshold")
    _integer(config["stability_trials"], f"{path}.stability_trials", minimum=2)


def _validate_absorb(value: Any) -> None:
    path = "refinement.operations.absorb"
    config = _mapping(value, path)
    _exact_keys(
        config,
        {
            "enabled",
            "auto_apply",
            "maximum_segment_area_pixels",
            "score_threshold",
            "score_margin_threshold",
            "minimum_shared_boundary_fraction",
        },
        path,
    )
    _operation_flags(config, path)
    if config["auto_apply"] is not False:
        raise ValueError(
            f"{path}.auto_apply must remain false; use refinement.mode=automatic "
            "and refinement.automatic.absorb"
        )
    _integer(config["maximum_segment_area_pixels"], f"{path}.maximum_segment_area_pixels", minimum=1)
    for name in ("score_threshold", "score_margin_threshold", "minimum_shared_boundary_fraction"):
        _fraction(config[name], f"{path}.{name}")


def _validate_unknown_cleanup(value: Any) -> None:
    path = "refinement.operations.unknown_cleanup"
    config = _mapping(value, path)
    _exact_keys(
        config,
        {
            "enabled",
            "auto_apply",
            "maximum_segment_area_pixels",
            "allowed_target_classes",
            "minimum_mean_target_probability",
            "minimum_probability_margin",
            "minimum_dominant_pixel_fraction",
            "minimum_shared_boundary_fraction",
            "minimum_weak_color_boundary_score",
            "minimum_region_mean_color_similarity",
            "require_not_tile_boundary",
        },
        path,
    )
    _operation_flags(config, path)
    _integer(
        config["maximum_segment_area_pixels"],
        f"{path}.maximum_segment_area_pixels",
        minimum=1,
    )
    classes = config["allowed_target_classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{path}.allowed_target_classes must be a non-empty list")
    if any(not isinstance(name, str) or name not in FUSED_CLASS_NAMES for name in classes):
        raise ValueError(
            f"{path}.allowed_target_classes must contain exact Stage 2 class names"
        )
    if len(set(classes)) != len(classes) or "unknown" in classes:
        raise ValueError(
            f"{path}.allowed_target_classes must be unique and cannot include 'unknown'"
        )
    for protected in ("road", "agriculture land", "buildings"):
        if protected in classes:
            raise ValueError(
                f"{path}.allowed_target_classes cannot include protected class {protected!r}"
            )
    for name in (
        "minimum_mean_target_probability",
        "minimum_probability_margin",
        "minimum_dominant_pixel_fraction",
        "minimum_shared_boundary_fraction",
        "minimum_weak_color_boundary_score",
        "minimum_region_mean_color_similarity",
    ):
        _fraction(config[name], f"{path}.{name}")
    _boolean(config["require_not_tile_boundary"], f"{path}.require_not_tile_boundary")


def _validate_contextual_cleanup(value: Any) -> None:
    path = "refinement.operations.contextual_cleanup"
    config = _mapping(value, path)
    fraction_keys = {
        "semantic_consensus_minimum_probability",
        "semantic_consensus_minimum_margin",
        "semantic_consensus_minimum_purity",
        "maximum_selected_energy",
        "minimum_energy_margin",
        "standalone_tree_minimum_probability",
        "standalone_tree_minimum_margin",
        "standalone_tree_minimum_purity",
        "standalone_building_minimum_probability",
        "standalone_building_minimum_margin",
        "standalone_building_minimum_purity",
        "agriculture_minimum_boundary_fraction",
        "building_minimum_probability",
        "road_minimum_probability",
    }
    _exact_keys(
        config,
        {
            "enabled",
            "auto_apply",
            "maximum_segment_area_pixels",
            "standalone_tree_minimum_area_pixels",
            "standalone_building_minimum_area_pixels",
            *fraction_keys,
        },
        path,
    )
    _operation_flags(config, path)
    maximum = _integer(
        config["maximum_segment_area_pixels"],
        f"{path}.maximum_segment_area_pixels",
        minimum=1,
    )
    if maximum > 40:
        raise ValueError(f"{path}.maximum_segment_area_pixels cannot exceed 40")
    for name in (
        "standalone_tree_minimum_area_pixels",
        "standalone_building_minimum_area_pixels",
    ):
        _integer(config[name], f"{path}.{name}", minimum=1)
    for name in fraction_keys:
        _fraction(config[name], f"{path}.{name}")


def _validate_merge(value: Any) -> None:
    path = "refinement.operations.merge"
    config = _mapping(value, path)
    _exact_keys(
        config,
        {
            "enabled",
            "auto_apply",
            "score_threshold",
            "minimum_shared_boundary_pixels",
            "minimum_shared_boundary_fraction",
        },
        path,
    )
    _operation_flags(config, path)
    if config["auto_apply"] is not False:
        raise ValueError(
            f"{path}.auto_apply must remain false; use refinement.mode=automatic "
            "and refinement.automatic.merge"
        )
    _fraction(config["score_threshold"], f"{path}.score_threshold")
    _integer(config["minimum_shared_boundary_pixels"], f"{path}.minimum_shared_boundary_pixels", minimum=1)
    _fraction(config["minimum_shared_boundary_fraction"], f"{path}.minimum_shared_boundary_fraction")


def _operation_flags(config: Mapping[str, Any], path: str) -> None:
    enabled = _boolean(config["enabled"], f"{path}.enabled")
    auto_apply = _boolean(config["auto_apply"], f"{path}.auto_apply")
    if auto_apply and not enabled:
        raise ValueError(f"{path}.auto_apply cannot be true when the operation is disabled")


def _deep_merge(target: dict[str, Any], override: Mapping[str, Any], path: str) -> None:
    for key, value in override.items():
        if str(key).startswith("_"):
            continue
        if key not in target:
            raise ValueError(f"Unknown configuration key: {path}.{key}")
        if isinstance(target[key], Mapping):
            nested = _mapping(value, f"{path}.{key}")
            _deep_merge(target[key], nested, f"{path}.{key}")
        else:
            target[key] = copy.deepcopy(value)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(f"{path} has invalid keys: {'; '.join(details)}")


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{path} must be finite")
    return result


def _fraction(value: Any, path: str) -> float:
    result = _number(value, path)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be between 0 and 1")
    return result


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "CONFIG_VERSION",
    "DEFAULT_ONTOLOGY",
    "DEFAULT_REFINEMENT_CONFIG",
    "config_hash",
    "default_refinement_config",
    "ontology_for_class",
    "normalize_refinement_mode",
    "refinement_directory",
    "select_refinement_config",
    "validate_refinement_config",
]
