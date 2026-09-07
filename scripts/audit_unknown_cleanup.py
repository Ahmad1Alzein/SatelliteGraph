"""Measure unknown areas and contextual evidence without changing segmentation."""
from collections import Counter
from pathlib import Path
import json

from src.config import load_config
from src.refinement.config import select_refinement_config
from src.refinement.pipeline import _load_stage2_source, _normalize
from src.refinement.evidence import pair_evidence
from src.refinement.contextual_cleanup import evaluate_contextual_micro_cleanup
from src.stage2.artifacts import atomic_json


def main():
    root = Path.cwd()
    cfg = select_refinement_config(load_config())
    cfg["operations"]["contextual_cleanup"]["supported_maximum_energy"] = 0.40
    output = []
    for index in range(4):
        tile_id = f"IKONOS_2005-002__tile_{index:05d}"
        source = _load_stage2_source(root, tile_id, cfg)
        normalized = _normalize(source, cfg)
        rows = evaluate_contextual_micro_cleanup(
            normalized.records, normalized.instance_map,
            pair_evidence(normalized.instance_map, source.rgb),
            source.semantic_probabilities, cfg["operations"]["contextual_cleanup"],
        )
        summary = {"tile_id": tile_id, "reasons": dict(Counter(
            reason for row in rows if row["action"] == "abstain" for reason in row["reasons"]
        ))}
        summary["sweep"] = []
        for ceiling in (0.40, 0.45, 0.50, 0.55, 0.60):
            selected = []
            for row in rows:
                ev = row["evidence"]
                if row["action"] == "absorb":
                    selected.append(row)
                    continue
                if row["reasons"] != ["selected_energy_above_automatic_maximum"]:
                    continue
                candidates = sorted((r for r in ev["target_candidates"] if r["eligible"]), key=lambda r: r["energy"])
                if not candidates:
                    continue
                best = candidates[0]
                # Require at least moderate source probability and RGB support,
                # plus contact with an adjacent target; no size-only relabeling.
                if (best["energy"] <= ceiling
                    and best["target_class_id"] in ev["top_two_class_ids"]
                    and best["mean_target_probability_on_source"] >= 0.20
                    and best["source_boundary_fraction"] >= 0.25
                    and best["weak_color_boundary_score"] >= 0.50
                    and best["region_mean_color_similarity"] >= 0.50
                    and (ev["best_to_runner_up_energy_margin"] is None or ev["best_to_runner_up_energy_margin"] >= 0.02)):
                    selected.append(row)
            summary["sweep"].append({"ceiling": ceiling, "absorbs": len(selected), "pixels": sum(r["evidence"]["source_area_pixels"] for r in selected)})
        print(json.dumps(summary), flush=True)
        output.append({**summary, "evaluations": rows})
    atomic_json(root / "artifacts" / "unknown_cleanup_audit.json", {"tiles": output})


if __name__ == "__main__":
    main()
