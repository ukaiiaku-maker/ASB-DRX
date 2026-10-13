#!/usr/bin/env python3
"""Classify threshold-component changes in accepted V65 checkpoints.

This audit prevents a split of one ``mean+std`` plastic-power component from
being mistaken for the birth of a narrow localization band.  It is read-only:
the accepted checkpoint states and the production trajectory are not changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v53_asb_mechanism import (
    field_metrics,
    overlap,
    periodic_components,
)
from full_model.analysis.postprocess_v59_physical_asb import (
    _physical_configuration,
    _wall_parameters,
    checkpoint_snapshot,
    periodic_identity,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.multigrain_production import (
    multigrain_instantaneous_dissipation_fields,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power_components(path: Path) -> tuple[dict, list[np.ndarray], float]:
    """Reconstruct the accepted-state power field and threshold components."""
    state, _, _, gamma, _, configuration, _ = _load_checkpoint(path)
    if configuration is None:
        raise ValueError(f"checkpoint lacks bound configuration: {path}")
    configuration = _physical_configuration(configuration)
    spacing = float(configuration["length_m"])/int(configuration["n"])
    wall = _wall_parameters(configuration, spacing)
    strain = np.array([[0.0, 0.5*gamma], [0.5*gamma, 0.0]])
    fields = multigrain_instantaneous_dissipation_fields(
        state,
        driving=CommonWallDriving(mean_strain=strain),
        systems=bcc_four_family_systems(),
        topologies=(),
        wall_parameters=wall,
    )
    power = np.asarray(fields["plastic_power_W_m3"], dtype=float)
    metrics, _ = field_metrics(power, spacing)
    components = periodic_components(power > metrics["threshold"])
    components.sort(key=np.count_nonzero, reverse=True)
    return metrics, components, spacing


def transition_metrics(
    previous: list[np.ndarray], current: list[np.ndarray], spacing_m: float
) -> dict:
    """Measure identity of the old largest component and all its fragments."""
    if not previous or not current:
        raise ValueError("transition audit requires components at both checkpoints")
    old = previous[0]
    identities = []
    for item in current:
        identity = periodic_identity(
            old, item, spacing_m, maximum_displacement_m=spacing_m
        )
        identity["unshifted_overlap"] = overlap(old, item)
        identities.append(identity)
    associated_indices = [
        index
        for index, item in enumerate(identities)
        if item["unshifted_overlap"] > 0.0
    ]
    associated_union = np.logical_or.reduce(
        [current[index] for index in associated_indices]
    )
    union_identity = periodic_identity(
        old, associated_union, spacing_m, maximum_displacement_m=spacing_m
    )
    return {
        "previous_component_cell_counts": [int(item.sum()) for item in previous],
        "current_component_cell_counts": [int(item.sum()) for item in current],
        "current_component_count_change": len(current)-len(previous),
        "old_largest_to_each_current": identities,
        "associated_current_component_indices": associated_indices,
        "associated_fragment_union_cell_count": int(associated_union.sum()),
        "old_largest_to_associated_fragment_union": union_identity,
        "unshifted_old_largest_to_associated_fragment_union_overlap": overlap(
            old, associated_union
        ),
    }


def classify_transition(
    transition: dict, previous_participation: float, current_participation: float
) -> dict:
    """Separate threshold splitting from largest-component rank exchange."""
    retained = transition["old_largest_to_associated_fragment_union"]["overlap"]
    associated = transition["associated_current_component_indices"]
    component_split = (
        transition["current_component_count_change"] > 0
        and len(associated) > 1
        and retained >= 0.90
    )
    largest_rank_swap = bool(
        associated and 0 not in associated and retained >= 0.90
    )
    stable_largest_identity = bool(
        associated == [0] and retained >= 0.90
        and transition["current_component_count_change"] == 0
    )
    participation_nonlocalizing = current_participation >= previous_participation
    if component_split and participation_nonlocalizing:
        classification = "THRESHOLD_COMPONENT_SPLIT_WITHOUT_GLOBAL_LOCALIZATION"
        reason = (
            "The old largest threshold component split into multiple fragments "
            "whose union preserves its identity, while the whole-field inverse "
            "participation fraction increased. The largest-fragment width and "
            "aspect jump is therefore not evidence of global physical localization."
        )
    elif largest_rank_swap and participation_nonlocalizing:
        classification = "LARGEST_COMPONENT_RANK_SWAP_WITHOUT_GLOBAL_LOCALIZATION"
        reason = (
            "The former largest component retains its identity as a lower-ranked "
            "component while another component becomes largest by cell count, and "
            "the whole-field inverse participation fraction increases. Consecutive "
            "largest-component morphology values therefore describe different "
            "physical structures and do not establish localization."
        )
    elif stable_largest_identity and participation_nonlocalizing:
        classification = "STABLE_COMPONENT_IDENTITY_WITHOUT_GLOBAL_LOCALIZATION"
        reason = (
            "The largest component retains its physical identity across the "
            "checkpoint pair, but the whole-field inverse participation fraction "
            "increases. Its morphology is continuous over this pair without "
            "providing evidence of global physical localization."
        )
    else:
        classification = "UNRESOLVED_COMPONENT_TRANSITION"
        reason = (
            "The registered split/rank-swap and nonlocalization conditions were "
            "not jointly established; no strict ASB claim is made."
        )
    return {
        "classification": classification,
        "component_split": component_split,
        "largest_component_rank_swap": largest_rank_swap,
        "stable_largest_component_identity": stable_largest_identity,
        "associated_fragment_union_identity_overlap": retained,
        "global_power_participation_became_no_more_localized": (
            participation_nonlocalizing
        ),
        "strict_asb_claim": False,
        "reason": reason,
    }


def analyze(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("at least two checkpoints are required")
    records = []
    components = []
    spacing = None
    for path in paths:
        snapshot, _ = checkpoint_snapshot(path)
        power, found, item_spacing = power_components(path)
        if spacing is not None and item_spacing != spacing:
            raise ValueError("all transition checkpoints must use one grid")
        spacing = item_spacing
        morphology = snapshot["plastic_power_component_morphology"]
        records.append({
            "adaptive_interval": int(snapshot["step"]),
            "physical_time_s": float(snapshot["physical_time_s"]),
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": digest(path),
            "threshold_W_m3": float(power["threshold"]),
            "component_count": len(found),
            "component_cell_counts": [int(item.sum()) for item in found],
            "largest_component_area_fraction": float(morphology["area_fraction"]),
            "largest_component_topology": morphology["second_moment_widths"][
                "topology"
            ],
            "largest_component_aspect_ratio": float(morphology["aspect_ratio"]),
            "largest_component_minor_fwhm_m": float(
                snapshot["power_width_minor_m"]
            ),
            "inverse_participation_fraction": float(
                power["inverse_participation_fraction"]
            ),
            "heat_power_component_overlap": float(
                snapshot["heat_power_component_overlap"]
            ),
        })
        components.append(found)
    transitions = [
        {
            "from_interval": records[index]["adaptive_interval"],
            "to_interval": records[index+1]["adaptive_interval"],
            **transition_metrics(components[index], components[index+1], spacing),
        }
        for index in range(len(records)-1)
    ]
    last = transitions[-1]
    return {
        "schema": "asb-drx-v65-power-component-transition-v1",
        "selection_rule": "periodic components of plastic power > mean+std",
        "trajectory_modified": False,
        "records": records,
        "transitions": transitions,
        "decision": classify_transition(
            last,
            records[-2]["inverse_participation_fraction"],
            records[-1]["inverse_participation_fraction"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.checkpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
