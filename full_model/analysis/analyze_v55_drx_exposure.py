#!/usr/bin/env python3
"""Diagnose physical exposure and limiting channels in a V54/V55 DRX path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


KB_J_K = 1.380649e-23


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owner_total(archive, owner: str) -> np.ndarray:
    prefix = f"common__{owner}__"
    names = (
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
        "junction_m2",
    )
    return sum(np.sum(np.asarray(archive[prefix+name]), axis=2) for name in names)


def weighted_mean(field: np.ndarray, weight: np.ndarray) -> float | None:
    denominator = float(np.sum(weight))
    return None if denominator <= 0.0 else float(np.sum(field*weight)/denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.case_dir/"result.json"
    result = json.loads(result_path.read_text())
    rows = result["records"]
    endpoint = rows[-1]
    first_checkpoint = sorted(args.case_dir.glob("checkpoint_*.npz"))[0]
    final_checkpoint = sorted(args.case_dir.glob("checkpoint_*.npz"))[-1]

    with np.load(first_checkpoint, allow_pickle=False) as initial, np.load(
            final_checkpoint, allow_pickle=False) as final:
        initial_child = np.asarray(initial["eta"])[..., 1]
        final_child = np.asarray(final["eta"])[..., 1]
        initial_parent_core = (initial_child <= 0.1).astype(float)
        initial_child_core = (initial_child >= 0.9).astype(float)
        support = {
            "initial_parent_core_fraction": float(np.mean(initial_parent_core)),
            "initial_child_core_fraction": float(np.mean(initial_child_core)),
            "initial_parent_owner_density_m2": weighted_mean(
                owner_total(initial, "parent"), initial_parent_core),
            "final_parent_owner_density_on_initial_parent_core_m2": weighted_mean(
                owner_total(final, "parent"), initial_parent_core),
            "initial_child_owner_density_m2": weighted_mean(
                owner_total(initial, "child"), initial_child_core),
            "final_child_owner_density_on_initial_child_core_m2": weighted_mean(
                owner_total(final, "child"), initial_child_core),
        }
        resolved_new = final_child-initial_child >= 1.0e-6
        support["resolved_new_material_cell_count_at_delta_eta_1e-6"] = int(
            np.count_nonzero(resolved_new))
        support["final_child_owner_density_on_resolved_new_material_m2"] = (
            weighted_mean(owner_total(final, "child"), resolved_new.astype(float)))

    channel = endpoint.get("front_channel_diagnostics")
    energy = endpoint.get("complete_energy_decision")
    component_displacement: dict[str, float] = {}
    for row in rows:
        for item in (row.get("front_channel_diagnostics") or {}).get(
                "component_motion", []):
            key = str(item["component_id"])
            component_displacement[key] = component_displacement.get(key, 0.0)+float(
                item["normal_displacement_m"])

    energy_changes = None
    if energy is not None:
        energy_changes = {
            key: float(energy["candidate"][key]-energy["before"][key])
            for key in energy["before"] if key in energy["candidate"]
        }
    proposed = None if channel is None else float(channel["proposed_signed_volume_m3"])
    accepted = None if channel is None else float(channel["accepted_signed_volume_m3"])
    payload = {
        "schema": "asb-drx/v55/drx-exposure-diagnosis/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_result": str(result_path.resolve()),
        "source_result_sha256": digest(result_path),
        "physical_time_s": float(result["physical_time_s"]),
        "configuration": result["configuration"],
        "continuum_averaged_motion": {
            "area_equivalent_displacement_m": float(sum(
                row["accepted_contour_displacement_m"] for row in rows)),
            "individual_interface_displacement_m": component_displacement,
            "cumulative_newly_swept_volume_m3": float(
                endpoint["cumulative_newly_swept_volume_m3"]),
            "cumulative_revisit_volume_m3": float(
                endpoint["cumulative_revisit_volume_m3"]),
            "cumulative_processed_line_m": float(
                endpoint["cumulative_processed_line_m"]),
            "cumulative_boundary_stored_line_m": float(
                endpoint["cumulative_boundary_stored_line_m"]),
        },
        "endpoint_kinetics": None if channel is None else {
            "temperature_K": float(channel["channel_a_to_b"]["temperature_K"]),
            "forward_rate_per_site_s": float(channel["rate_a_to_b_s"]),
            "reverse_rate_per_site_s": float(channel["rate_b_to_a_s"]),
            "net_velocity_m_s": float(channel["net_velocity_a_to_b_m_s"]),
            "forward_transition_state_rate_s": float(
                channel["channel_a_to_b"]["transition_state_rate_s"]),
            "reverse_transition_state_rate_s": float(
                channel["channel_b_to_a"]["transition_state_rate_s"]),
            "forward_barrier_over_kT": float(
                channel["channel_a_to_b"]["activation_free_barrier_J"]
                /(KB_J_K*channel["channel_a_to_b"]["temperature_K"])),
            "forward_channel_pressure_Pa": float(
                channel["channel_a_to_b"]["front_channel_pressure_Pa"]),
            "reverse_channel_pressure_Pa": float(
                channel["channel_b_to_a"]["front_channel_pressure_Pa"]),
            "gross_expected_events_this_interval": float(
                channel["expected_events_a_to_b"]+channel["expected_events_b_to_a"]),
            "net_expected_events_this_interval": float(
                channel["expected_signed_event_count"]),
            "accepted_to_proposed_volume_ratio": (
                None if not proposed else abs(accepted/proposed)),
            "topology_backtrack_fraction": float(
                channel["topology_backtrack_fraction"]),
            "maximum_abs_phase_change": float(channel["maximum_abs_phase_change"]),
        },
        "endpoint_complete_energy_changes_J": energy_changes,
        "fixed_material_support_diagnostics": support,
        "limitation_classification": (
            "PHYSICAL_KINETIC_EXPOSURE_LIMITED_NOT_TOPOLOGY_OR_PROPOSAL_LIMITED"
            if channel is not None
            and channel["topology_backtrack_fraction"] == 1.0
            and proposed and abs(accepted/proposed) < 1.0e-4
            else "UNRESOLVED_OR_OTHER_LIMITATION"),
        "unavailable": [
            "a uniquely attributable continuum mobility independent of the declared atomic-event law",
            "material rehardening of newly transformed cells when no cell reaches delta_eta >= 1e-6",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "output": str(args.output),
        "classification": payload["limitation_classification"],
        "sha256": digest(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
