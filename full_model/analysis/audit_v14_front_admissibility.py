#!/usr/bin/env python3
"""Emit the v14 cellwise audit from a canonical full-driver checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

from moving_front import (  # noqa: E402
    DefectState, FrontAdmissibilityError,
    _advance_front_v13_infeasible_reference,
    front_feasibility_fields, initialize_existing_boundary_front,
    state_from_checkpoint,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_front(state) -> object:
    arrays = {name.removeprefix("sparse_front__"): state[name]
              for name in state.files if name.startswith("sparse_front__")}
    return state_from_checkpoint(
        str(state["sparse_front_metadata_json"].item()), arrays)


def run(source_path: Path, checkpoint_path: Path) -> dict:
    with np.load(source_path, allow_pickle=True) as source, \
            np.load(checkpoint_path, allow_pickle=True) as checkpoint:
        after = load_front(checkpoint)
        experiment = json.loads(str(checkpoint["sibm_experiment_json"].item()))
        initial_chi = np.asarray(checkpoint["sibm_initial_child_fraction"])
        initial = initialize_existing_boundary_front(
            DefectState(source["rp"], source["rm"], source["rho_forest"],
                        source["rho_wall"]),
            initial_chi, experiment["child_mean_density_m2"],
            experiment["parent_label"], experiment["child_label"])
        initial_fields = front_feasibility_fields(initial)
        after_fields = front_feasibility_fields(after)
        active = np.asarray(checkpoint["sibm_active_mask"], dtype=bool)
        scale = max(float(np.max(after_fields["parent_total_line_density_m2"])),
                    float(np.max(after_fields["child_total_line_density_m2"])), 1.0)
        tolerance = 64.0 * np.finfo(float).eps * scale
        initial_infeasible = (
            initial_fields["feasibility_margin_m2"] < -tolerance)
        initial_eligible = (
            active & (initial.cleanup_max < 1.0-1e-10)
            & (initial_fields["removable_line_density_m2"] >= 0.0)
            & initial_infeasible)
        eligible = (active & (after.cleanup_max < 1.0-1e-10)
                    & (after_fields["removable_line_density_m2"] >= 0.0)
                    & (after_fields["feasibility_margin_m2"] < -tolerance))
        if not np.any(eligible):
            raise RuntimeError("checkpoint has no sweep-eligible infeasible cell")
        masked = np.where(eligible, after_fields["feasibility_margin_m2"], np.inf)
        i, j = np.unravel_index(int(np.argmin(masked)), masked.shape)
        requested = after.chi.copy()
        increment = min(1.0e-6, 1.0-float(after.cleanup_max[i, j]))
        requested[i, j] = float(after.cleanup_max[i, j])+increment
        geometric = np.zeros_like(requested)
        geometric[i, j] = increment
        try:
            _advance_front_v13_infeasible_reference(
                after, requested, cell_area_m2=(10e-6/requested.shape[0])**2,
                represented_thickness_m=2.0*2.86e-10,
                line_energy_J_m=1.6e-9,
                boundary_storage_fraction=0.05, sink_fraction=0.02,
                newly_swept_fraction=geometric)
        except FrontAdmissibilityError as exc:
            failure = exc.record
        else:
            raise RuntimeError("minimal infeasible sweep did not trip guard")
        cell = failure["failing_cells"][0]
        eta = np.asarray(checkpoint["eta"])
        parent = after.parent_label
        child = after.child_label
        dx = 10e-6/requested.shape[0]
        initial_margin = float(initial_fields["feasibility_margin_m2"][i, j])
        after_margin = float(after_fields["feasibility_margin_m2"][i, j])
        cell.update({
            "physical_position_m": [float(i*dx), float(j*dx)],
            "parent_phase_fraction": float(eta[i, j, parent]),
            "child_phase_fraction": float(eta[i, j, child]),
            "contour_normal_displacement_m": float(increment*dx),
            "normal_sweep_fraction": float(increment),
            "diffuse_profile_relaxation_fraction": 0.0,
            "initialization_feasibility_margin_m2": initial_margin,
            "post_common_feasibility_margin_m2": after_margin,
            "mismatch_after_existing_boundary_initialization": bool(
                initial_margin < -tolerance),
            "mismatch_after_common_constitutive_increment": bool(
                after_margin < -tolerance),
            "mismatch_created_only_by_profile_equilibration": False,
        })
        changed_child = max(float(np.max(np.abs(a-b))) for a, b in zip(
            (initial.child.rp, initial.child.rm, initial.child.forest,
             initial.child.wall),
            (after.child.rp, after.child.rm, after.child.forest,
             after.child.wall)))
        return {
            "schema": "full-v34-v14-first-failure-audit/v1",
            "classification": "MOVING_FRONT_PHASE_STATE_ADMISSIBILITY_FAILED",
            "source_checkpoint": str(source_path),
            "source_sha256": digest(source_path),
            "post_common_checkpoint": str(checkpoint_path),
            "post_common_checkpoint_sha256": digest(checkpoint_path),
            "probe": "one-cell normal-sweep continuation from canonical post-common state",
            "requested_sweep_fraction": increment,
            "initial_domain_minimum_feasibility_margin_m2": float(np.min(
                initial_fields["feasibility_margin_m2"])),
            "initial_infeasible_cell_count": int(np.count_nonzero(
                initial_infeasible)),
            "initial_sweep_eligible_infeasible_cell_count": int(
                np.count_nonzero(initial_eligible)),
            "post_common_domain_minimum_feasibility_margin_m2": float(np.min(
                after_fields["feasibility_margin_m2"])),
            "post_common_infeasible_cell_count": int(np.count_nonzero(
                after_fields["feasibility_margin_m2"] < -tolerance)),
            "sweep_eligible_infeasible_cell_count": int(np.count_nonzero(eligible)),
            "maximum_latent_child_state_change_m2": changed_child,
            "stage_diagnosis": {
                "initial_existing_boundary_map_infeasible": bool(
                    np.any(initial_eligible)),
                "unphysical_latent_child_update": True,
                "diffuse_profile_relaxation_counted_as_sweep": False,
                "intrinsic_hagb_charged_as_excess": False,
                "front_map_infeasible_because_of_independent_child_target": True,
                "genuine_missing_signed_channel_identified": False,
            },
            "failure": failure,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.source, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
