#!/usr/bin/env python3
"""Screen conservative core-to-wall collective climb on a retained state."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import ndimage

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.collective_climb_wall_capture import (
    collective_climb_wall_capture_candidate,
    collective_climb_wall_orientation_candidate,
)
from full_model.production.collective_wall_nucleation import _periodic_radius
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.fixed_scale_refinement import (
    refine_multigrain_common_state,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _separated_centers(score, count, exclusion_cells):
    work = np.asarray(score, dtype=float).copy()
    centers = []
    x, y = np.indices(work.shape)
    for _ in range(int(count)):
        if not np.any(np.isfinite(work)) or float(np.max(work)) <= 0.0:
            break
        center = tuple(map(int, np.unravel_index(np.argmax(work), work.shape)))
        centers.append(center)
        dx = np.minimum((x-center[0]) % work.shape[0],
                        (center[0]-x) % work.shape[0])
        dy = np.minimum((y-center[1]) % work.shape[1],
                        (center[1]-y) % work.shape[1])
        work[dx*dx+dy*dy <= int(exclusion_cells)**2] = -np.inf
    return centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--centers-per-owner", type=int, default=4)
    parser.add_argument("--refinement-factor", type=int, default=1)
    parser.add_argument("--clearance-m", type=float, default=2e-7)
    parser.add_argument("--match-length-m", type=float, default=1.5e-7)
    parser.add_argument("--ring-width-m", type=float, default=4.5e-7)
    parser.add_argument("--order-increment", type=float, default=.01)
    parser.add_argument("--radii-m", type=float, nargs="+",
                        default=(4.5e-7, 7.5e-7, 1.05e-6, 1.35e-6, 1.65e-6))
    parser.add_argument("--coordinates", type=float, nargs="+",
                        default=(.01, .025, .05, .10, .20))
    args = parser.parse_args()
    state, runtime, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(args.checkpoint))
    physics_args = _args_from_configuration(configuration)
    spacing, _, systems, _, _, _, context = _physics(physics_args)
    refinement_audit = None
    if args.refinement_factor != 1:
        state, refinement_audit = refine_multigrain_common_state(
            state, args.refinement_factor)
        spacing /= args.refinement_factor
    wall_parameters = replace(
        context.wall_parameters,
        wall_nye_match_coefficient_J_m=(
            context.wall_parameters.c44_Pa*args.match_length_m**2))
    strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    energy_kwargs = dict(context.energy_kwargs)
    energy_kwargs.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing,
        represented_thickness_m=context.represented_thickness_m,
        wall_parameters=wall_parameters, mean_strain=strain,
        **energy_kwargs)
    before = evaluate_complete_multigrain_energy(state, **common)
    records = []
    for owner_index, (grain_id, support, owner) in enumerate(zip(
            state.grain_ids, state.supports, state.owners)):
        owner_domain = np.asarray(support) > .75
        clearance_cells = int(np.ceil(args.clearance_m/spacing))
        domain = (ndimage.binary_erosion(owner_domain,
                                         iterations=clearance_cells)
                  if clearance_cells else owner_domain)
        excess = np.sum(np.abs(
            np.asarray(owner.mobile_plus_m2)
            -np.asarray(owner.mobile_minus_m2)), axis=2)
        score = np.where(domain, excess, -np.inf)
        centers = _separated_centers(
            score, args.centers_per_owner,
            max(1, int(round(min(args.radii_m)/spacing))))
        for center in centers:
            radial = _periodic_radius(domain.shape, center, spacing)
            for radius in args.radii_m:
                receiver = (np.exp(-.5*((radial-radius)
                                        /args.ring_width_m)**2)*domain)
                donor = ((radial < max(radius-.5*args.ring_width_m, spacing))
                         &domain).astype(float)
                if float(np.sum(receiver)) < 8.0 or not np.any(donor):
                    continue
                for coordinate in args.coordinates:
                    wall, moved = collective_climb_wall_capture_candidate(
                        state, owner_index, donor, receiver, coordinate,
                        args.order_increment)
                    if moved <= 0.0:
                        continue
                    wall_energy = evaluate_complete_multigrain_energy(
                        wall, **common)
                    joint, _, rotation = (
                        collective_climb_wall_orientation_candidate(
                            state, owner_index, donor, receiver, coordinate,
                            args.order_increment, systems, spacing, domain))
                    joint_energy = evaluate_complete_multigrain_energy(
                        joint, **common)
                    records.append({
                        "grain_id": int(grain_id),
                        "center": list(center),
                        "radius_m": float(radius),
                        "coordinate": float(coordinate),
                        "moved_line_cell_sum_m2": moved,
                        "maximum_incremental_rotation_deg": float(
                            np.degrees(rotation)),
                        "wall_only_delta_helmholtz_J": float(
                            wall_energy.helmholtz_J-before.helmholtz_J),
                        "joint_delta_helmholtz_J": float(
                            joint_energy.helmholtz_J-before.helmholtz_J),
                    })
    best = (min(records, key=lambda row: row["joint_delta_helmholtz_J"])
            if records else None)
    root = Path(__file__).resolve().parents[2]
    payload = {
        "schema": "v67-conservative-collective-climb-wall-probe-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=root, text=True).strip(),
        "source_tree_clean": not bool(subprocess.check_output(
            ("git", "status", "--short", "--untracked-files=no"),
            cwd=root, text=True).strip()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha(args.checkpoint),
        "configuration": {
            "centers_per_owner": args.centers_per_owner,
            "refinement_factor": args.refinement_factor,
            "spacing_m": spacing,
            "clearance_m": args.clearance_m,
            "match_length_m": args.match_length_m,
            "ring_width_m": args.ring_width_m,
            "order_increment": args.order_increment,
            "radii_m": list(map(float, args.radii_m)),
            "coordinates": list(map(float, args.coordinates)),
            "transport_invariant": "support_weighted_sign_and_family",
            "orientation_map": "incremental_ordered_Nye_reconstruction",
        },
        "records": records,
        "refinement_audit": refinement_audit,
        "decision": {
            "candidate_count": len(records),
            "wall_only_downhill_count": sum(
                row["wall_only_delta_helmholtz_J"] <= 0.0
                for row in records),
            "joint_downhill_count": sum(
                row["joint_delta_helmholtz_J"] <= 0.0
                for row in records),
            "best_joint_candidate": best,
            "production_promotion_authorized": False,
        },
        "claim_boundary": (
            "Finite conservative collective-climb discriminator only. A "
            "downhill endpoint would still require path, kinetics, balance, "
            "timestep, and production qualification before promotion."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
