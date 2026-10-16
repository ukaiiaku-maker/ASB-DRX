#!/usr/bin/env python3
"""Screen finite wall topology jointly with its Nye-implied orientation.

This is a discriminator, not a production constitutive operator.  It asks
whether the existing conservative mobile-to-wall ring candidate is rejected
only because it was previously priced before its conjugate orientation could
respond.  No grain label or target orientation is supplied: every orientation
candidate is reconstructed from that candidate's own ordered wall Nye field.
"""

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
from full_model.production.collective_wall_nucleation import (
    CollectiveWallNucleationParameters, _collective_wall_candidate,
    _periodic_radius,
)
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.orientation_wall_compatibility import (
    ordered_wall_nye_from_common, reconstruct_orientation_from_ordered_nye,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _joint_candidate(state, owner_index, ring, coordinate, systems, spacing,
                     clearance_m, order_increment):
    wall, transferred = _collective_wall_candidate(
        state, owner_index, ring, order_increment, coordinate)
    old = state.owners[owner_index]
    candidate = wall.owners[owner_index]
    support = np.asarray(state.supports[owner_index])
    domain = support > .75
    cells = int(np.ceil(float(clearance_m)/float(spacing)))
    if cells:
        domain = ndimage.binary_erosion(domain, iterations=cells)
    if np.count_nonzero(domain) < 2:
        return wall, transferred, 0.0
    alpha = ordered_wall_nye_from_common(candidate, systems)
    reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        alpha, candidate.orientation_rad, domain, spacing)
    increment = np.arctan2(
        np.sin(reconstructed-np.asarray(old.orientation_rad)),
        np.cos(reconstructed-np.asarray(old.orientation_rad)))
    orientation = np.asarray(old.orientation_rad).copy()
    orientation[domain] += float(coordinate)*increment[domain]
    # Structural interface content owns each grain's mean orientation.  The
    # finite topology probe can create gradients but cannot rotate an owner as
    # a rigid body for free.
    orientation += float(np.mean(old.orientation_rad)-np.mean(orientation))
    owners = list(wall.owners)
    owners[owner_index] = replace(candidate, orientation_rad=orientation)
    joint = replace(wall, owners=tuple(owners))
    joint.validate()
    change = float(np.max(np.abs(orientation-old.orientation_rad)))
    return joint, transferred, change


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clearance-m", type=float, default=2e-7)
    parser.add_argument("--match-length-m", type=float, default=1.5e-7)
    parser.add_argument("--order-increment", type=float, default=.01)
    parser.add_argument(
        "--coordinates", type=float, nargs="+",
        default=(.05, .10, .20, .40, .70, 1.0))
    args = parser.parse_args()
    state, runtime, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(args.checkpoint))
    physics_args = _args_from_configuration(configuration)
    spacing, _, systems, _, _, _, context = _physics(physics_args)
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
    parameters = CollectiveWallNucleationParameters(
        order_increment=args.order_increment)
    records = []
    for owner_index, (grain_id, support, owner) in enumerate(zip(
            state.grain_ids, state.supports, state.owners)):
        domain = np.asarray(support) > parameters.owner_support_threshold
        mobile_excess = np.sum(np.abs(
            np.asarray(owner.mobile_plus_m2)
            -np.asarray(owner.mobile_minus_m2)), axis=2)
        score = np.where(domain, mobile_excess, -np.inf)
        if not np.any(np.isfinite(score)) or float(np.max(score)) <= 0.0:
            continue
        center = tuple(map(int, np.unravel_index(np.argmax(score), score.shape)))
        radial = _periodic_radius(domain.shape, center, spacing)
        for radius in parameters.candidate_radii_m:
            ring = (np.exp(-.5*((radial-radius)/parameters.ring_width_m)**2)
                    *domain)
            if float(np.sum(ring)) < 8.0:
                continue
            for coordinate in args.coordinates:
                wall, transferred = _collective_wall_candidate(
                    state, owner_index, ring, args.order_increment, coordinate)
                if transferred <= 0.0:
                    continue
                wall_energy = evaluate_complete_multigrain_energy(wall, **common)
                joint, _, maximum_rotation = _joint_candidate(
                    state, owner_index, ring, coordinate, systems, spacing,
                    args.clearance_m, args.order_increment)
                joint_energy = evaluate_complete_multigrain_energy(joint, **common)
                records.append({
                    "grain_id": int(grain_id), "center": list(center),
                    "radius_m": float(radius),
                    "coordinate": float(coordinate),
                    "transferred_line_cell_sum_m2": transferred,
                    "maximum_wall_implied_rotation_deg": float(np.degrees(
                        maximum_rotation)),
                    "wall_only_delta_helmholtz_J": float(
                        wall_energy.helmholtz_J-before.helmholtz_J),
                    "joint_delta_helmholtz_J": float(
                        joint_energy.helmholtz_J-before.helmholtz_J),
                })
    best = (min(records, key=lambda item: item["joint_delta_helmholtz_J"])
            if records else None)
    payload = {
        "schema": "v67-joint-wall-topology-orientation-probe-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[2],
            text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha(args.checkpoint),
        "configuration": {
            "clearance_m": args.clearance_m,
            "match_length_m": args.match_length_m,
            "order_increment": args.order_increment,
            "coordinates": list(map(float, args.coordinates)),
            "candidate_radii_m": list(parameters.candidate_radii_m),
            "center_rule": "maximum_mobile_signed_excess_per_owner",
            "preserve_owner_mean_orientation": True,
        },
        "records": records,
        "decision": {
            "candidate_count": len(records),
            "wall_only_downhill_count": sum(
                row["wall_only_delta_helmholtz_J"] <= 0.0 for row in records),
            "joint_downhill_count": sum(
                row["joint_delta_helmholtz_J"] <= 0.0 for row in records),
            "best_joint_candidate": best,
            "production_promotion_authorized": False,
            "claim_boundary": (
                "This endpoint screen can identify an accessible conjugate "
                "path but cannot itself publish state or qualify DRX."),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
