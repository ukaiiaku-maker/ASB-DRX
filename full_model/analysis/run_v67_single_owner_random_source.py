#!/usr/bin/env python3
"""Production-derived one-owner test of random sources and wall rotation.

This is a prepared mechanism control, not an endogenous grain-birth claim.
One retained material owner is extended over the periodic section so the
calculation can resolve intragranular topology without the n16 polycrystal's
2--20-cell owner interiors.  No orientation, density, or plastic-history
field is reset.  The ordinary production mechanics, complete-energy random
Hall--Petch pair source, and ordered-wall orientation transaction share one
physical clock.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, _save_checkpoint,
)
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)
from full_model.production.common_subgrain_recognition import (
    recognize_common_orientation_plateau,
)
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from full_model.production.multigrain_production import MultiGrainProductionRuntime
from full_model.production.orientation_wall_compatibility import (
    evaluate_orientation_wall_compatibility, ordered_wall_nye_from_common,
)
from full_model.production.wall_orientation_relaxation import (
    WallOrientationRelaxationParameters,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _density(owner):
    return sum(np.asarray(getattr(owner, name)) for name in (
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2"
    )).sum(axis=2)+np.sum(owner.junction_m2, axis=2)


def _serial_recognition(value):
    return {key: item for key, item in value.items()
            if not isinstance(item, np.ndarray)}


def _recognitions(owner, systems, spacing):
    domain = np.ones(owner.orientation_rad.shape, dtype=bool)
    density = _density(owner)
    plastic = recognize_common_orientation_plateau(
        owner.orientation_rad, owner.family_nye_m1, owner.wall_order,
        density, spacing, domain_mask=domain)
    ordered = ordered_wall_nye_from_common(owner, systems)
    structural = recognize_common_orientation_plateau(
        owner.orientation_rad, ordered, owner.wall_order, density, spacing,
        domain_mask=domain)
    combined = recognize_common_orientation_plateau(
        owner.orientation_rad,
        np.sum(owner.family_nye_m1, axis=2)+ordered,
        owner.wall_order, density, spacing, domain_mask=domain)
    return {
        "plastic_incompatibility_nye": _serial_recognition(plastic),
        "ordered_structural_nye": _serial_recognition(structural),
        "combined_declared_nye": _serial_recognition(combined),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--owner-id", type=int, default=10)
    parser.add_argument("--intervals", type=int, default=10)
    parser.add_argument("--dt", type=float, default=8e-4)
    parser.add_argument("--shear-rate", type=float, default=10.0)
    parser.add_argument("--source-seed", type=int, default=67)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_status = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=ROOT, text=True)
    state, _, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(args.checkpoint))
    try:
        owner_index = state.grain_ids.index(args.owner_id)
    except ValueError as error:
        raise ValueError("requested material owner is absent") from error
    owner = state.owners[owner_index]
    grid = owner.temperature_K.shape
    single = MultiGrainCommonState(
        (args.owner_id,), np.ones((1,)+grid), (owner,))
    single.validate()
    spacing, thickness, systems, wall, _, _, context = _physics(
        _args_from_configuration(configuration))
    wall = replace(wall, wall_nye_match_coefficient_J_m=2.0e-5)
    source_parameters = ArrheniusSourceParameters(
        grain_size_m=float(configuration.get("length_m", spacing*grid[0])),
        pair_increment_m2=1.0e10)
    context = replace(
        context, wall_parameters=wall, shear_rate_s=args.shear_rate,
        front_enabled=False, source_parameters=source_parameters,
        source_mode="random_bulk", source_random_seed=args.source_seed,
        sink_mode="disabled", interface_sink_mode="disabled",
        structural_absorption_mode="disabled",
        lattice_absorption_mode="disabled",
        wall_orientation_relaxation_mode="ordered_nye",
        wall_orientation_relaxation_parameters=(
            WallOrientationRelaxationParameters()))
    endpoint = make_endpoint(
        single, MultiGrainProductionRuntime(()), load, context)
    initial = owner
    records = []
    for index in range(1, args.intervals+1):
        interval = advance_complete_multigrain_interval(
            endpoint, args.dt, context)
        endpoint = interval.endpoint
        current = endpoint.state.owners[0]
        compatibility = evaluate_orientation_wall_compatibility(
            current, systems, np.ones(grid, dtype=bool), spacing)
        records.append({
            "interval": index,
            "physical_time_s": endpoint.runtime.ledger.physical_time_s,
            "applied_shear_strain": endpoint.load,
            "source_classification": interval.audit["source_classification"],
            "source_event_count": interval.audit["source_event_count"],
            "source_relative_first_law_residual": interval.audit[
                "source_relative_first_law_residual"],
            "source_line_balance_residual_m2": interval.audit[
                "source_line_balance_residual_m2"],
            "mechanical_relative_first_law_residual": interval.audit[
                "mechanical_relative_first_law_residual"],
            "wall_orientation_classification": interval.audit[
                "wall_orientation_relaxation_classification"],
            "wall_orientation_fraction": interval.audit[
                "wall_orientation_relaxation_fractions"].get(args.owner_id, 0.0),
            "wall_orientation_affinity_J_m": interval.audit[
                "wall_orientation_relaxation_affinities_J_m"].get(
                    args.owner_id, 0.0),
            "orientation_span_deg": float(np.degrees(np.ptp(
                current.orientation_rad))),
            "compatible_target_span_deg": float(np.degrees(
                compatibility.reconstructed_orientation_span_rad)),
            "wall_order_mean": float(np.mean(current.wall_order)),
            "wall_order_range": [float(np.min(current.wall_order)),
                                 float(np.max(current.wall_order))],
        })
        print(json.dumps(records[-1]), flush=True)
    final = endpoint.state.owners[0]
    result_configuration = dict(configuration)
    result_configuration.update(
        schema="v67-single-owner-random-source-v1",
        owner_id=args.owner_id, intervals=args.intervals, dt_s=args.dt,
        shear_rate_s=args.shear_rate, source_mode="random_bulk",
        source_seed=args.source_seed,
        wall_nye_match_coefficient_J_m=2.0e-5)
    checkpoint = args.out/"checkpoint_latest.npz"
    temporary = args.out/"checkpoint_latest.tmp.npz"
    _save_checkpoint(
        temporary, endpoint.state, endpoint.runtime, step+args.intervals,
        endpoint.load, initial_volume, result_configuration,
        {"source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
         "parent_provenance": provenance})
    os.replace(temporary, checkpoint)
    recognitions = _recognitions(final, systems, spacing)
    payload = {
        "schema": "v67-single-owner-random-source-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "source_tree_clean": not bool(source_status),
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha(args.checkpoint),
        "scope": "prepared retained-material one-owner mechanism control",
        "configuration": {
            "owner_id": args.owner_id, "grid": list(grid),
            "spacing_m": spacing, "represented_thickness_m": thickness,
            "intervals": args.intervals, "dt_s": args.dt,
            "added_physical_time_s": args.intervals*args.dt,
            "shear_rate_s": args.shear_rate,
            "added_engineering_shear": (
                args.intervals*args.dt*args.shear_rate),
            "source_mode": "random_bulk", "source_seed": args.source_seed,
            "source_parameters": asdict(source_parameters),
            "wall_nye_match_coefficient_J_m": 2.0e-5,
        },
        "initial": {
            "orientation_span_deg": float(np.degrees(np.ptp(
                initial.orientation_rad))),
            "wall_order_mean": float(np.mean(initial.wall_order)),
        },
        "records": records,
        "final_recognition": recognitions,
        "ledger": asdict(endpoint.runtime.ledger),
        "decision": {
            "all_source_events_accepted": all(
                row["source_classification"].startswith("ACCEPTED")
                for row in records),
            "all_hard_balances_passed": all(
                row["source_relative_first_law_residual"] < .05
                and row["mechanical_relative_first_law_residual"] < .05
                for row in records),
            "strict_plastic_nye_subgrain": bool(
                recognitions["plastic_incompatibility_nye"].get(
                    "qualified", False)),
            "strict_ordered_structural_subgrain": bool(
                recognitions["ordered_structural_nye"].get(
                    "qualified", False)),
            "strict_combined_nye_subgrain": bool(
                recognitions["combined_declared_nye"].get(
                    "qualified", False)),
            "grain_count_change": 0,
        },
        "claim_boundary": (
            "Prepared one-owner topology test; extending one retained owner "
            "over the section is not endogenous nucleation or DRX."),
        "artifacts": {"checkpoint_latest.npz": _sha(checkpoint)},
    }
    (args.out/"result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
