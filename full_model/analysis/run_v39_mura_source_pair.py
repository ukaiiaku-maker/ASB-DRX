#!/usr/bin/env python3
"""Matched legacy/current ordering continuation from a rescued V37 state."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v36_mura_rate_limit import context
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.density_state_map import derived_density_fields
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from full_model.production.wall_topology_supply import reservoir_nye_m1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observables(state, systems, topologies, spacing):
    fields = derived_density_fields(state.density, topologies)
    nye = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)["total"]
    area = spacing**2
    return {
        "ordered_line_m_per_m_thickness": float(np.sum(
            fields["rho_wall_ordered_m2"], dtype=np.longdouble)*area),
        "tangle_line_m_per_m_thickness": float(np.sum(
            fields["rho_wall_tangle_m2"], dtype=np.longdouble)*area),
        "total_line_m_per_m_thickness": float(np.sum(
            fields["rho_total_m2"], dtype=np.longdouble)*area),
        "nye_rms_m1": float(np.sqrt(np.mean(nye*nye))),
        "orientation_span_deg": float(np.rad2deg(
            np.max(state.common.orientation_rad)-np.min(
                state.common.orientation_rad))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requested-dt-s", type=float, default=1e-6)
    args = parser.parse_args()
    (state, metadata, fixed, support, systems, topologies, common, extensive,
     kinetics, spacing) = context(args.checkpoint)
    strain = float(metadata["applied_strain"])
    driving = CommonWallDriving(
        mean_strain=np.array([[0.0, .5*strain], [.5*strain, 0.0]]),
        fixed_eigenstrain=fixed)
    variants = {
        "legacy_v37_single_capped_extent": replace(
            extensive, ordering_integration_method="complete_time_explicit",
            ordering_internal_substep_s=1.0,
            ordering_internal_max_substeps=1),
        "current_v39_complete_time_dispatch": replace(
            extensive, ordering_integration_method="implicit_backward_euler",
            ordering_internal_substep_s=5e-11,
            ordering_internal_max_substeps=8192),
    }
    records = {}
    initial = observables(state, systems, topologies, spacing)
    for name, parameters in variants.items():
        start = time.perf_counter()
        updated, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, parameters,
            kinetics, args.requested_dt_s, topology_route_enabled=False,
            mura_work_budget_mode="energy_limited_feasible_extents",
            feasible_family_extent_levels=12)
        records[name] = {
            "wall_seconds": time.perf_counter()-start,
            "accepted_dt_s": float(ledger["accepted_dt_s"]),
            "ordering_integration_method": ledger[
                "ordering_thermodynamics"].get("integration_method"),
            "ordering_stiff_dispatch": ledger[
                "ordering_thermodynamics"].get("stiff_dispatch"),
            "ordering_internal_substeps": int(ledger[
                "ordering_thermodynamics"].get("internal_substeps", 1)),
            "ordering_discarded_reaction_time_s": float(ledger[
                "ordering_thermodynamics"].get(
                    "discarded_reaction_time_s", 0.0)),
            "hard_invariant_passed": bool(ledger["nye_suboperator_audit"][
                "accepted_step_hard_invariant_passed"]),
            "observables": observables(updated, systems, topologies, spacing),
        }
    old = records["legacy_v37_single_capped_extent"]
    new = records["current_v39_complete_time_dispatch"]
    if old["accepted_dt_s"] != new["accepted_dt_s"]:
        raise RuntimeError("matched continuation accepted different physical times")
    comparisons = {}
    for key in initial:
        a = old["observables"][key]; b = new["observables"][key]
        comparisons[key] = {"legacy": a, "current": b,
                            "absolute_difference": abs(a-b),
                            "relative_to_current": abs(a-b)/max(abs(b), 1e-30)}
    result = {
        "schema": "asb-drx/v39/mura-source-pair/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_source_sha": metadata.get("source_sha"),
        "checkpoint_applied_strain": strain,
        "requested_dt_s": args.requested_dt_s,
        "initial_observables": initial, "records": records,
        "comparisons": comparisons,
        "legacy_ordered_observable_inherits_current_accuracy": False,
        "claim_boundary": (
            "current-source restart from a legacy-prepared valid state; this "
            "does not claim current source generated the precursor history"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
