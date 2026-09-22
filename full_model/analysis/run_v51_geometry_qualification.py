#!/usr/bin/env python3
"""Decision evidence for V51 event measure and commuting geometry."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v46_geometry_representation import energy_terms_J
from full_model.analysis.run_v50_subcell_qualification import (
    dimensionless_reference, fixture, row,
)
from full_model.production.subcell_segment_geometry import (
    propose_subcell_face_extension, propose_subcell_translation,
    subcell_line_surface_nye,
)
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, accepted_subcell_face_transaction,
    accepted_subcell_x_faces_shared_clock, synchronize_common,
)
from tests.test_v50_production_subcell_geometry import intrinsic_kinetics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_state(state, data, displacement):
    proposal = propose_subcell_face_extension(
        state.subcell_geometry, state.density, state.reservoir_alignment,
        state.common, data[4], displacement)
    return synchronize_common(V24MechanicalWallState(
        proposal[3], proposal[1], proposal[2], None, proposal[0]), data[5])


def compatibility_rows():
    rows = []
    for n in (32, 64, 128):
        base, data = fixture(n)
        state, data = fixture(n, offset_fraction=0.0)
        # Match the reviewed y offset exactly; fixture already applies -0.17 dx.
        _, _, initial = subcell_line_surface_nye(
            state.subcell_geometry, len(data[4]))
        proposal = propose_subcell_face_extension(
            state.subcell_geometry, state.density,
            state.reservoir_alignment, state.common, data[4], 1e-8)
        rows.append({
            "grid": n, "spacing_m": data[9],
            "initial_line_surface_relative": initial[
                "line_surface_nye_residual_relative"],
            "event_line_surface_relative": proposal[-1][
                "line_surface_event_residual_relative"],
            "event_residual_rms_m-1": proposal[-1][
                "line_surface_event_residual_rms_m-1"],
            "event_reference_rms_m-1": proposal[-1][
                "line_surface_event_reference_rms_m-1"],
            "surface_affine_scale": initial["positivity_affine_scale"],
            "surface_minimum": initial["minimum_surface_fraction"],
            "surface_maximum": initial["maximum_surface_fraction"],
            "scientific_compatibility_passed": bool(
                initial["scientific_compatibility_passed"]
                and proposal[-1]["line_surface_event_compatibility_passed"]),
        })
    return rows


def force_records():
    state, data = fixture(128)
    h_values = (1e-9, .5e-9)
    local = []
    for h in h_values:
        plus = energy_terms_J(candidate_state(state, data, h), data)
        minus = energy_terms_J(candidate_state(state, data, -h), data)
        local.append({
            "shape_step_m": h,
            "ordered_gradient_local_force_N": -(
                plus["ordered_gradient"]-minus["ordered_gradient"])/(2*h),
            "complete_local_force_N": -(plus["total"]-minus["total"])/(2*h),
        })
    secant = row(128)
    reference_integral, reference_energy = dimensionless_reference()
    return {
        "label_correction": (
            "finite 10 nm quotient is a secant; centered values are local derivatives"),
        "continuous_dimensionless_integral": reference_integral,
        "continuous_ordered_gradient_force_N": -reference_energy/1e-8,
        "ordered_gradient_10nm_secant_N": -secant[
            "ordered_gradient_increment_J"]/1e-8,
        "complete_10nm_secant_N": -secant["complete_increment_J"]/1e-8,
        "local_centered_derivatives": local,
    }


def event_measure_record():
    state, data = fixture(32)
    result, ledger = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": -1e-8},
        data[4], data[5], data[1], data[6], data[7],
        intrinsic_kinetics(), 1e-6)
    if not ledger["accepted"] or result is state:
        raise RuntimeError("V51 event-measure qualification transaction rejected")
    keys = (
        "event_measure_convention", "physical_event_count",
        "event_count_from_site_jump_identity",
        "event_count_site_jump_identity_residual",
        "signed_material_exchange_count", "independent_geometry_exchange_count",
        "legacy_area_over_b2_event_count_comparator",
        "legacy_mixed_event_count_comparator",
        "legacy_mixed_affinity_rate_s_comparator", "event_rate_s",
        "observed_accepted_velocity_m_s", "consumed_duration_s",
        "complete_energy_change_J_m3_cells",
        "heat_plus_complete_energy_residual_J_m3_cells",
    )
    return {key: ledger[key] for key in keys}


def shared_clock_record():
    state, data = fixture(32)
    events = [
        {"face": "lower_x", "proposed_displacement_m": 1e-7,
         "fixed_rate_s": 1e8},
        {"face": "upper_x", "proposed_displacement_m": -1e-7,
         "fixed_rate_s": 2e8},
    ]
    result, ledger = accepted_subcell_x_faces_shared_clock(
        state, events, data[4], data[5], data[1], data[6], data[7],
        intrinsic_kinetics(), 1e-7)
    if not ledger["accepted"]:
        raise RuntimeError("V51 shared-clock qualification transaction rejected")
    return {
        "face_ledgers": ledger["face_ledgers"],
        "common_elapsed_time_s": ledger["common_elapsed_time_s"],
        "sum_of_face_active_durations_s": ledger[
            "sum_of_face_active_durations_s"],
        "clock_combination_rule": ledger["clock_combination_rule"],
        "face_physical_event_counts": ledger["face_physical_event_counts"],
        "total_nonnegative_physical_event_count": ledger[
            "total_nonnegative_physical_event_count"],
        "signed_material_exchange_count": ledger[
            "signed_material_exchange_count"],
        "independent_trace_exchange_count": ledger[
            "independent_trace_exchange_count"],
        "complete_energy_change_J_m3_cells": ledger[
            "complete_energy_change_J_m3_cells"],
        "heat_plus_complete_energy_residual_J_m3_cells": ledger[
            "heat_plus_complete_energy_residual_J_m3_cells"],
        "accepted_event_count": int(result.subcell_geometry.accepted_event_count),
    }


def overshoot_record():
    state, data = fixture(32)
    kinetics = replace(
        intrinsic_kinetics(), chemical_potential_J_per_defect=6e-21)
    result, ledger = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": 5e-8,
                "search_partial_displacement": True,
                "partial_displacement_levels": 16},
        data[4], data[5], data[1], data[6], data[7], kinetics, 1.0)
    if not ledger["accepted"] or result is state:
        raise RuntimeError("V51 finite-proposal search did not advance")
    return {
        "capability_control_chemical_potential_J_per_defect": 6e-21,
        "selected_displacement_m": ledger["physical_displacement_m"],
        "selected_complete_energy_change_J_m3_cells": ledger[
            "complete_energy_change_J_m3_cells"],
        **ledger["same_state_partial_displacement_search"],
    }


def translation_record():
    state, data = fixture(64)
    displacement = np.asarray((.37*data[9], -.23*data[9]))
    before = energy_terms_J(state, data)
    proposal = propose_subcell_translation(
        state.subcell_geometry, state.density, state.reservoir_alignment,
        state.common, data[4], displacement)
    moved = synchronize_common(V24MechanicalWallState(
        proposal[3], proposal[1], proposal[2], None, proposal[0]), data[5])
    after = energy_terms_J(moved, data)
    return {"displacement_m": displacement.tolist(),
            "sampled_complete_energy_change_J": after["total"]-before["total"],
            "no_fitted_mesh_potential_subtracted": True}


def main():
    compatibility = compatibility_rows()
    forces = force_records()
    payload = {
        "schema": "asb-drx/v51/event-measure-commuting-geometry/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "compatibility": compatibility,
        "force_and_secant": forces,
        "event_measure": event_measure_record(),
        "shared_clock_two_faces": shared_clock_record(),
        "finite_proposal_overshoot": overshoot_record(),
        "noninteger_translation": translation_record(),
        "energy_map_retained": all(row(n)[
            "ordered_gradient_increment_J"] > 0.0 for n in (32, 64, 128)),
        "scientific_compatibility_passed": all(
            item["scientific_compatibility_passed"] for item in compatibility),
        "drx_claimed": False, "lagb_claimed": False,
        "strict_asb_claimed": False,
    }
    output = Path("full_model/verification/v51_geometry_qualification.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    grids = [item["grid"] for item in compatibility]
    axes[0].semilogy(grids, [item["initial_line_surface_relative"]
                             for item in compatibility], "o-", label="initial")
    axes[0].semilogy(grids, [item["event_line_surface_relative"]
                             for item in compatibility], "s-", label="10 nm event")
    axes[0].axhline(.05, color="k", ls="--", label="5% budget")
    axes[0].set(xlabel="grid", ylabel="relative line/surface residual")
    axes[0].legend()
    local = forces["local_centered_derivatives"]
    labels = ["continuous", "10 nm secant"]+[f"local {x['shape_step_m']*1e9:g} nm"
             for x in local]
    values = [forces["continuous_ordered_gradient_force_N"],
              forces["ordered_gradient_10nm_secant_N"]]+[
                  x["ordered_gradient_local_force_N"] for x in local]
    axes[1].bar(labels, np.asarray(values)*1e8)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(ylabel="ordered-gradient force (1e-8 N)")
    fig.tight_layout()
    figure = Path("full_model/verification/v51_geometry_qualification.png")
    fig.savefig(figure, dpi=180); plt.close(fig)
    print(json.dumps({"passed": payload["scientific_compatibility_passed"],
                      "output_sha256": digest(output),
                      "figure_sha256": digest(figure)}, sort_keys=True))


if __name__ == "__main__":
    main()
