#!/usr/bin/env python3
"""Qualify the V50 physical segment map against the continuous kernel."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.analysis.run_v46_geometry_representation import (
    LENGTH_M, REPRESENTATION_LENGTH_M, THICKNESS_M, energy_terms_J,
)
from full_model.production.arrhenius_kinetics import (
    ActivatedProcess, activated_rate_s, exp_floor_enthalpy_j,
)
from full_model.production.subcell_segment_geometry import (
    initialize_subcell_rectangle, propose_subcell_face_extension,
    propose_subcell_translation, subcell_face_field_derivative,
)
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, V43GeometryKinetics,
    accepted_subcell_face_transaction, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays, synchronize_common,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dimensionless_reference():
    def gprime(x):
        if abs(x) >= 1.0:
            return 0.0
        ymax = np.sqrt(1.0-x*x)
        value = quad(
            lambda y: -140.0/np.pi*x
            *(1.0-np.sqrt(x*x+y*y))**3,
            -ymax, ymax, epsabs=2e-12, epsrel=2e-12, limit=200)[0]
        return value
    integral = quad(lambda x: gprime(x)**2, -1.0, 1.0,
                    epsabs=2e-10, epsrel=2e-10, limit=300)[0]
    energy = (1e-33/THICKNESS_M*1e-8
              *integral/REPRESENTATION_LENGTH_M**3)
    return integral, energy


def fixture(n, offset_fraction=0.0, quadrature_divisor=32):
    data = build_case(n, length_m=LENGTH_M, periodic_nye_consistent=True)
    base = data[0]; dx = data[9]
    lower = np.asarray((1.2e-6+offset_fraction*dx,
                        1.2e-6-.17*dx))
    initialized = initialize_subcell_rectangle(
        base.density, base.reservoir_alignment, base.common, data[4],
        base.common.orientation_rad, lower_left_m=lower,
        upper_right_m=lower+np.asarray((.8e-6, .8e-6)), family=0,
        burgers_sign=1, spacing_m=dx, section_thickness_m=THICKNESS_M,
        representation_length_m=REPRESENTATION_LENGTH_M,
        line_quadrature_spacing_m=(
            REPRESENTATION_LENGTH_M/quadrature_divisor))
    state = synchronize_common(V24MechanicalWallState(
        initialized[3], initialized[1], initialized[2], None,
        initialized[0]), data[5])
    return state, data


def row(n, offset=0.0, quadrature_divisor=32):
    state, data = fixture(n, offset, quadrature_divisor)
    before = energy_terms_J(state, data)
    proposed = propose_subcell_face_extension(
        state.subcell_geometry, state.density, state.reservoir_alignment,
        state.common, data[4], 1e-8)
    candidate = synchronize_common(V24MechanicalWallState(
        proposed[3], proposed[1], proposed[2], None, proposed[0]), data[5])
    after = energy_terms_J(candidate, data)
    return {
        "grid": n, "spacing_m": data[9], "offset_fraction_cell": offset,
        "quadrature_divisor": quadrature_divisor,
        "ordered_gradient_increment_J": (
            after["ordered_gradient"]-before["ordered_gradient"]),
        "complete_increment_J": after["total"]-before["total"],
        "line_length_change_m": proposed[-1]["line_length_change_m"],
        "swept_area_m2": proposed[-1]["signed_swept_area_m2"],
    }


def transaction_record():
    state, data = fixture(32)
    kinetics = V43GeometryKinetics(
        ActivatedProcess("v50-physical-subcell", 1e9), enthalpy_J=0.0,
        critical_stress_Pa=1e9, chemical_species="vacancy",
        atomic_volume_m3_per_atom=1.8e-29,
        exchange_stoichiometry_defects_per_atom=1.0,
        continuum_representation_length_m=REPRESENTATION_LENGTH_M)
    args = (data[4], data[5], data[1], data[6], data[7], kinetics)
    rejected, uphill = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": 1e-8}, *args, 1e-6)
    first, accepted1 = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": -1e-8}, *args, 1e-6)
    restarted = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), data[4], data[5])
    second, accepted2 = accepted_subcell_face_transaction(
        restarted, {"proposed_displacement_m": -1e-8}, *args, 1e-6)
    return {
        "uphill_extension": {
            "classification": uphill["classification"],
            "exact_atomic_rollback": rejected is state,
            "consumed_duration_s": uphill["consumed_duration_s"],
        },
        "first_downhill_event": {key: accepted1[key] for key in (
            "classification", "physical_displacement_m",
            "complete_energy_change_J_m3_cells", "event_rate_s",
            "observed_accepted_velocity_m_s", "consumed_duration_s",
            "signed_material_exchange_count",
            "independent_geometry_exchange_count",
            "exchange_count_identity_residual",
            "heat_plus_complete_energy_residual_J_m3_cells")},
        "second_evolved_event": {
            "classification": accepted2["classification"],
            "accepted_event_count": int(
                second.subcell_geometry.accepted_event_count),
        },
        "restart_exact_before_second_event": all(np.array_equal(
            value, mechanical_checkpoint_arrays(restarted)[name])
            for name, value in mechanical_checkpoint_arrays(first).items()),
    }


def clock_records():
    subdivision = []
    for parts in (1, 2, 4, 8):
        state, data = fixture(32)
        kinetics = V43GeometryKinetics(
            ActivatedProcess("v50-physical-subcell", 1e9), enthalpy_J=0.0,
            critical_stress_Pa=1e9, chemical_species="vacancy",
            atomic_volume_m3_per_atom=1.8e-29,
            exchange_stoichiometry_defects_per_atom=1.0,
            continuum_representation_length_m=REPRESENTATION_LENGTH_M)
        args = (data[4], data[5], data[1], data[6], data[7], kinetics)
        elapsed = energy = 0.0
        for _ in range(parts):
            state, ledger = accepted_subcell_face_transaction(
                state, {"proposed_displacement_m": -1e-8/parts},
                *args, 1e-5)
            if not ledger["accepted"]:
                raise RuntimeError("subdivision qualification event rejected")
            elapsed += ledger["consumed_duration_s"]
            energy += ledger["complete_energy_change_J_m3_cells"]
        subdivision.append({
            "parts": parts, "elapsed_time_s": elapsed,
            "summed_complete_energy_change_J_m3_cells": energy,
            "endpoint_upper_x_m": float(state.subcell_geometry.upper_right_m[0]),
            "accepted_event_count": int(state.subcell_geometry.accepted_event_count),
        })

    state, data = fixture(32)
    fixed_rate = 2.5e8
    fixed_kinetics = V43GeometryKinetics(
        ActivatedProcess("v50-fixed-rate", 1e9), enthalpy_J=0.0,
        critical_stress_Pa=1e9, chemical_species="vacancy",
        atomic_volume_m3_per_atom=1.8e-29,
        exchange_stoichiometry_defects_per_atom=1.0,
        continuum_representation_length_m=REPRESENTATION_LENGTH_M,
        affinity_coupling_mode="legacy_energy_guard_only")
    _, fixed = accepted_subcell_face_transaction(
        state, {"proposed_displacement_m": -1e-8,
                "_affinity_rate_override_s": fixed_rate},
        data[4], data[5], data[1], data[6], data[7], fixed_kinetics, 1e-5)

    process = ActivatedProcess("v50-nonzero-barrier", 1e9)
    barrier_kinetics = replace(
        fixed_kinetics, process=process, enthalpy_J=1e-19)
    barrier = []
    for stress in (.25e9, 1e9, 100e9):
        enthalpy = exp_floor_enthalpy_j(
            stress, barrier_kinetics.enthalpy_J,
            barrier_kinetics.critical_stress_Pa, barrier_kinetics.exp_a,
            barrier_kinetics.exp_n, barrier_kinetics.exp_floor)
        _, ledger = accepted_subcell_face_transaction(
            state, {"proposed_displacement_m": -1e-12,
                    "resolved_stress_Pa": stress},
            data[4], data[5], data[1], data[6], data[7],
            barrier_kinetics, 1.0)
        barrier.append({
            "resolved_stress_Pa": stress,
            "activation_enthalpy_J": enthalpy,
            "independent_rate_s": activated_rate_s(process, enthalpy, 1100.0),
            "ledger_rate_s": ledger["event_rate_s"],
        })
    return {
        "fixed_rate_normalization": {
            "site_rate_s": fixed_rate,
            "event_jump_m": fixed["physical_event_jump_m"],
            "observed_velocity_m_s": fixed["observed_accepted_velocity_m_s"],
            "elapsed_time_s": fixed["consumed_duration_s"],
        },
        "segment_subdivision": subdivision,
        "nonzero_exp_floor_barrier": barrier,
        "subdivision_time_last_pair_relative_change": abs(
            subdivision[-1]["elapsed_time_s"]-subdivision[-2]["elapsed_time_s"]
        )/subdivision[-1]["elapsed_time_s"],
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
    reverse = propose_subcell_translation(
        moved.subcell_geometry, moved.density, moved.reservoir_alignment,
        moved.common, data[4], -displacement)
    restored = synchronize_common(V24MechanicalWallState(
        reverse[3], reverse[1], reverse[2], None, reverse[0]), data[5])
    return {
        "displacement_m": displacement.tolist(),
        "production_operator": proposal[-1]["operator"],
        "line_length_change_m": proposal[-1]["line_length_change_m"],
        "signed_swept_area_m2": proposal[-1]["signed_swept_area_m2"],
        "total_energy_change_J": after["total"]-before["total"],
        "reverse_density_max_abs": float(np.max(np.abs(
            restored.density.wall_ordered_plus_m2
            -state.density.wall_ordered_plus_m2))),
        "reverse_beta_max_abs": float(np.max(np.abs(
            restored.common.beta_p-state.common.beta_p))),
    }


def main():
    reference_integral, reference_energy = dimensionless_reference()
    refinement = [row(n) for n in (32, 64, 128)]
    sensitivities = [row(n, offset, q) for n in (64, 128)
                     for offset in (0.0, .23) for q in (16, 32, 64)]
    state, data = fixture(64)
    derivative = subcell_face_field_derivative(
        state.subcell_geometry, len(data[4]))
    volume = data[9]**2*THICKNESS_M
    derivative_closure = {
        "line_length_derivative": float(np.sum(
            derivative["scalar_density_derivative_m3"])*volume),
        "oriented_content_derivative_norm": float(np.linalg.norm(np.sum(
            derivative["alignment_derivative_m3"], axis=(0, 1))*volume)),
        "step_m": derivative["step_m"],
    }
    for item in refinement:
        item["gradient_relative_error_to_continuous_reference"] = abs(
            item["ordered_gradient_increment_J"]-reference_energy
            )/reference_energy
        item["ordered_gradient_force_N"] = -(
            item["ordered_gradient_increment_J"]/1e-8)
    payload = {
        "schema": "asb-drx/v50/production-subcell-qualification/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "continuous_reference": {
            "dimensionless_integral": reference_integral,
            "ordered_gradient_increment_J": reference_energy,
            "review_target_dimensionless": 7.191832866,
            "review_target_increment_J": 3.51163714e-16,
        },
        "production_refinement": refinement,
        "offset_and_quadrature_sensitivity": sensitivities,
        "shape_derivative_closure": derivative_closure,
        "component_force_comparison": {
            "continuous_ordered_gradient_force_N": -reference_energy/1e-8,
            "production_force_N_by_grid": [
                x["ordered_gradient_force_N"] for x in refinement],
            "n128_relative_error": refinement[-1][
                "gradient_relative_error_to_continuous_reference"],
        },
        "production_transaction": transaction_record(),
        "production_noninteger_translation": translation_record(),
        "physical_clock": clock_records(),
        "production_segment_map_passed": bool(
            all(x["ordered_gradient_increment_J"] > 0.0 for x in refinement)
            and refinement[-1][
                "gradient_relative_error_to_continuous_reference"] < .05),
        "drx_claimed": False, "asb_claimed": False,
    }
    output = Path("full_model/verification/v50_subcell_qualification.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([x["spacing_m"]*1e9 for x in refinement],
            [x["ordered_gradient_increment_J"]*1e16 for x in refinement],
            "o-", label="production")
    ax.axhline(reference_energy*1e16, color="k", ls="--",
               label="continuous reference")
    ax.set(xlabel="field spacing (nm)",
           ylabel="gradient increment (1e-16 J)",
           title="Physical 10 nm rectangle extension")
    ax.legend(); fig.tight_layout()
    figure = Path("full_model/verification/v50_subcell_qualification.png")
    fig.savefig(figure, dpi=180); plt.close(fig)
    print(json.dumps({"passed": payload["production_segment_map_passed"],
                      "sha256": digest(output),
                      "figure_sha256": digest(figure)}, sort_keys=True))


if __name__ == "__main__":
    main()
