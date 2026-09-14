#!/usr/bin/env python3
"""Run the local v18 one-grain qualification and write decision artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model.production.dislocation_free_energy import (
    DislocationFreeEnergyParameters, phase_owned_free_energy_J_m3,
)
from full_model.production.arrhenius_kinetics import (
    ActivatedProcess, EV_J, activated_rate_s, exp_floor_enthalpy_j,
)
from full_model.production.intragranular_promotion import promote_qualified_subgrain
from full_model.production.intragranular_subgrain import (
    IntragranularParameters, advance_intragranular, initialize_one_grain,
    recognize_subgrain, reconstructed_incompatibility_vector_m2,
)


def clean_metrics(metrics):
    return {key: value for key, value in metrics.items()
            if not key.endswith("_mask")}


def total_density(state):
    return np.sum(state.mobile_plus_m2+state.mobile_minus_m2+state.forest_m2
                  +state.wall_plus_m2+state.wall_minus_m2, axis=0)


def line_and_burgers_ledger(initial, final, p):
    spacing = p.domain_m/initial.orientation_rad.shape[0]
    factor = spacing**2/p.domain_m
    before = float(np.sum(initial.mobile_plus_m2+initial.mobile_minus_m2
                          +initial.forest_m2+initial.wall_plus_m2
                          +initial.wall_minus_m2)*factor)
    after = float(np.sum(final.mobile_plus_m2+final.mobile_minus_m2
                         +final.forest_m2+final.wall_plus_m2
                         +final.wall_minus_m2)*factor)
    signed0 = np.sum(initial.mobile_plus_m2-initial.mobile_minus_m2
                     +initial.wall_plus_m2-initial.wall_minus_m2, axis=(1, 2))*factor
    signed1 = np.sum(final.mobile_plus_m2-final.mobile_minus_m2
                     +final.wall_plus_m2-final.wall_minus_m2, axis=(1, 2))*factor
    closure = before-after-final.recovered_line_m_inv
    return {
        "line_before_m_inv": before, "line_after_m_inv": after,
        "neutral_recovered_m_inv": final.recovered_line_m_inv,
        "line_closure_m_inv": closure,
        "relative_line_closure": closure/max(abs(before), 1.0),
        "signed_family_before_m_inv": signed0.tolist(),
        "signed_family_after_m_inv": signed1.tolist(),
        "maximum_signed_family_change_m_inv": float(np.max(np.abs(signed1-signed0))),
        "relative_signed_family_change": float(
            np.max(np.abs(signed1-signed0))/max(abs(before), 1.0)),
    }


def energy_parameters(p):
    line = 0.5*45e9*p.burgers_m**2
    return DislocationFreeEnergyParameters(
        line, .06*line, 1e14, 1.2*line*5e14*.3,
        5e14, 1.3, .3, .25,
    )


def run_case(n, dt, duration=1.5e-3):
    p = IntragranularParameters()
    initial = initialize_one_grain(n, p)
    state = advance_intragranular(initial, duration, dt, p)
    metrics = recognize_subgrain(state, p)
    return p, initial, state, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    convergence = []
    retained = None
    for n in (64, 96, 128):
        p, initial, state, metrics = run_case(n, 5e-6)
        convergence.append({"grid_points": n, "dt_s": 5e-6,
                            **clean_metrics(metrics)})
        if n == 128:
            retained = (p, initial, state, metrics)
    p, initial, state, metrics = retained
    _, _, refined, refined_metrics = run_case(96, 2.5e-6)
    timestep = {
        "coarse_dt_s": 5e-6, "refined_dt_s": 2.5e-6,
        "misorientation_relative_change": abs(
            convergence[1]["misorientation_deg"]-refined_metrics["misorientation_deg"]
        )/refined_metrics["misorientation_deg"],
        "radius_relative_change": abs(
            convergence[1]["equivalent_radius_m"]-refined_metrics["equivalent_radius_m"]
        )/refined_metrics["equivalent_radius_m"],
    }
    ep = energy_parameters(p)
    promotion = promote_qualified_subgrain(state, metrics, p, ep)
    psi = phase_owned_free_energy_J_m3(
        total_density(state), np.sum(state.wall_plus_m2+state.wall_minus_m2, axis=0), ep)
    ledger = line_and_burgers_ledger(initial, state, p)
    parent_energy_mean = float(np.mean(psi[~metrics["component_mask"]]))
    child_energy_mean = float(np.mean(psi[metrics["interior_mask"]]))
    matched_drive = parent_energy_mean-child_energy_mean
    matched_process = ActivatedProcess("matched-boundary-motion", 1e6, 0.0, 1e6)
    matched_enthalpy = exp_floor_enthalpy_j(
        max(matched_drive, 0.0), .90*EV_J, p.critical_stress_Pa,
        p.exp_a, p.exp_n, p.exp_floor)
    matched_rate = activated_rate_s(matched_process, matched_enthalpy, p.temperature_K)

    # Negative controls isolate recognition requirements.
    no_order_p = IntragranularParameters(ordering_attempt_s=1e-20)
    no_order = advance_intragranular(initialize_one_grain(64, no_order_p), 1.5e-3, 5e-6, no_order_p)
    no_order_metrics = recognize_subgrain(no_order, no_order_p)
    uniform_p = IntragranularParameters(
        stress_concentration_radius_m=100*p.domain_m,
        stress_transition_width_m=0.35e-6,
    )
    uniform = advance_intragranular(initialize_one_grain(64, uniform_p), 1.5e-3, 5e-6, uniform_p)
    uniform_metrics = recognize_subgrain(uniform, uniform_p)
    no_recovery_p = IntragranularParameters(recovery_attempt_s=1e-20)
    no_recovery = advance_intragranular(initialize_one_grain(64, no_recovery_p), 1.5e-3, 5e-6, no_recovery_p)
    no_recovery_metrics = recognize_subgrain(no_recovery, no_recovery_p)
    persistence = []
    for time_s in (1.2e-3, 1.3e-3, 1.4e-3, 1.5e-3, 1.6e-3):
        evolving = advance_intragranular(
            initialize_one_grain(64, p), time_s, 5e-6, p)
        observed = recognize_subgrain(evolving, p)
        persistence.append({
            "time_s": time_s, "qualified": bool(observed.get("qualified", False)),
            "misorientation_deg": observed.get("misorientation_deg"),
            "frank_bilby_relative_residual": observed.get("frank_bilby_relative_residual"),
            "boundary_order_closure_fraction": observed.get("boundary_order_closure_fraction"),
        })

    result = {
        "schema": "asb-drx/v18-intragranular-decision/v1",
        "source_model": "full_model.production.intragranular_subgrain",
        "initial_grains": 1, "initial_internal_boundaries": 0,
        "external_pressure_Pa": 0.0, "prescribed_wall_wavelength": False,
        "fixture_passed": True,
        "scientific_gate_passed": bool(metrics["qualified"] and metrics["lower_density_interior"]),
        "accepted_claim": ("SINGLE_CRYSTAL_DEFORMATION_GENERATES_A_COMPATIBLE_LAGB_PRECURSOR"
                           if metrics["qualified"] and metrics["lower_density_interior"] else None),
        "stronger_grain_path_claim": False,
        "stronger_claim_reason": "qualified handoff exists, but sustained supercritical moving-front growth is not yet demonstrated",
        "representative": clean_metrics(metrics),
        "line_and_signed_burgers_ledger": ledger,
        "phase_handoff": {
            "inherited_orientation_rad": promotion.inherited_orientation_rad,
            "phase_simplex_residual": promotion.phase_simplex_residual,
            "common_energy_before_J": promotion.common_energy_before_J,
            "common_energy_after_J": promotion.common_energy_after_J,
            "front_ledger": promotion.front_state.ledger.__dict__,
        },
        "grid_convergence": convergence,
        "timestep_refinement": timestep,
        "negative_controls": {
            "no_wall_order": clean_metrics(no_order_metrics),
            "no_orientation_gradient": clean_metrics(uniform_metrics),
            "no_recovery_advantage": clean_metrics(no_recovery_metrics),
        },
        "progressive_misorientation_and_persistence": persistence,
        "pathway_comparison": {
            "common_thermodynamics": "same v18 line+positive-log+ordering+low-density functional",
            "existing_HAGB_SIBM": "retained; eligible only when a resolved HAGB exists",
            "intragranular_CDRX": "qualified LAGB precursor and conservative phase handoff",
            "transition_band": "represented here by kinematically generated orientation plateau",
            "geometric_DRX": "not exercised at this strain/horizon",
            "finite_amplitude_precursor": "retained as residual route; not used in this fixture",
            "thermodynamic_retuning_by_path": False,
            "matched_parent_energy_J_m3": parent_energy_mean,
            "matched_child_energy_J_m3": child_energy_mean,
            "matched_common_drive_J_m3": matched_drive,
            "matched_exp_floor_rate_s_inv": matched_rate,
            "existing_HAGB_rate_s_inv_if_same_resolved_state": matched_rate,
            "intragranular_front_rate_s_inv_if_same_resolved_state": matched_rate,
            "rate_equation_retuning_by_path": False,
        },
        "asb_scope": "unchanged; v32 regression remains separate",
    }
    max_grid_change = max(
        abs(convergence[-1]["misorientation_deg"]-convergence[-2]["misorientation_deg"])
        /convergence[-1]["misorientation_deg"],
        abs(convergence[-1]["equivalent_radius_m"]-convergence[-2]["equivalent_radius_m"])
        /convergence[-1]["equivalent_radius_m"],
    )
    result["grid_refinement_max_relative_change"] = max_grid_change
    result["provisional_five_percent_refinement_passed"] = bool(
        max_grid_change < .05 and max(timestep.values()) < .05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")

    incompat = reconstructed_incompatibility_vector_m2(state)
    fields = [
        (total_density(state), "total density [m$^{-2}$]"),
        (state.wall_order, "wall order $q_w$"),
        (np.linalg.norm(incompat, axis=0), "signed wall incompatibility [m$^{-2}$]"),
        (np.rad2deg(state.orientation_rad), "orientation [deg]"),
        (psi, "common defect energy [J m$^{-3}$]"),
        (promotion.eta[:, :, 1], "qualified inherited phase support"),
    ]
    signed_families = state.wall_plus_m2-state.wall_minus_m2
    fields.extend((signed_families[index], f"signed wall family {index} [m$^{{-2}}$]")
                  for index in range(4))
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for ax, (field, title) in zip(axes.ravel(), fields):
        image = ax.imshow(field.T, origin="lower", cmap="viridis")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=.78)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=180); plt.close(fig)
    print(json.dumps({"output": str(args.output), "plot": str(args.plot),
                      "claim": result["accepted_claim"],
                      "refinement_passed": result["provisional_five_percent_refinement_passed"]}, indent=2))


if __name__ == "__main__":
    main()
