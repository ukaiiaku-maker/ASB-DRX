#!/usr/bin/env python3
"""Run and serialize the substantive Bertin BCC Gate A verification."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from asb_drx.bertin_bcc import (
    BertinBCCParameters,
    BertinBCCState,
    advance_bertin_bcc,
    cubic_family_angle_deg,
    evaluate_bertin_bcc,
    initial_orientation,
)


INITIAL_DENSITIES = np.asarray([4.8, 4.7, 4.9, 4.75]) * 1.0e14


def state(axis, perturbation_deg=0.1):
    return BertinBCCState(
        np.eye(3),
        initial_orientation(axis, perturbation_axis_lab=(1.0, 0.0, 0.0), perturbation_deg=perturbation_deg),
        INITIAL_DENSITIES,
    )


def simulate(label, axis, rate, target, family, parameters, increment=0.001, perturbation_deg=0.1):
    initial = state(axis, perturbation_deg)
    initial_response = evaluate_bertin_bcc(initial, 300.0, parameters)
    final, history = advance_bertin_bcc(
        initial, rate, target, increment, 300.0, parameters, sample_stride=25
    )
    response = evaluate_bertin_bcc(final, 300.0, parameters)
    return {
        "label": label,
        "loading_axis_hkl": list(axis),
        "loading": "tension" if rate > 0.0 else "compression",
        "target_strain": target,
        "attractor_family": family,
        "initial_attractor_angle_deg": cubic_family_angle_deg(initial_response.loading_axis_crystal, family),
        "final_attractor_angle_deg": cubic_family_angle_deg(response.loading_axis_crystal, family),
        "final_axial_stress_Pa": float(response.cauchy_stress_Pa[2, 2]),
        "final_densities_m2": final.densities_m2.tolist(),
        "final_total_density_m2": float(np.sum(final.densities_m2)),
        "history": history,
    }, final, response


def main(output: Path) -> None:
    parameters = BertinBCCParameters()
    cases = []
    for args in (
        ("001_compression_stable", (0, 0, 1), -2.0e8, -1.0, "001", parameters),
        ("111_compression_stable", (1, 1, 1), -2.0e8, -1.0, "111", parameters),
        ("101_tension_stable", (1, 0, 1), 2.0e8, 1.0, "101", parameters),
        ("419_compression_to_111", (4, 1, 9), -2.0e8, -1.0, "111", parameters),
        ("419_tension_to_101", (4, 1, 9), 2.0e8, 1.0, "101", parameters),
    ):
        case, _, _ = simulate(*args, perturbation_deg=1.0 if args[0].startswith("419") else 0.1)
        cases.append(case)

    full_case, full_state, full_response = simulate(
        "111_tension_full", (1, 1, 1), 2.0e8, 1.0, "101", parameters, perturbation_deg=1.0
    )
    no_spin_case, no_spin_state, no_spin_response = simulate(
        "111_tension_no_plastic_spin",
        (1, 1, 1),
        2.0e8,
        1.0,
        "101",
        replace(parameters, plastic_spin_scale=0.0),
        perturbation_deg=1.0,
    )
    no_relax_case, no_relax_state, no_relax_response = simulate(
        "111_tension_no_inactive_relaxation",
        (1, 1, 1),
        2.0e8,
        1.0,
        "101",
        replace(parameters, inactive_relaxation_scale=0.0),
        perturbation_deg=1.0,
    )
    cases.extend((full_case, no_spin_case, no_relax_case))

    stable_checks = [case["final_attractor_angle_deg"] < 1.0 for case in cases[:3]]
    checks = {
        "published_stable_orientations_remain_within_one_degree": all(stable_checks),
        "419_compression_moves_over_ten_degrees_toward_111": cases[3]["final_attractor_angle_deg"] < cases[3]["initial_attractor_angle_deg"] - 10.0,
        "419_tension_moves_over_five_degrees_toward_101": cases[4]["final_attractor_angle_deg"] < cases[4]["initial_attractor_angle_deg"] - 5.0,
        "111_tension_rotation_is_plastic_spin_dependent": full_case["final_attractor_angle_deg"] < no_spin_case["final_attractor_angle_deg"] - 5.0,
        "inactive_family_removal_is_relaxation_dependent": full_state.densities_m2[0] < no_relax_state.densities_m2[0] / 5.0,
        "orientation_changes_axial_stress": abs(full_response.cauchy_stress_Pa[2, 2] - no_spin_response.cauchy_stress_Pa[2, 2]) > 10.0e6,
        "density_relaxation_changes_axial_stress": abs(full_response.cauchy_stress_Pa[2, 2] - no_relax_response.cauchy_stress_Pa[2, 2]) > 5.0e6,
    }
    checks = {name: bool(value) for name, value in checks.items()}

    coarse, _, coarse_response = simulate(
        "419_compression_refinement_coarse", (4, 1, 9), -2.0e8, -0.5, "111", parameters, increment=0.001, perturbation_deg=1.0
    )
    fine, _, fine_response = simulate(
        "419_compression_refinement_fine", (4, 1, 9), -2.0e8, -0.5, "111", parameters, increment=0.0005, perturbation_deg=1.0
    )
    refinement = {
        "stress_relative_change": float(abs(coarse["final_axial_stress_Pa"] - fine["final_axial_stress_Pa"]) / abs(fine["final_axial_stress_Pa"])),
        "density_relative_change": float(abs(coarse["final_total_density_m2"] - fine["final_total_density_m2"]) / fine["final_total_density_m2"]),
        "attractor_angle_relative_change": float(abs(coarse["final_attractor_angle_deg"] - fine["final_attractor_angle_deg"]) / max(fine["final_attractor_angle_deg"], 1.0)),
    }
    checks["final_timestep_refinement_below_five_percent"] = max(refinement.values()) < 0.05
    gate_passed = all(checks.values())
    if not gate_passed:
        raise RuntimeError(f"Gate A checks failed: {checks}")

    result = {
        "schema": "asb-drx-gate-A-bertin-bcc/v1",
        "scientific_gate_passed": True,
        "scope": "qualitative reference verification; published Ta fit is not a production calibration",
        "source": {
            "citation": "Bertin et al., Acta Materialia 260 (2023) 119336",
            "doi": "10.1016/j.actamat.2023.119336",
            "open_manuscript": "https://www.osti.gov/servlets/purl/2005100",
        },
        "checks": checks,
        "refinement": refinement,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/gate_A_bertin_bcc.json"))
    main(parser.parse_args().output)
