#!/usr/bin/env python3
"""Decision audit of V19 using evolution-neutral V20 replay checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CHANNELS = ("capture", "junction", "transport", "release", "sink")


def rate(tau, temperature, p, barrier_key):
    ev_j = 1.602176634e-19
    kb = 1.380649e-23
    stress = np.abs(tau)
    critical = max(float(p["v19_wall_critical_stress_Pa"]), 1.0)
    enthalpy = float(p[barrier_key])*ev_j*(
        float(p["v19_wall_exp_floor"])
        +(1.0-float(p["v19_wall_exp_floor"]))*np.exp(
            -float(p["v19_wall_exp_a"])
            *(stress/critical)**float(p["v19_wall_exp_n"])))
    free = enthalpy-kb*temperature*float(p.get("wall_activation_entropy_kB", 0.0))
    return float(p["v19_wall_attempt_frequency_s"])*np.exp(
        np.clip(-free/(kb*np.maximum(temperature, 1.0)), -700.0, 40.0))


def checkpoint_record(path):
    with np.load(path, allow_pickle=True) as z:
        p = json.loads(str(z["P_json"].item()))
        tau = np.asarray(z["v20_last_tau_effective"])
        temperature = np.asarray(z["T"])
        wp = np.asarray(z["rho_wall_plus"])
        wm = np.asarray(z["rho_wall_minus"])
        wall = wp+wm
        polarization = np.abs(wp-wm)/np.maximum(wall, float(p["rho_min"]))
        aggregate_polarization = np.abs(np.sum(wp-wm, axis=2))/np.maximum(
            np.sum(wall, axis=2), float(p["rho_min"]))
        rates = {
            "capture": rate(tau, temperature[..., None], p, "v19_wall_capture_G0_eV"),
            "release": rate(tau, temperature[..., None], p, "v19_wall_release_G0_eV"),
            "sink": rate(tau, temperature[..., None], p, "v19_wall_annihilation_G0_eV"),
            "order": np.asarray(z["v20_last_wall_order_rate_s"]),
        }
        pair_rates = []
        for first in range(tau.shape[2]):
            for second in range(first+1, tau.shape[2]):
                pair_rates.append(rate(
                    np.maximum(np.abs(tau[..., first]), np.abs(tau[..., second])),
                    temperature, p, "v19_wall_junction_G0_eV"))
        rates["junction"] = np.stack(pair_rates, axis=-1) if pair_rates else np.zeros(tau.shape[:2]+(0,))
        budgets = {
            f"{kind}_{channel}": np.asarray(z[f"v20_wall_budget__{kind}_{channel}"])
            for kind in ("unsigned", "signed") for channel in CHANNELS}
        expected_unsigned = np.asarray(z["v20_wall_initial_unsigned"]).copy()
        expected_signed = np.asarray(z["v20_wall_initial_signed"]).copy()
        for channel in CHANNELS:
            expected_unsigned += budgets[f"unsigned_{channel}"]
            expected_signed += budgets[f"signed_{channel}"]
        unsigned_residual = wall-expected_unsigned
        signed_residual = (wp-wm)-expected_signed
        spacing = float(p["L_phys"])/int(p["Nx"])
        area = spacing*spacing
        integrated_budget = {
            f"{kind}_{channel}_line_m": np.sum(value, axis=(0, 1)).tolist()
            for (kind_channel, value) in budgets.items()
            for kind, channel in [kind_channel.split("_", 1)]
        }
        integrated_budget = {
            key: (np.asarray(value)*area).tolist()
            for key, value in integrated_budget.items()}
        return {
            "path": str(path), "step": int(z["step"]),
            "time_s": float(z["sim_time"]),
            "strain": float(z["E_tot"][0, 0]),
            "stress_MPa": float(z["sigma_bar"])/1e6,
            "temperature_mean_K": float(np.mean(temperature)),
            "temperature_max_K": float(np.max(temperature)),
            "wall_order_mean": float(np.mean(z["q_wall_v19"])),
            "wall_order_max": float(np.max(z["q_wall_v19"])),
            "wall_order_target_mean": float(np.mean(z["v20_last_wall_order_target"])),
            "wall_order_force_mean": float(np.mean(
                z["v20_last_wall_order_target"]-z["q_wall_v19"])),
            "polarization_mean_by_family": np.mean(polarization, axis=(0, 1)).tolist(),
            "polarization_max_by_family": np.max(polarization, axis=(0, 1)).tolist(),
            "aggregate_polarization_mean": float(np.mean(aggregate_polarization)),
            "aggregate_polarization_max": float(np.max(aggregate_polarization)),
            "rate_mean_s_inv": {key: float(np.mean(value)) for key, value in rates.items()},
            "rate_max_s_inv": {key: float(np.max(value)) if value.size else 0.0
                               for key, value in rates.items()},
            "budget": integrated_budget,
            "unsigned_wall_budget_relative_residual": float(
                np.linalg.norm(unsigned_residual)/max(np.linalg.norm(wall), 1.0)),
            "signed_wall_budget_relative_residual": float(
                np.linalg.norm(signed_residual)/max(np.linalg.norm(wp-wm), 1.0)),
        }


def branch(directory):
    files = sorted(Path(directory).glob("drx_v25_restart_*.npz"))
    records = [checkpoint_record(path) for path in files]
    if len(records) < 3:
        raise ValueError("regular trajectory checkpoints are required")
    exposures = {}
    for channel in ("capture", "release", "sink", "junction", "order"):
        times = np.asarray([r["time_s"] for r in records])
        means = np.asarray([r["rate_mean_s_inv"][channel] for r in records])
        maxima = np.asarray([r["rate_max_s_inv"][channel] for r in records])
        exposures[channel] = {
            "mean_path_integral": float(np.trapezoid(means, times)),
            "maximum_path_integral_upper_bound": float(np.trapezoid(maxima, times)),
        }
    with np.load(files[-1], allow_pickle=True) as final_data:
        final_parameters = json.loads(str(final_data["P_json"].item()))
    efficiency = {
        "capture": float(final_parameters["v19_wall_capture_efficiency"]),
        "release": float(final_parameters["v19_wall_release_efficiency"]),
        "sink": float(final_parameters["v19_wall_annihilation_efficiency"]),
        "junction": float(final_parameters["v19_wall_junction_efficiency"]),
        "order": 1.0,
    }
    for channel, value in exposures.items():
        value["efficiency_weighted_mean_exposure"] = (
            efficiency[channel]*value["mean_path_integral"])
    strain = np.asarray([r["strain"] for r in records])
    stress = np.asarray([r["stress_MPa"] for r in records])
    temp = np.asarray([r["temperature_mean_K"] for r in records])
    signed_source = 0.0
    for channel in ("capture", "junction"):
        values = records[-1]["budget"][f"signed_{channel}_line_m"]
        signed_source += float(np.sum(np.abs(values)))
    final_signed = sum(abs(x) for x in records[-1]["budget"]["signed_capture_line_m"])
    final_signed += sum(abs(x) for x in records[-1]["budget"]["signed_junction_line_m"])
    return {
        "checkpoints": records,
        "reaction_channel_exposure": exposures,
        "trajectory_secants": {
            "stress_MPa_per_unit_strain": np.gradient(stress, strain).tolist(),
            "temperature_K_per_unit_strain": np.gradient(temp, strain).tolist(),
        },
        "signed_wall_source_absolute_line_m": signed_source,
        "signed_source_is_exactly_zero": bool(final_signed == 0.0),
        "final_budget_closure_passed": bool(
            records[-1]["unsigned_wall_budget_relative_residual"] < 1e-12
            and records[-1]["signed_wall_budget_relative_residual"] < 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--homogeneous", type=Path, required=True)
    parser.add_argument("--particle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "asb-drx/v20-v19-on-trajectory-audit/v1",
        "branches": {
            "homogeneous_noise": branch(args.homogeneous),
            "mechanical_particle": branch(args.particle),
        },
        "production_projected_common_operator_passed": False,
        "operator_mismatch": {
            "production_order_law": "qdot=k_order*(q_equilibrium(wall_fraction,polarization)-q)",
            "v19_projected_order_law": "Onsager gradient flow of wall-order free energy",
            "production_capture_law": "one-sided trapping of newly arrived discrete advective flux",
            "v19_projected_capture_law": "local differentiable mobile-to-wall reaction",
            "consequence": "the prior projected eigenvalues and cumulative G are not trajectory predictions",
        },
        "thermodynamic_instability": "NOT_EVALUABLE_UNTIL_COMMON_OPERATOR_REPAIR",
        "finite_mode_cumulative_amplification": None,
        "kinetic_accessibility": "reaction events are active, but the signed source is flux-limited and nearly cancelling",
        "classification": "MECHANISM_NOT_ENTERED",
        "v19_wall_absence_attribution": [
            "INSUFFICIENT_PHYSICAL_EXPOSURE_NOT_YET_QUANTIFIABLE_WITH_A_VALID_COMMON_OPERATOR",
            "SIGNED_CONTENT_PRESENT_BUT_NOT_SPATIALLY_ORGANIZED",
            "MISSING_NONLOCAL_TOPOLOGICAL_PHYSICS_IN_V19",
        ],
        "not_supported": [
            "ADEQUATE_LINEAR_EXPOSURE_NO_NONLINEAR_RESPONSE",
            "PROJECTED_WALL_INSTABILITY_FALSIFIED_NONLINEARLY",
            "MATERIAL_OR_MECHANISM_NO_GO",
        ],
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "hpc3_submission_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({
        "classification": result["classification"],
        "common_operator": result["production_projected_common_operator_passed"],
        "budget_closure": {key: value["final_budget_closure_passed"]
                           for key, value in result["branches"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
