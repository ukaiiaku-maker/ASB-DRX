#!/usr/bin/env python3
"""Decision-grade reduction of the V55 coupled DRX/ASB trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "asb-drx/v55/integrated-campaign-decision/v2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("drx_v25_restart_*.npz"), key=lambda path: int(
        re.search(r"(\d+)$", path.stem).group(1)))


def concentration(field: np.ndarray) -> dict:
    signed = np.asarray(field, dtype=float)
    value = np.maximum(signed, 0.0)
    negative = np.minimum(signed, 0.0)
    total = float(np.sum(value))
    participation = (1.0 if total == 0.0 else
                     total**2/(value.size*float(np.sum(value*value))))
    count = max(1, int(np.ceil(.05*value.size)))
    return {
        "participation_fraction": participation,
        "top_five_percent_fraction": (
            0.0 if total == 0.0 else
            float(np.partition(value.ravel(), -count)[-count:].sum()/total)),
        "maximum_to_mean": float(value.max()/max(value.mean(), 1e-300)),
        "signed_sum": float(signed.sum()),
        "positive_sum": float(value.sum()),
        "negative_sum": float(negative.sum()),
        "negative_absolute_fraction": float(
            -negative.sum()/max(value.sum()-negative.sum(), 1e-300)),
    }


def _physical_protocol(params: dict) -> dict:
    """Historical comparison contract using fields present in V55 records."""
    keys = (
        "Nx", "Ny", "L_phys", "T0", "edot_app", "dt", "dt_strain_step",
        "moving_front_attempt_frequency_s", "sibm_clean_bicrystal_initialize",
        "sibm_clean_parent_density_m2", "sibm_clean_child_density_m2",
        "sibm_clean_misorientation_deg", "sibm_initial_bulge_radius_um",
        "sibm_seed_half_chord_um", "sibm_active_window_radius_um",
        "v19_noise_seed", "v19_particle_radius_um", "k_thermal", "cp_rho_vol",
    )
    return {key: params.get(key) for key in keys}


def comparison_contract(left: dict, right: dict) -> dict:
    time_scale = max(abs(left["physical_time_s"]),
                     abs(right["physical_time_s"]), 1e-30)
    strain_scale = max(abs(left["applied_strain"]),
                       abs(right["applied_strain"]), 1e-30)
    checks = {
        "physical_time_equal": abs(left["physical_time_s"]-
                                   right["physical_time_s"]) <= 1e-12*time_scale,
        "applied_strain_equal": abs(left["applied_strain"]-
                                    right["applied_strain"]) <= 1e-12*strain_scale,
        "physical_protocol_equal": left.get("physical_protocol") ==
                                   right.get("physical_protocol"),
        "initial_material_fraction_equal": abs(
            left["initial_child_material_fraction"]-
            right["initial_child_material_fraction"]) <= 1e-14,
    }
    return {"checks": checks, "comparable": bool(all(checks.values()))}


def outcome_neutral_decision(evidence: dict) -> dict:
    """Classify only from explicit, independently testable prerequisites."""
    required = (
        "execution_complete", "all_supporting_cases_hard_valid",
        "physical_comparability", "nontrivial_transformation",
        "selected_observable_refinement", "restart_qualified",
        "temporal_refinement_qualified", "coupled_trajectory_valid",
        "causal_controls_valid",
    )
    missing = [key for key in required if evidence.get(key) is None]
    failed = [key for key in required if evidence.get(key) is not None
              and not evidence[key]]
    passed = not missing and not failed
    if passed:
        classification = ("VALID_GENERIC_EXISTING_BOUNDARY_DRX_WITH_"
                          "COUPLED_THERMOMECHANICAL_RESPONSE")
    elif missing:
        classification = "INCOMPLETE_EVIDENCE"
    elif evidence.get("all_supporting_cases_hard_valid", False) and evidence.get(
            "nontrivial_transformation", False):
        classification = ("VALID_SUBSTANTIAL_PREPARED_BOUNDARY_MIGRATION_"
                          "CERTIFICATION_INCOMPLETE")
    elif evidence.get("execution_complete", False) and evidence.get(
            "all_supporting_cases_hard_valid", False):
        classification = "VALID_NO_NONTRIVIAL_TRANSFORMATION"
    else:
        classification = "INVALID_SUPPORTING_EVIDENCE"
    return {"scientific_gate_passed": passed, "classification": classification,
            "required_conditions": {key: evidence.get(key) for key in required},
            "missing_conditions": missing, "failed_conditions": failed}


def record(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as raw:
        params = json.loads(str(raw["P_json"].item()))
        front = json.loads(str(raw["coupled_front_metadata_json"].item()))
        common = json.loads(str(raw["common_front_metadata_json"].item()))
        energy = json.loads(str(raw["v30_asb_cumulative_json"].item()))
        mura = json.loads(str(raw["v21_balance_ledger_json"].item()))
        experiment = json.loads(str(raw["sibm_experiment_json"].item()))
        temperature = np.asarray(raw["T"], dtype=float)
        chi = np.asarray(raw["sparse_front__chi"], dtype=float)
        power = np.asarray(raw["asb_last_plastic_power_W_m3"], dtype=float)
        scale = max(abs(float(energy["external_work_J_m3"])),
                    abs(float(energy["deposited_heat_J_m3"])),
                    abs(float(energy["physical_stored_change_J_m3"])), 1.0)
        initial_fraction = float(np.asarray(
            raw["sibm_initial_child_fraction"], dtype=float).mean())
        return {
            "checkpoint": str(path.resolve()), "checkpoint_sha256": digest(path),
            "source_commit": params.get("v55_source_commit"),
            "step": int(raw["step"]), "physical_time_s": float(raw["sim_time"]),
            "applied_strain": float(raw["E_tot"][0, 0]),
            "stress_Pa": float(raw["sigma_bar"]),
            "temperature_mean_K": float(temperature.mean()),
            "temperature_peak_K": float(temperature.max()),
            "temperature_standard_deviation_K": float(temperature.std()),
            "temperature_peak_minus_mean_K": float(
                temperature.max()-temperature.mean()),
            "child_phase_fraction": float(np.mean(raw["eta"][:, :, 1])),
            "child_material_fraction": float(chi.mean()),
            "initial_child_material_fraction": initial_fraction,
            "net_transformed_material_fraction": float(
                chi.mean()-initial_fraction),
            "gross_swept_volume_m3": float(front["ledger"][
                "a_to_b_swept_volume_m3"]+front["ledger"][
                    "b_to_a_swept_volume_m3"]),
            "revisit_volume_m3": float(front["ledger"]["revisit_volume_m3"]),
            "mean_absolute_slip": float(np.mean(np.abs(raw["gamma_slip"]))),
            "maximum_absolute_slip": float(np.max(np.abs(raw["gamma_slip"]))),
            "plastic_power": concentration(power),
            "front_ledger": front["ledger"],
            "common_front_ledger": common["ledger"],
            "energy_ledger": energy,
            "mura_ledger": mura,
            "physical_protocol": _physical_protocol(params),
            "last_front_classification": experiment["front_last_decision"][
                "classification"],
            "absolute_cumulative_first_law_residual_J_m3": float(
                abs(energy["first_law_residual_J_m3"])),
            "relative_cumulative_first_law_residual": float(
                abs(energy["first_law_residual_J_m3"])/scale),
            "hard_valid": bool(
                common["ledger"]["maximum_abs_first_law_residual_J"] < 1e-20
                and mura["maximum_relative_burgers_rate_residual"] < 1e-10
                and mura["maximum_relative_line_balance_residual"] < 1e-10
                and mura["maximum_relative_energy_balance_residual"] < 1e-10
                and (abs(energy["first_law_residual_J_m3"])/scale < 1e-8
                     or abs(energy["first_law_residual_J_m3"]) < 1e-6)),
        }


def series(directory: Path) -> list[dict]:
    return [record(path) for path in checkpoint_paths(directory)]


def exact_restart_audit(segmented: Path, continuous: Path) -> dict:
    keys = ("eta", "rp", "rm", "rho_forest_plus", "rho_forest_minus",
            "rho_wall_plus", "rho_wall_minus", "gamma_slip", "v20_beta_p",
            "v20_family_nye_m1", "psi_lat", "T", "rho_GB", "sigma_bar",
            "E_tot", "sparse_front__chi", "sparse_front__processed_max")
    with np.load(segmented, allow_pickle=True) as left, np.load(
            continuous, allow_pickle=True) as right:
        fields = {key: bool(np.array_equal(left[key], right[key])) for key in keys}
        metadata = {key: bool(str(left[key].item()) == str(right[key].item()))
                    for key in ("sparse_front_metadata_json",
                                "coupled_front_metadata_json",
                                "common_front_metadata_json",
                                "v30_asb_cumulative_json",
                                "v21_balance_ledger_json")}
    return {"field_identity": fields, "ledger_identity": metadata,
            "bitwise_passed": bool(all(fields.values()) and all(metadata.values()))}


def endpoint(directory: Path) -> dict:
    paths = checkpoint_paths(directory)
    if not paths:
        raise ValueError(f"no checkpoints in {directory}")
    return record(paths[-1])


def checkpoint_at_step(directory: Path, step: int) -> dict | None:
    path = directory/f"drx_v25_restart_{step:06d}.npz"
    return record(path) if path.is_file() else None


def successive_relative(values: list[float]) -> list[float]:
    return [abs(values[i+1]-values[i])/max(abs(values[i+1]), 1e-300)
            for i in range(len(values)-1)]


def causal_pair(root: Path, feedback_name: str, control_name: str) -> dict:
    feedback = series(root/feedback_name)
    control = series(root/control_name)
    controls = {row["step"]: row for row in control}
    matched = []
    for row in feedback:
        other = controls.get(row["step"])
        if other is None:
            continue
        contract = comparison_contract(row, other)
        matched.append({
            "step": row["step"], "applied_strain": row["applied_strain"],
            "stress_feedback_Pa": row["stress_Pa"],
            "stress_control_Pa": other["stress_Pa"],
            "controlled_stress_reduction_fraction": float(
                (other["stress_Pa"]-row["stress_Pa"])
                /max(abs(other["stress_Pa"]), 1e-300)),
            "temperature_contrast_feedback_K": row[
                "temperature_peak_minus_mean_K"],
            "temperature_contrast_control_K": other[
                "temperature_peak_minus_mean_K"],
            "power_participation_feedback": row["plastic_power"][
                "participation_fraction"],
            "power_participation_control": other["plastic_power"][
                "participation_fraction"],
            "transformed_fraction_feedback": row[
                "net_transformed_material_fraction"],
            "transformed_fraction_control": other[
                "net_transformed_material_fraction"],
            "comparison_contract": contract,
        })
    return {
        "feedback_history": feedback, "freeze_flow_history": control,
        "matched_causal_history": matched,
        "latest_feedback": feedback[-1] if feedback else None,
        "latest_control": control[-1] if control else None,
        "latest_matched_comparison": matched[-1] if matched else None,
        "all_matched_comparable": bool(
            matched and all(row["comparison_contract"]["comparable"]
                            for row in matched)),
        "all_prefixes_hard_valid": bool(
            feedback and control and all(row["hard_valid"]
                                         for row in feedback+control)),
    }


def plot_history(feedback: list[dict], control: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for rows, label in ((feedback, "feedback"), (control, "freeze-flow")):
        if not rows:
            continue
        strain = 100*np.asarray([row["applied_strain"] for row in rows])
        axes[0, 0].plot(strain, [100*row["net_transformed_material_fraction"]
                                for row in rows], marker="o", ms=2, label=label)
        axes[0, 1].plot(strain, [row["stress_Pa"]/1e9 for row in rows],
                        marker="o", ms=2, label=label)
        axes[1, 0].plot(strain, [row["temperature_peak_K"] for row in rows],
                        marker="o", ms=2, label=f"{label} peak")
        axes[1, 0].plot(strain, [row["temperature_mean_K"] for row in rows],
                        ls="--", label=f"{label} mean")
        axes[1, 1].plot(strain, [row["plastic_power"]["participation_fraction"]
                                for row in rows], marker="o", ms=2, label=label)
    labels = (("Net transformed material (%)", ""), ("Stress (GPa)", ""),
              ("Temperature (K)", ""), ("Power participation", ""))
    for axis, (ylabel, _) in zip(axes.flat, labels):
        axis.set_xlabel("Applied strain (%)"); axis.set_ylabel(ylabel)
        axis.grid(alpha=.25); axis.legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)


def plot_endpoint_pair(feedback_dir: Path, control_dir: Path,
                       output: Path) -> None:
    paths = (checkpoint_paths(feedback_dir)[-1],
             checkpoint_paths(control_dir)[-1])
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=True) as raw:
            params = json.loads(str(raw["P_json"].item()))
            rows.append((
                float(params["L_phys"])*1e6,
                np.asarray(raw["T"]),
                np.asarray(raw["asb_last_plastic_power_W_m3"]),
                np.asarray(raw["eta"][:, :, 1]),
                np.asarray(raw["rho"])))
    fields = tuple(zip(*(item[1:] for item in rows)))
    limits = []
    for index, pair in enumerate(fields):
        transformed = ([np.log10(np.maximum(value, 1.0)) for value in pair]
                       if index == 3 else list(pair))
        limits.append((min(float(value.min()) for value in transformed),
                       max(float(value.max()) for value in transformed)))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    titles = ("Temperature (K)", "Plastic power (W m$^{-3}$)",
              "Child phase fraction", "log$_{10}$ total density (m$^{-2}$)")
    for i, (length, temperature, power, phase, density) in enumerate(rows):
        values = (temperature, power, phase, np.log10(np.maximum(density, 1.0)))
        for j, value in enumerate(values):
            image = axes[i, j].imshow(
                value.T, origin="lower", extent=(0, length, 0, length),
                vmin=limits[j][0], vmax=limits[j][1], cmap="viridis",
                interpolation="nearest")
            axes[i, j].set_title(titles[j]); axes[i, j].set_xlabel("x (µm)")
            axes[i, j].set_ylabel("y (µm)")
            fig.colorbar(image, ax=axes[i, j], shrink=.78)
        axes[i, 0].text(
            .02, .96, "feedback" if i == 0 else "freeze-flow",
            transform=axes[i, 0].transAxes, va="top", color="white",
            bbox={"facecolor": "black", "alpha": .55, "pad": 2})
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--continuous-step60", type=Path, required=True)
    parser.add_argument("--intermediate-root", type=Path, required=True)
    parser.add_argument("--high-rate-root", type=Path)
    parser.add_argument("--small-particle-root", type=Path)
    parser.add_argument("--low-conductivity-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--endpoint-figure", type=Path)
    args = parser.parse_args()

    grids = {}
    grid_histories = {}
    matched_time = {}
    for grid, step in ((32, 102), (64, 110), (128, 118)):
        directory = args.qualification_root.parent/f"step_sync_n{grid}"/(
            "high_mobility_feedback")
        grid_histories[str(grid)] = series(directory)
        grids[str(grid)] = endpoint(directory)
        matched_time[str(grid)] = checkpoint_at_step(directory, 100)
    transformed = [grids[str(grid)]["net_transformed_material_fraction"]
                   for grid in (32, 64, 128)]
    refinement = [abs(transformed[i+1]-transformed[i])
                  /max(abs(transformed[i+1]), 1e-300) for i in range(2)]
    matched_available = all(matched_time.values())
    matched_clocks = bool(matched_available and len({
        row["physical_time_s"] for row in matched_time.values()}) == 1 and len({
            row["applied_strain"] for row in matched_time.values()}) == 1)
    matched_transformed = ([matched_time[str(grid)][
        "net_transformed_material_fraction"] for grid in (32, 64, 128)]
        if matched_available else [])
    matched_refinement = (successive_relative(matched_transformed)
        if matched_available else [])
    matched_observables = {}
    if matched_available:
        accessors = {
            "net_transformed_material_fraction": lambda row: row[
                "net_transformed_material_fraction"],
            "gross_swept_volume_m3": lambda row: row["gross_swept_volume_m3"],
            "processed_line_m": lambda row: row["front_ledger"][
                "a_to_b_line_processed_m"]+row["front_ledger"][
                    "b_to_a_line_processed_m"],
            "stress_Pa": lambda row: row["stress_Pa"],
            "temperature_mean_K": lambda row: row["temperature_mean_K"],
            "temperature_peak_K": lambda row: row["temperature_peak_K"],
            "mean_absolute_slip": lambda row: row["mean_absolute_slip"],
        }
        for name, accessor in accessors.items():
            values = [accessor(matched_time[str(grid)])
                      for grid in (32, 64, 128)]
            errors = successive_relative(values)
            matched_observables[name] = {
                "values_n32_n64_n128": values,
                "successive_relative_errors": errors,
                "below_five_percent": bool(max(errors) < .05),
            }
    enabled = grids["64"]
    disabled = endpoint(args.qualification_root/
                        "high_mobility_front_disabled_control")
    frozen = endpoint(args.qualification_root/
                      "high_mobility_freeze_flow_control")
    feedback_rows = series(args.intermediate_root/
                           "intermediate_mobility_feedback")
    freeze_dir = (args.intermediate_root /
                  "intermediate_mobility_freeze_flow_control")
    freeze_rows = series(freeze_dir) if freeze_dir.is_dir() else []
    intermediate = feedback_rows[-1]
    disabled_rows = series(args.qualification_root/
                           "high_mobility_front_disabled_control")
    frozen_rows = series(args.qualification_root/
                         "high_mobility_freeze_flow_control")
    supporting_rows = ([row for rows in grid_histories.values() for row in rows]
                       +disabled_rows+frozen_rows+feedback_rows+freeze_rows)
    high_mobility_contracts = (
        comparison_contract(enabled, disabled),
        comparison_contract(enabled, frozen),
    )
    evidence = {
        "execution_complete": bool(supporting_rows),
        "all_supporting_cases_hard_valid": bool(
            supporting_rows and all(row["hard_valid"] for row in supporting_rows)),
        "physical_comparability": bool(
            matched_clocks and all(item["comparable"]
                                   for item in high_mobility_contracts)),
        "nontrivial_transformation": bool(
            enabled["net_transformed_material_fraction"] > .25),
        "selected_observable_refinement": bool(
            matched_observables and all(item["below_five_percent"]
                                        for item in matched_observables.values())),
        "temporal_refinement_qualified": None,
        "restart_qualified": False,
        "coupled_trajectory_valid": bool(
            feedback_rows and all(row["hard_valid"] for row in feedback_rows)),
        "causal_controls_valid": bool(
            disabled_rows and frozen_rows and all(
                row["hard_valid"] for row in disabled_rows+frozen_rows)),
    }
    restart = exact_restart_audit(
        args.qualification_root/"high_mobility_feedback"/
        "drx_v25_restart_000060.npz",
        args.continuous_step60/"high_mobility_feedback"/
        "drx_v25_restart_000060.npz")
    evidence["restart_qualified"] = restart["bitwise_passed"]
    decision = outcome_neutral_decision(evidence)
    strict_snapshot = bool(
        intermediate["plastic_power"]["participation_fraction"] <= .25
        and intermediate["temperature_peak_minus_mean_K"] >= 50.0)
    result = {
        "schema": SCHEMA,
        "source_scope": "generic kinetic hypothesis; not material calibration",
        "qualification": {
            "grid_endpoints": grids,
            "successive_relative_transformed_fraction_errors": refinement,
            "terminal_window_exit_comparison_below_five_percent": bool(
                max(refinement) < .05),
            "terminal_window_exit_is_matched_time_refinement": False,
            "matched_time_step_100": {
                "records": matched_time,
                "actual_clocks_and_loads_equal": matched_clocks,
                "successive_relative_transformed_fraction_errors":
                    matched_refinement,
                "selected_observables": matched_observables,
                "selected_observable_below_five_percent": evidence[
                    "selected_observable_refinement"],
            },
            "n64_front_disabled_endpoint": disabled,
            "n64_freeze_flow_endpoint": frozen,
            "front_causal_transformed_fraction": (
                enabled["net_transformed_material_fraction"]
                -disabled["net_transformed_material_fraction"]),
            "all_supporting_prefixes_hard_valid": evidence[
                "all_supporting_cases_hard_valid"],
            "physical_comparison_contracts": high_mobility_contracts,
            "restart": restart,
        },
        "coupled_response": {
            "feedback_history": feedback_rows,
            "freeze_flow_history": freeze_rows,
            "latest_feedback": intermediate,
            "strict_asb_snapshot_conjunction": strict_snapshot,
            "strict_asb_qualified": False,
            "strict_asb_reason": (
                "persistence and matched spatial/timestep refinement remain required; "
                "a single snapshot cannot qualify strict ASB"),
        },
        "fixture_passed": True,
        "evidence_conditions": evidence,
        **decision,
        "claim_limits": [
            "prepared existing boundary, not spontaneous intragranular grain birth",
            "generic 1e12 s^-1 kinetic hypothesis, not a calibrated material mobility",
            "window-exit endpoints are not matched-time kinetic convergence",
            "PAIR_LEFT_ACTIVE_WINDOW is a representation limit, not physical arrest",
            "strict ASB is not claimed without the inherited persistence and refinement tests",
        ],
    }
    if args.high_rate_root is not None:
        result["high_rate_discriminator"] = causal_pair(
            args.high_rate_root, "high_rate_intermediate_mobility_feedback",
            "high_rate_intermediate_mobility_freeze_flow_control")
        if args.endpoint_figure is not None:
            plot_endpoint_pair(
                args.high_rate_root/"high_rate_intermediate_mobility_feedback",
                args.high_rate_root/
                "high_rate_intermediate_mobility_freeze_flow_control",
                args.endpoint_figure)
    if args.small_particle_root is not None:
        result["small_particle_discriminator"] = causal_pair(
            args.small_particle_root, "small_particle_high_rate_feedback",
            "small_particle_high_rate_freeze_flow_control")
    if args.low_conductivity_root is not None:
        result["low_conductivity_discriminator"] = causal_pair(
            args.low_conductivity_root, "low_conductivity_high_rate_feedback",
            "low_conductivity_high_rate_freeze_flow_control")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    plot_history(feedback_rows, freeze_rows, args.figure)
    print(json.dumps({"classification": result["classification"],
                      "scientific_gate_passed": result["scientific_gate_passed"],
                      "latest_strain": intermediate["applied_strain"],
                      "strict_asb_snapshot_conjunction": strict_snapshot},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
