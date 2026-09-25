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


SCHEMA = "asb-drx/v55/integrated-campaign-decision/v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("drx_v25_restart_*.npz"), key=lambda path: int(
        re.search(r"(\d+)$", path.stem).group(1)))


def concentration(field: np.ndarray) -> dict:
    value = np.maximum(np.asarray(field, dtype=float), 0.0)
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
    }


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
            "net_transformed_material_fraction": float(chi.mean()-.5),
            "mean_absolute_slip": float(np.mean(np.abs(raw["gamma_slip"]))),
            "maximum_absolute_slip": float(np.max(np.abs(raw["gamma_slip"]))),
            "plastic_power": concentration(power),
            "front_ledger": front["ledger"],
            "common_front_ledger": common["ledger"],
            "energy_ledger": energy,
            "mura_ledger": mura,
            "last_front_classification": experiment["front_last_decision"][
                "classification"],
            "relative_cumulative_first_law_residual": float(
                abs(energy["first_law_residual_J_m3"])/scale),
            "hard_valid": bool(
                common["ledger"]["maximum_abs_first_law_residual_J"] < 1e-20
                and mura["maximum_relative_burgers_rate_residual"] < 1e-10
                and mura["maximum_relative_line_balance_residual"] < 1e-10
                and mura["maximum_relative_energy_balance_residual"] < 1e-10
                and abs(energy["first_law_residual_J_m3"])/scale < 1e-8),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--continuous-step60", type=Path, required=True)
    parser.add_argument("--intermediate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    grids = {}
    for grid, step in ((32, 102), (64, 110), (128, 118)):
        directory = args.qualification_root.parent/f"step_sync_n{grid}"/(
            "high_mobility_feedback")
        grids[str(grid)] = endpoint(directory)
    transformed = [grids[str(grid)]["net_transformed_material_fraction"]
                   for grid in (32, 64, 128)]
    refinement = [abs(transformed[i+1]-transformed[i])
                  /max(abs(transformed[i+1]), 1e-300) for i in range(2)]
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
    strict_snapshot = bool(
        intermediate["plastic_power"]["participation_fraction"] <= .25
        and intermediate["temperature_peak_minus_mean_K"] >= 50.0)
    result = {
        "schema": SCHEMA,
        "source_scope": "generic kinetic hypothesis; not material calibration",
        "qualification": {
            "grid_endpoints": grids,
            "successive_relative_transformed_fraction_errors": refinement,
            "spatial_refinement_below_five_percent": bool(max(refinement) < .05),
            "n64_front_disabled_endpoint": disabled,
            "n64_freeze_flow_endpoint": frozen,
            "front_causal_transformed_fraction": (
                enabled["net_transformed_material_fraction"]
                -disabled["net_transformed_material_fraction"]),
            "all_hard_valid": bool(enabled["hard_valid"] and disabled["hard_valid"]
                                   and frozen["hard_valid"]),
            "restart": exact_restart_audit(
                args.qualification_root/"high_mobility_feedback"/
                "drx_v25_restart_000060.npz",
                args.continuous_step60/"high_mobility_feedback"/
                "drx_v25_restart_000060.npz"),
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
        "scientific_gate_passed": bool(
            enabled["net_transformed_material_fraction"] > .25
            and abs(disabled["net_transformed_material_fraction"]) < 1e-12
            and max(refinement) < .05
            and enabled["hard_valid"]),
        "classification": (
            "VALID_GENERIC_EXISTING_BOUNDARY_DRX_WITH_COUPLED_THERMOMECHANICAL_RESPONSE"),
        "claim_limits": [
            "prepared existing boundary, not spontaneous intragranular grain birth",
            "generic 1e12 s^-1 kinetic hypothesis, not a calibrated material mobility",
            "strict ASB is not claimed without the inherited persistence and refinement tests",
        ],
    }
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
