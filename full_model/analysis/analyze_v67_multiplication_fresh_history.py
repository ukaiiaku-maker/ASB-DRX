#!/usr/bin/env python3
"""Join and classify a checkpointed fresh-origin V67 production history.

Each production segment contains its own initial response row.  This tool
verifies the checkpoint chain, removes only those duplicated seam rows, and
retains every accepted multiplication interval.  It therefore does not
reconstruct a trajectory from endpoint summaries or mix the earlier ad400
causal fork into the fresh-origin claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.postprocess_v53_asb_mechanism import field_metrics
from full_model.analysis.render_v67_poly_rx_asb_fields import frame
from full_model.analysis.spatial_localization import spatial_localization_metrics


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def _profile_participation(values: np.ndarray) -> float:
    value = np.maximum(np.asarray(values, dtype=float), 0.0)
    return float(np.sum(value)**2
                 /(value.size*max(float(np.sum(value*value)), 1e-300)))


def _endpoint_spatial_metrics(result_path: Path) -> dict:
    endpoint = frame(result_path.parent)
    # The renderer stores log10(power) only to keep one field representation
    # for plots. Reconstructing here is exact to floating-point roundoff.
    power = np.power(10.0, np.asarray(endpoint["fields"][-1], dtype=float))
    temperature = np.asarray(endpoint["fields"][-2], dtype=float)
    spacing = endpoint["extent_um"][1]*1e-6/power.shape[0]
    continuous, _ = field_metrics(power, spacing)
    correlation = float(np.corrcoef(power.ravel(), temperature.ravel())[0, 1])
    return {
        "checkpoint_sha256": endpoint["checkpoint_sha256"],
        "fixed_threshold_topology": spatial_localization_metrics(
            power, threshold_multiple=2.0),
        "continuous_power": continuous,
        "x_profile_participation_fraction": _profile_participation(
            np.mean(power, axis=1)),
        "y_profile_participation_fraction": _profile_participation(
            np.mean(power, axis=0)),
        "power_temperature_pearson_correlation": correlation,
        "power_field_semantics": (
            "instantaneous accepted work-conjugate plastic power reconstructed "
            "from the terminal common state; native grid, no resampling"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.segments) < 2:
        raise ValueError("at least two production segments are required")

    data = [load(path) for path in args.segments]
    laws = {item["configuration"]["multiplication_law"] for item in data}
    coefficients = {
        float(item["configuration"]["multiplication_coefficient_m1"])
        for item in data
    }
    if laws != {"v34_km_forest_storage_v1"} or len(coefficients) != 1:
        raise ValueError("segments do not share the repaired v34 law")
    seam_metadata = {"parent_checkpoint", "temperature_seed_peak_K"}
    reference_configuration = {
        key: value for key, value in data[0]["configuration"].items()
        if key not in seam_metadata
    }
    physics_configuration_matched = all(
        {key: value for key, value in item["configuration"].items()
         if key not in seam_metadata} == reference_configuration
        for item in data[1:])

    chain = []
    chain_valid = True
    for index, (path, item) in enumerate(zip(args.segments, data)):
        checkpoint = path.parent/"checkpoint_latest.npz"
        endpoint_hash = digest(checkpoint)
        if index:
            parent_hash = item["parent_checkpoint_sha256"]
            preceding_hash = digest(args.segments[index-1].parent/
                                    "checkpoint_latest.npz")
            chain_valid &= parent_hash == preceding_hash
        chain.append({
            "result": str(path.resolve()),
            "result_sha256": digest(path),
            "endpoint_checkpoint": str(checkpoint.resolve()),
            "endpoint_checkpoint_sha256": endpoint_hash,
            "parent_checkpoint": item["parent_checkpoint"],
            "parent_checkpoint_sha256": item["parent_checkpoint_sha256"],
            "source_commit": item["source_commit"],
            "source_tree_clean": bool(item["source_tree_clean"]),
        })

    response = list(data[0]["response_history"])
    for previous, current in zip(data, data[1:]):
        prior = previous["response_history"][-1]
        seam = current["response_history"][0]
        for key in ("load", "mean_shear_stress_Pa", "mobile_density_m2",
                    "forest_density_m2", "wall_density_m2"):
            chain_valid &= bool(np.isclose(prior[key], seam[key], rtol=2e-12,
                                           atol=1e-14))
        response.extend(current["response_history"][1:])
    multiplication = [row for item in data
                      for row in item.get("multiplication_history", [])]

    applied_load = np.asarray([row["load"] for row in response])
    stress = np.asarray([row["mean_shear_stress_Pa"] for row in response])
    flow = np.asarray([row["plastic_to_imposed_power_fraction"]
                       for row in response])
    bands = np.asarray([row["connected_physical_band"] for row in response],
                       dtype=bool)
    peak_index = int(np.argmax(np.abs(stress)))
    peak = float(abs(stress[peak_index]))
    softening = float(max(0.0, (peak-abs(stress[-1]))/max(peak, 1e-300)))
    active_loads = applied_load[bands]
    terminal_persistence = 0
    for value in bands[::-1]:
        if not value:
            break
        terminal_persistence += 1
    raw_line = float(sum(row["raw_unsigned_physical_line_increment_m"]
                         for row in multiplication))
    accepted_line = float(sum(
        row["accepted_unsigned_physical_line_increment_m"]
        for row in multiplication))
    hard_pass = all(
        item["decision"][key]
        for item in data
        for key in ("all_hard_absorption_balances_passed",
                    "first_law_hard_balance_passed",
                    "mura_hard_balance_passed"))
    failed_trials = int(sum(
        row.get("hard_balance_failed_trial_count", 0)
        for row in multiplication))
    subinterval_max = int(max(
        (row.get("hard_balance_subinterval_count", 1)
         for row in multiplication), default=1))

    owner_changes = {}
    for owner in data[0]["owner_results"]:
        initial = data[0]["owner_results"][owner]["area_before_m2"]
        final = data[-1]["owner_results"][owner]["area_after_m2"]
        owner_changes[owner] = {
            "area_before_m2": initial,
            "area_after_m2": final,
            "relative_area_change": (final-initial)/initial,
        }
    endpoint_spatial = _endpoint_spatial_metrics(args.segments[-1])

    if not hard_pass or not chain_valid or not physics_configuration_matched:
        classification = "FRESH_HISTORY_INVALID"
    elif bands[-1] and terminal_persistence >= 20 and softening > 0.0:
        classification = "FRESH_REPAIRED_MULTIPLICATION_SUPPORTS_TERMINAL_ASB"
    elif active_loads.size and not bands[-1]:
        classification = (
            "FRESH_REPAIRED_MULTIPLICATION_PRODUCES_TRANSIENT_BAND_THEN_ARRESTS")
    elif flow[-1] < 0.05:
        classification = (
            "FRESH_REPAIRED_STORAGE_MULTIPLICATION_ARRESTS_WITHOUT_PHYSICAL_BAND")
    else:
        classification = (
            "FRESH_REPAIRED_STORAGE_MULTIPLICATION_REMAINS_FLOWING_WITHOUT_"
            "QUALIFIED_TERMINAL_ASB")

    payload = {
        "schema": "v67-multiplication-fresh-history-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "fresh_origin": True,
        "checkpoint_chain_valid": bool(chain_valid),
        "physics_configuration_matched_across_segments": bool(
            physics_configuration_matched),
        "all_hard_balances_passed": bool(hard_pass),
        "multiplication_law": next(iter(laws)),
        "multiplication_coefficient_m1": next(iter(coefficients)),
        "segment_chain": chain,
        "initial_load": float(applied_load[0]),
        "final_load": float(applied_load[-1]),
        "production_intervals": len(multiplication),
        "cumulative_raw_multiplication_line_m": raw_line,
        "cumulative_accepted_multiplication_line_m": accepted_line,
        "accepted_to_raw_fraction": accepted_line/max(raw_line, 1e-300),
        "accumulated_domain_mean_absolute_family_slip": float(sum(
            np.sum(np.asarray(row["absolute_slip_increment_by_owner_family"]))
            for row in multiplication)
            /int(data[0]["configuration"]["n"])**2),
        "accumulated_domain_mean_signed_family_slip": float(sum(
            np.sum(np.asarray(row["signed_slip_increment_by_owner_family"]))
            for row in multiplication)
            /int(data[0]["configuration"]["n"])**2),
        "hard_balance_failed_trials_safely_rolled_back": failed_trials,
        "maximum_accepted_subinterval_count": subinterval_max,
        "initial_flow_exposure": float(flow[0]),
        "endpoint_flow_exposure": float(flow[-1]),
        "maximum_flow_exposure": float(np.max(flow)),
        "peak_absolute_mean_shear_stress_Pa": peak,
        "peak_stress_load": float(applied_load[peak_index]),
        "endpoint_post_peak_softening_fraction": softening,
        "first_physical_band_load": (
            float(active_loads[0]) if active_loads.size else None),
        "last_physical_band_load": (
            float(active_loads[-1]) if active_loads.size else None),
        "terminal_physical_band": bool(bands[-1]),
        "terminal_band_persistence_intervals": terminal_persistence,
        "endpoint": response[-1],
        "endpoint_spatial_localization": endpoint_spatial,
        "owner_area_changes": owner_changes,
        "claim_boundary": (
            "Fresh-origin, checkpoint-linked production history under one "
            "repaired v34 multiplication parameterization. Classification "
            "does not by itself qualify grid convergence or DRX inside ASB."),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"fresh_history.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")

    fig, axes = plt.subplots(3, 2, figsize=(10.2, 10.2), sharex=True)
    axes[0, 0].plot(applied_load, stress/1e9)
    axes[0, 1].plot(applied_load, flow)
    for axis, key, scale, label in (
            (axes[1, 0], "mobile_density_m2", 1e12, "Mobile density"),
            (axes[1, 1], "forest_density_m2", 1e12, "Forest density"),
            (axes[2, 0], "wall_density_m2", 1e12, "Wall density")):
        axis.plot(applied_load, np.asarray([row[key] for row in response])/scale)
        axis.set_ylabel(label+r" ($10^{12}$ m$^{-2}$)")
    axes[2, 1].plot(applied_load,
                    [row["temperature_range_K"] for row in response])
    axes[0, 0].set_ylabel("Mean shear stress (GPa)")
    axes[0, 1].set_ylabel("Accepted flow / imposed power")
    axes[2, 1].set_ylabel("Temperature range (K)")
    for axis in axes[2]:
        axis.set_xlabel("Applied shear strain")
    fig.tight_layout()
    fig.savefig(args.output/"fresh_history.png", dpi=180)
    plt.close(fig)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
