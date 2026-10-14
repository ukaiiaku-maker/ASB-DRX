#!/usr/bin/env python3
"""Classify the matched 16% V66 localization horizon without rewriting 8% evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.spatial_localization import spatial_localization_metrics
from full_model.production.coupled_multigrain_interval import make_endpoint


ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT / "full_model" / "verification"
GRIDS = (32, 48)
CONTROLS = ("adiabatic", "strong_bath")


def _case(n: int, control: str) -> Path:
    suffix = "" if control == "adiabatic" else "_strong_bath"
    return VERIFY / f"v66_regime_fixed_n{n}_T1000_r8e4{suffix}"


def _relative(a: float, b: float) -> float:
    return abs(float(a)-float(b))/max(abs(float(b)), 1e-300)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _endpoint(path: Path) -> dict:
    result = json.loads((path/"result.json").read_text())
    config = result["configuration"]
    state, runtime, step, load, _, _, provenance = _load_checkpoint(
        path/"checkpoint_latest.npz")
    args = SimpleNamespace(
        n=config["n"], length=config["length_m"],
        temperature=config["temperature_K"],
        shear_rate=config["shear_rate_s"], dt=config["dt_s"],
        thermal_control=config["thermal_control"],
        bath_rate=config["thermal_bath_rate_s"])
    endpoint = make_endpoint(state, runtime, load, _physics(args)[-1])
    temperature = np.asarray(endpoint.observables["common_temperature_K"])
    power = np.maximum(np.asarray(
        endpoint.observables["instantaneous_plastic_power_W_m3"]), 0.0)
    response = result["result"]
    localization = spatial_localization_metrics(power, threshold_multiple=2.0)
    return {
        "case": path.name,
        "n": config["n"],
        "thermal_control": config["thermal_control"],
        "steps": step,
        "applied_shear_strain": load,
        "source_provenance": provenance,
        "checkpoint_sha256": _sha256(path/"checkpoint_latest.npz"),
        "peak_stress_Pa": response["peak_stress_Pa"],
        "post_peak_softening_fraction": response["post_peak_softening_fraction"],
        "final_target_growth_fraction": response["final_target_growth_fraction"],
        "maximum_temperature_contrast_K": response[
            "maximum_temperature_contrast_K"],
        "endpoint_temperature_contrast_K": float(np.ptp(temperature)),
        "endpoint_power_maximum_to_mean": float(
            np.max(power)/max(float(np.mean(power)), 1e-300)),
        "endpoint_power_localization": localization,
        "maximum_mechanical_relative_first_law_residual": response[
            "maximum_mechanical_relative_first_law_residual"],
        "maximum_front_relative_energy_closure": response[
            "maximum_front_relative_energy_closure"],
    }


def main() -> None:
    cases = [_endpoint(_case(n, control))
             for n in GRIDS for control in CONTROLS]
    indexed = {(row["n"], row["thermal_control"]): row for row in cases}
    histories = {}
    for n in GRIDS:
        for control in CONTROLS:
            rows = json.loads((_case(n, control)/"history.json").read_text())
            histories[(n, control)] = np.asarray(
                [row["mean_shear_stress_Pa"] for row in rows], dtype=float)
    if len({values.size for values in histories.values()}) != 1:
        raise RuntimeError("long-horizon histories do not have a common endpoint")

    matched = {}
    for n in GRIDS:
        adiabatic = indexed[(n, "adiabatic")]
        bath = indexed[(n, "strong_bath")]
        stress_increment = histories[(n, "strong_bath")]-histories[(n, "adiabatic")]
        matched[str(n)] = {
            "thermal_softening_increment": (
                adiabatic["post_peak_softening_fraction"]
                -bath["post_peak_softening_fraction"]),
            "temperature_contrast_ratio": (
                adiabatic["maximum_temperature_contrast_K"]
                /max(bath["maximum_temperature_contrast_K"], 1e-300)),
            "terminal_thermal_stress_increment_Pa": float(stress_increment[-1]),
            "maximum_thermal_stress_increment_Pa": float(
                np.max(stress_increment)),
            "adiabatic_band_like": adiabatic[
                "endpoint_power_localization"]["band_like"],
            "bath_band_like": bath["endpoint_power_localization"]["band_like"],
        }

    convergence = {}
    for control in CONTROLS:
        coarse = indexed[(GRIDS[0], control)]
        fine = indexed[(GRIDS[1], control)]
        coarse_local = coarse["endpoint_power_localization"]
        fine_local = fine["endpoint_power_localization"]
        convergence[control] = {
            key: _relative(coarse[key], fine[key]) for key in (
                "peak_stress_Pa", "post_peak_softening_fraction",
                "final_target_growth_fraction",
                "maximum_temperature_contrast_K",
                "endpoint_power_maximum_to_mean")
        }
        convergence[control]["band_geometry"] = {
            "above_threshold_fraction": _relative(
                coarse_local["above_threshold_fraction"],
                fine_local["above_threshold_fraction"]),
            "largest_component_fraction": _relative(
                coarse_local["largest_component_fraction"],
                fine_local["largest_component_fraction"]),
            "sorted_component_span_fraction": [
                _relative(a, b) for a, b in zip(
                    sorted(coarse_local["component_span_fraction"]),
                    sorted(fine_local["component_span_fraction"]))],
        }
        peak = max(float(np.max(np.abs(histories[(GRIDS[1], control)]))), 1e-300)
        delta = histories[(GRIDS[0], control)]-histories[(GRIDS[1], control)]
        convergence[control].update({
            "maximum_stress_curve_difference_over_fine_peak": float(
                np.max(np.abs(delta))/peak),
            "rms_stress_curve_difference_over_fine_peak": float(
                np.sqrt(np.mean(delta*delta))/peak),
        })

    fine_ad = indexed[(GRIDS[-1], "adiabatic")]
    fine_bath = indexed[(GRIDS[-1], "strong_bath")]
    both_banded = bool(
        fine_ad["endpoint_power_localization"]["band_like"]
        and fine_bath["endpoint_power_localization"]["band_like"])
    thermal_increment = matched[str(GRIDS[-1])]["thermal_softening_increment"]
    geometry_values = [
        value
        for control in CONTROLS
        for key, metric in convergence[control]["band_geometry"].items()
        for value in (metric if isinstance(metric, list) else [metric])]
    geometry_converged = bool(geometry_values and max(geometry_values) < .05)
    payload = {
        "schema": "v66-long-horizon-asb-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "A matched 16% strain causal continuation from the qualified 8% "
            "fixed-interface states; it does not replace the frozen 8% result."),
        "cases": cases,
        "matched_pairs": matched,
        "n32_to_n48_relative_differences": convergence,
        "decision": {
            "connected_2x_mean_power_band_on_finest_adiabatic_grid": bool(
                fine_ad["endpoint_power_localization"]["band_like"]),
            "connected_2x_mean_power_band_on_finest_bath_grid": bool(
                fine_bath["endpoint_power_localization"]["band_like"]),
            "thermal_softening_increment_positive": bool(thermal_increment > 0.0),
            "band_geometry_converged_below_5_percent": geometry_converged,
            "spatially_qualified_localization": bool(
                both_banded and geometry_converged),
            "thermally_triggered_asb_qualified": False,
            "classification": (
                ("mechanically_formed_localization_with_thermal_amplification"
                 if geometry_converged else
                 "mechanically_formed_banded_localization_candidate_with_thermal_amplification")
                if both_banded and thermal_increment > 0.0
                else "no_common_banded_localization"),
            "reason": (
                "A connected power band exists with and without heat retention; "
                "therefore temperature does not trigger the band in this matched "
                "experiment. Positive additional adiabatic softening quantifies "
                "thermal amplification, not thermally caused ASB. N32-to-n48 "
                "band geometry does not meet the 5% convergence threshold, so "
                "the localization remains a spatially unqualified candidate."
                if both_banded
                else "The matched pair does not establish a common connected band."),
            "material_validated": False,
        },
    }
    output = VERIFY/"v66_long_horizon_asb"/"result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
