#!/usr/bin/env python3
"""Endpoint and response convergence for the matched V66 ASB candidates."""

from __future__ import annotations

from datetime import datetime, timezone
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


def _spectral_length(field: np.ndarray, length_m: float) -> float | None:
    value = np.asarray(field, dtype=float)-float(np.mean(field))
    power = np.abs(np.fft.fftn(value))**2
    k = 2.0*np.pi*np.fft.fftfreq(value.shape[0], d=length_m/value.shape[0])
    kx, ky = np.meshgrid(k, k, indexing="ij")
    magnitude = np.sqrt(kx*kx+ky*ky)
    power[0, 0] = 0.0
    total = float(np.sum(power))
    if total <= 0.0:
        return None
    centroid = float(np.sum(magnitude*power)/total)
    return 2.0*np.pi/centroid


def _endpoint(case: Path) -> dict:
    result = json.loads((case / "result.json").read_text())
    config = result["configuration"]
    state, runtime, step, load, _, _, provenance = _load_checkpoint(
        case / "checkpoint_latest.npz")
    args = SimpleNamespace(
        n=config["n"], length=config["length_m"],
        temperature=config["temperature_K"],
        shear_rate=config["shear_rate_s"], dt=config["dt_s"],
        thermal_control=config.get("thermal_control", "adiabatic"),
        bath_rate=config.get("thermal_bath_rate_s", 5e9))
    context = _physics(args)[-1]
    endpoint = make_endpoint(state, runtime, load, context)
    temperature = np.asarray(endpoint.observables["common_temperature_K"])
    power = np.maximum(np.asarray(
        endpoint.observables["instantaneous_plastic_power_W_m3"]), 0.0)
    top_temperature = temperature >= np.quantile(temperature, .9)
    top_power = power >= np.quantile(power, .9)
    response = result["result"]
    localization = spatial_localization_metrics(power, threshold_multiple=2.0)
    return {
        "case": case.name,
        "n": config["n"],
        "interface_width_m": config.get(
            "interface_width_m", 2.0*config["length_m"]/config["n"]),
        "thermal_control": config.get("thermal_control", "adiabatic"),
        "steps": step,
        "source_provenance": provenance,
        "peak_stress_Pa": response["peak_stress_Pa"],
        "post_peak_softening_fraction": response["post_peak_softening_fraction"],
        "maximum_temperature_contrast_K": response[
            "maximum_temperature_contrast_K"],
        "final_target_growth_fraction": response["final_target_growth_fraction"],
        "maximum_mechanical_relative_first_law_residual": response[
            "maximum_mechanical_relative_first_law_residual"],
        "maximum_front_relative_energy_closure": response[
            "maximum_front_relative_energy_closure"],
        "endpoint_temperature_contrast_K": float(np.ptp(temperature)),
        "endpoint_temperature_spectral_length_m": _spectral_length(
            temperature, config["length_m"]),
        "endpoint_power_maximum_to_mean": float(
            np.max(power)/max(float(np.mean(power)), 1e-300)),
        "endpoint_power_spectral_length_m": _spectral_length(
            power, config["length_m"]),
        "endpoint_top_decile_overlap": float(
            np.sum(top_temperature & top_power)/max(np.sum(top_power), 1)),
        "endpoint_power_localization": localization,
    }


def _relative(a, b) -> float:
    return abs(float(a)-float(b))/max(abs(float(b)), 1e-300)


def main() -> None:
    paths = []
    for n in (16, 24, 32, 48, 64):
        stem = ("v66_regime_T1000_r8e4" if n == 16
                else f"v66_regime_fixed_n{n}_T1000_r8e4")
        for suffix in ("", "_strong_bath"):
            path = VERIFY / f"{stem}{suffix}"
            if (path / "result.json").exists() and json.loads(
                    (path / "result.json").read_text())["decision"].get(
                        "run_complete", False):
                paths.append(path)
    cases = [_endpoint(path) for path in paths]
    indexed = {(case["n"], case["thermal_control"]): case for case in cases}
    paired = {}
    for n in sorted({case["n"] for case in cases}):
        adiabatic = indexed.get((n, "adiabatic"))
        bath = indexed.get((n, "strong_bath"))
        if adiabatic is None or bath is None:
            continue
        paired[str(n)] = {
            "temperature_contrast_ratio": (
                adiabatic["maximum_temperature_contrast_K"]
                / max(bath["maximum_temperature_contrast_K"], 1e-300)),
            "thermal_softening_increment": (
                adiabatic["post_peak_softening_fraction"]
                -bath["post_peak_softening_fraction"]),
            "peak_stress_decrement_Pa": (
                bath["peak_stress_Pa"]-adiabatic["peak_stress_Pa"]),
        }
    convergence = []
    available = sorted({case["n"] for case in cases})
    for coarse_n, fine_n in zip(available[:-1], available[1:]):
        row = {"coarse_n": coarse_n, "fine_n": fine_n, "controls": {}}
        for control in ("adiabatic", "strong_bath"):
            coarse = indexed.get((coarse_n, control)); fine = indexed.get((fine_n, control))
            if coarse is None or fine is None:
                continue
            row["controls"][control] = {
                key: _relative(coarse[key], fine[key]) for key in (
                    "peak_stress_Pa", "post_peak_softening_fraction",
                    "maximum_temperature_contrast_K",
                    "final_target_growth_fraction",
                    "endpoint_power_spectral_length_m",
                    "endpoint_temperature_spectral_length_m")
                if coarse[key] is not None and fine[key] is not None
            }
        convergence.append(row)
    history_convergence = None
    paired_grids = sorted(int(key) for key in paired)
    if len(paired_grids) >= 2:
        coarse_n, fine_n = paired_grids[-2:]
        histories = {}
        for n in (coarse_n, fine_n):
            stem = ("v66_regime_T1000_r8e4" if n == 16
                    else f"v66_regime_fixed_n{n}_T1000_r8e4")
            for control, suffix in (("adiabatic", ""),
                                    ("strong_bath", "_strong_bath")):
                rows = json.loads((VERIFY/f"{stem}{suffix}"/"history.json").read_text())
                histories[(n, control)] = np.asarray(
                    [row["mean_shear_stress_Pa"] for row in rows])
        scale = max(float(np.max(np.abs(histories[(fine_n, "adiabatic")]))),
                    1e-300)
        controls = {}
        for control in ("adiabatic", "strong_bath"):
            delta = (histories[(coarse_n, control)]
                     -histories[(fine_n, control)])
            controls[control] = {
                "maximum_absolute_difference_Pa": float(np.max(np.abs(delta))),
                "maximum_difference_over_fine_peak_stress": float(
                    np.max(np.abs(delta))/scale),
                "rms_difference_over_fine_peak_stress": float(
                    np.sqrt(np.mean(delta*delta))/scale),
            }
        causal_coarse = (histories[(coarse_n, "strong_bath")]
                         -histories[(coarse_n, "adiabatic")])
        causal_fine = (histories[(fine_n, "strong_bath")]
                       -histories[(fine_n, "adiabatic")])
        history_convergence = {
            "coarse_n": coarse_n, "fine_n": fine_n, "controls": controls,
            "terminal_thermal_stress_increment_coarse_Pa": float(causal_coarse[-1]),
            "terminal_thermal_stress_increment_fine_Pa": float(causal_fine[-1]),
            "terminal_increment_relative_difference": _relative(
                causal_coarse[-1], causal_fine[-1]),
            "maximum_causal_curve_difference_Pa": float(
                np.max(np.abs(causal_coarse-causal_fine))),
        }
    latest_pair = paired.get(str(max(map(int, paired))) if paired else "")
    payload = {
        "schema": "v66-asb-spatial-refinement-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "matched_pairs": paired,
        "rejected_refinement_protocol": {
            "directories": [
                "v66_regime_n24_T1000_r8e4",
                "v66_regime_n24_T1000_r8e4_strong_bath",
                "v66_regime_n32_T1000_r8e4",
                "v66_regime_n32_T1000_r8e4_strong_bath",
            ],
            "reason": (
                "These diagnostic runs fixed the interface at two cells, so its "
                "physical width changed with grid spacing. They identify the "
                "protocol confound but are excluded from convergence claims."),
        },
        "successive_grid_relative_differences": convergence,
        "finest_pair_stress_history_convergence": history_convergence,
        "decision": {
            "matched_thermal_effect_present_on_finest_paired_grid": bool(
                latest_pair is not None
                and latest_pair["thermal_softening_increment"] > 1e-3),
            "spatially_converged_below_5_percent": bool(
                convergence and convergence[-1]["controls"]
                and all(value < .05
                        for metrics in convergence[-1]["controls"].values()
                        for value in metrics.values())),
            "finest_adiabatic_endpoint_has_2x_mean_power_band": bool(
                indexed[(max(available), "adiabatic")][
                    "endpoint_power_localization"]["band_like"]),
            "stress_histories_converged_below_5_percent_of_peak": bool(
                history_convergence is not None and all(
                    value["maximum_difference_over_fine_peak_stress"] < .05
                    for value in history_convergence["controls"].values())),
            "strict_asb_qualified": False,
            "reason": (
                "Strict ASB remains false: although the stress histories and most "
                "spatial observables converge, the finest endpoint has no connected "
                "region above twice mean plastic power and the causal softening "
                "increment remains history-sensitive."),
        },
    }
    out = VERIFY / "v66_asb_spatial_refinement" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
