#!/usr/bin/env python3
"""Separate endpoint sampling error from trajectory-level band differences."""

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
VERIFY = ROOT/"full_model"/"verification"
COMMON_N = 384


def _case(n: int, control: str) -> Path:
    suffix = "" if control == "adiabatic" else "_strong_bath"
    return VERIFY/f"v66_regime_fixed_n{n}_T1000_r8e4{suffix}"


def _periodic_bilinear(field: np.ndarray, target_n: int) -> np.ndarray:
    """Interpolate periodic cell-centred data onto a common cell-centred grid."""
    value = np.asarray(field, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("field must be square and two-dimensional")
    source_n = value.shape[0]
    coordinate = (np.arange(target_n)+0.5)*source_n/target_n-0.5
    lower = np.floor(coordinate).astype(int)
    fraction = coordinate-lower
    i0 = lower % source_n
    i1 = (lower+1) % source_n
    wx = fraction[:, None]
    wy = fraction[None, :]
    return ((1.0-wx)*(1.0-wy)*value[np.ix_(i0, i0)]
            +wx*(1.0-wy)*value[np.ix_(i1, i0)]
            +(1.0-wx)*wy*value[np.ix_(i0, i1)]
            +wx*wy*value[np.ix_(i1, i1)])


def _power(path: Path) -> np.ndarray:
    result = json.loads((path/"result.json").read_text())
    config = result["configuration"]
    state, runtime, _, load, _, _, _ = _load_checkpoint(
        path/"checkpoint_latest.npz")
    args = SimpleNamespace(
        n=config["n"], length=config["length_m"],
        temperature=config["temperature_K"],
        shear_rate=config["shear_rate_s"], dt=config["dt_s"],
        thermal_control=config["thermal_control"],
        bath_rate=config["thermal_bath_rate_s"])
    endpoint = make_endpoint(state, runtime, load, _physics(args)[-1])
    return np.asarray(endpoint.observables[
        "instantaneous_plastic_power_W_m3"], dtype=float)


def _relative(a: float, b: float) -> float:
    return abs(float(a)-float(b))/max(abs(float(b)), 1e-300)


def _geometry_difference(coarse: dict, fine: dict) -> dict:
    return {
        "above_threshold_fraction": _relative(
            coarse["above_threshold_fraction"], fine["above_threshold_fraction"]),
        "largest_component_fraction": _relative(
            coarse["largest_component_fraction"], fine["largest_component_fraction"]),
        "sorted_component_span_fraction": [
            _relative(a, b) for a, b in zip(
                sorted(coarse["component_span_fraction"]),
                sorted(fine["component_span_fraction"]))],
    }


def main() -> None:
    cases = {}
    comparisons = {}
    for control in ("adiabatic", "strong_bath"):
        metrics = {}
        for n in (32, 48):
            power = _power(_case(n, control))
            raw = spatial_localization_metrics(power, 2.0)
            common = spatial_localization_metrics(
                _periodic_bilinear(power, COMMON_N), 2.0)
            metrics[str(n)] = {"raw": raw, "common_grid": common}
        cases[control] = metrics
        comparisons[control] = {
            "raw": _geometry_difference(metrics["32"]["raw"],
                                        metrics["48"]["raw"]),
            "common_grid": _geometry_difference(
                metrics["32"]["common_grid"],
                metrics["48"]["common_grid"]),
        }

    common_values = [
        value for control in comparisons.values()
        for metric in control["common_grid"].values()
        for value in (metric if isinstance(metric, list) else [metric])]
    payload = {
        "schema": "v66-band-sampling-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "common_periodic_bilinear_grid_n": COMMON_N,
        "cases": cases,
        "n32_to_n48_relative_differences": comparisons,
        "decision": {
            "common_grid_geometry_converged_below_5_percent": bool(
                common_values and max(common_values) < .05),
            "trajectory_difference_persists_after_common_sampling": bool(
                common_values and max(common_values) >= .05),
            "reason": (
                "Common-grid interpolation removes cell-count sampling as the "
                "explanation only if all geometry differences fall below 5%. "
                "Otherwise the endpoint trajectories themselves remain spatially "
                "distinct and require physical-grid refinement."),
        },
    }
    output = VERIFY/"v66_band_sampling_audit"/"result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
