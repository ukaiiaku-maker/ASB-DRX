#!/usr/bin/env python3
"""Classify matched late-horizon polycrystal RX/ASB evolution.

Endpoint-only stress softening is intentionally labeled a coarse lower-bound
diagnostic.  A positive ASB decision also requires an entire terminal atomic
chunk of physical-band persistence, thermal separation from the strong-bath
control, existing-boundary RX overlap, and every hard production audit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.production.coupled_multigrain_interval import make_endpoint


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(path: Path) -> dict:
    root = Path(path)
    result = json.loads((root / "result.json").read_text())
    state, runtime, step, load, _, config, _ = _load_checkpoint(
        root / "checkpoint_latest.npz")
    args = SimpleNamespace(
        n=config["n"], length=config["length_m"],
        temperature=config["temperature_K"],
        shear_rate=config["shear_rate_s"], dt=config["dt_s"],
        thermal_control=config["thermal_control"],
        bath_rate=config["thermal_bath_rate_s"])
    endpoint = make_endpoint(state, runtime, load, _physics(args)[-1])
    stress = np.asarray(endpoint.observables["common_stress_tensor_Pa"])
    temperature = np.asarray(endpoint.observables["common_temperature_K"])
    return {
        "directory": str(root.resolve()),
        "checkpoint_sha256": _sha(root / "checkpoint_latest.npz"),
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "step": step,
        "load": load,
        "thermal_control": config["thermal_control"],
        "thermal_bath_rate_s": config["thermal_bath_rate_s"],
        "target_steps": config["target_steps"],
        "mean_shear_stress_Pa": float(np.mean(stress[..., 0, 1])),
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_range_K": float(np.ptp(temperature)),
        "active_physical_band_intervals": result["active_band_intervals"],
        "flow_regime": result["flow_regime"],
        "final_power_localization": result["final_power_localization"],
        "owner_results": result["owner_results"],
        "decision": result["decision"],
        "hard_balance_maxima": result["hard_balance_maxima"],
    }


def _coarse_softening(cases: list[dict]) -> float:
    stress = np.abs([case["mean_shear_stress_Pa"] for case in cases])
    peak = float(np.max(stress))
    return float(max(peak - stress[-1], 0.0) / max(peak, 1e-300))


def classify(adiabatic: list[dict], bath: list[dict]) -> dict:
    if len(adiabatic) != len(bath) or len(adiabatic) < 2:
        raise ValueError("at least two matched endpoint pairs are required")
    all_cases = adiabatic + bath
    pairs = list(zip(adiabatic, bath))
    matched = all(
        ad["step"] == ba["step"] and ad["load"] == ba["load"]
        for ad, ba in pairs)
    chain = all(
        series[index]["parent_checkpoint_sha256"]
        == series[index - 1]["checkpoint_sha256"]
        for series in (adiabatic, bath)
        for index in range(1, len(series)))
    controls = all(
        ad["thermal_control"] == "adiabatic"
        and ad["thermal_bath_rate_s"] == 0.0
        and ba["thermal_control"] == "strong_bath"
        and ba["thermal_bath_rate_s"] > 0.0
        for ad, ba in pairs)
    hard = all(
        case["source_tree_clean"]
        and case["decision"]["all_hard_absorption_balances_passed"]
        and case["decision"]["mura_hard_balance_passed"]
        and case["decision"]["first_law_hard_balance_passed"]
        and case["decision"]["grain_ids_unchanged"]
        for case in all_cases)
    ad_softening = _coarse_softening(adiabatic)
    bath_softening = _coarse_softening(bath)
    terminal_ad = adiabatic[-1]
    terminal_bath = bath[-1]
    ad_persistent = bool(
        terminal_ad["decision"]["final_connected_band"]
        and terminal_ad["decision"]["band_persisted_every_interval"])
    bath_persistent = bool(
        terminal_bath["decision"]["final_connected_band"]
        and terminal_bath["decision"]["band_persisted_every_interval"])
    bath_band_suppressed = bool(
        not terminal_bath["decision"]["final_connected_band"]
        and terminal_bath["active_physical_band_intervals"]
        < .2 * terminal_bath["target_steps"])
    rx_overlap = bool(
        terminal_ad["decision"]["growing_owner_intersects_active_band"])
    thermal_separation = bool(
        terminal_ad["temperature_range_K"]
        > terminal_bath["temperature_range_K"] + 20.0)
    thermal_softening = bool(
        ad_softening >= .05 and ad_softening > bath_softening + .01)
    persistent_rx_band = bool(
        matched and chain and controls and hard and ad_persistent and rx_overlap)
    heat_dependent_asb = bool(
        persistent_rx_band and thermal_separation and thermal_softening)
    strict_heat_triggered_band = bool(ad_persistent and bath_band_suppressed)
    if heat_dependent_asb:
        classification = "QUALIFIED_POLYCRYSTAL_DRX_WITH_HEAT_DEPENDENT_ASB"
    elif persistent_rx_band:
        classification = "PERSISTENT_RX_BAND_WITHOUT_ASB_SOFTENING"
    else:
        classification = "LATE_HORIZON_POLYCRYSTAL_RESPONSE_NOT_YET_QUALIFIED"
    return {
        "schema": "v67-poly-rx-asb-long-horizon-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"adiabatic": adiabatic, "strong_bath": bath},
        "audits": {
            "checkpoint_chains_valid": chain,
            "matched_load_time_endpoints": matched,
            "heat_control_rates_valid": controls,
            "all_hard_audits_passed": hard,
        },
        "observations": {
            "adiabatic_coarse_endpoint_softening_fraction": ad_softening,
            "bath_coarse_endpoint_softening_fraction": bath_softening,
            "adiabatic_terminal_chunk_fully_banded": ad_persistent,
            "bath_terminal_chunk_fully_banded": bath_persistent,
            "bath_terminal_band_suppressed": bath_band_suppressed,
            "terminal_existing_boundary_rx_intersects_adiabatic_band": rx_overlap,
            "terminal_temperature_contrast_separation_exceeds_20K": (
                thermal_separation),
        },
        "decision": {
            "persistent_physical_band_with_existing_boundary_rx": (
                persistent_rx_band),
            "heat_dependent_polycrystal_asb_with_rx": heat_dependent_asb,
            "strict_heat_triggered_terminal_band": strict_heat_triggered_band,
            "classification": classification,
        },
        "claim_boundary": (
            "The stress-drop metric uses stored atomic-chunk endpoints and is a "
            "coarse lower-bound diagnostic. Qualification additionally requires "
            "a fully banded terminal chunk, >5% adiabatic post-peak loss, >1% "
            "softening separation from the matched bath, >20 K spatial thermal "
            "contrast separation, RX overlap, exact branch continuity, and all "
            "hard ledgers. It does not establish spontaneous grain birth or "
            "material calibration."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adiabatic", nargs="+", type=Path, required=True)
    parser.add_argument("--bath", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = classify(
        [load_case(path) for path in args.adiabatic],
        [load_case(path) for path in args.bath])
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload["observations"], **payload["decision"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
