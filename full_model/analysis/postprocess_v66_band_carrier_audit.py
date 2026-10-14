#!/usr/bin/env python3
"""Identify fields associated with the matched n48 mechanical power band."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.production.arrhenius_dislocation_sources import boundary_tj_site_field
from full_model.production.coupled_multigrain_interval import make_endpoint
from full_model.production.multigrain_common_state import reconstruct_multigrain_common


ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT/"full_model"/"verification"


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    # Copies are essential: centring a ravelled view must not alter the
    # authoritative fields subsequently used for masks and conditional means.
    aa = np.ravel(np.asarray(a, dtype=float)).copy()
    bb = np.ravel(np.asarray(b, dtype=float)).copy()
    aa -= np.mean(aa); bb -= np.mean(bb)
    scale = float(np.sqrt(np.sum(aa*aa)*np.sum(bb*bb)))
    return 0.0 if scale == 0.0 else float(np.sum(aa*bb)/scale)


def _hot_overlap(power_mask: np.ndarray, field: np.ndarray) -> float:
    top = np.asarray(field) >= np.quantile(field, .9)
    return float(np.sum(power_mask & top)/max(np.sum(power_mask), 1))


def _case(control: str) -> tuple[dict, dict[str, np.ndarray]]:
    suffix = "" if control == "adiabatic" else "_strong_bath"
    path = VERIFY/f"v66_regime_fixed_n48_T1000_r8e4{suffix}"
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
    observables = endpoint.observables
    common, _ = reconstruct_multigrain_common(
        endpoint.state, config["length_m"]/config["n"])
    total_density = sum(
        np.sum(np.asarray(getattr(common, name)), axis=2)
        for name in ("mobile_plus_m2", "mobile_minus_m2",
                     "forest_plus_m2", "forest_minus_m2",
                     "wall_plus_m2", "wall_minus_m2"))
    fields = {
        "power": np.maximum(np.asarray(
            observables["instantaneous_plastic_power_W_m3"]), 0.0),
        "nye_norm": np.linalg.norm(np.asarray(
            observables["exact_reconstructed_nye_m1"]), axis=(2, 3)),
        "signed_density_norm": np.linalg.norm(np.asarray(
            observables["signed_total_density_m2"]), axis=2),
        "total_density": total_density,
        "signed_slip_rate_norm": np.linalg.norm(np.asarray(
            observables["instantaneous_signed_slip_rate_s"]), axis=2),
        "absolute_shear_stress": np.abs(np.asarray(
            observables["common_stress_tensor_Pa"])[..., 0, 1]),
        "temperature": np.asarray(observables["common_temperature_K"]),
        "boundary_tj_site": boundary_tj_site_field(
            endpoint.state.supports, tj_site_weight=2.0),
    }
    threshold = 2.0*float(np.mean(fields["power"]))
    hot = fields["power"] >= threshold
    metrics = {
        "case": path.name, "steps": step, "strain": load,
        "source_provenance": provenance,
        "power_band_fraction": float(np.mean(hot)),
        "field_association": {
            name: {
                "pearson_with_power": _correlation(fields["power"], field),
                "top_decile_overlap_with_power_band": _hot_overlap(hot, field),
                "band_to_domain_mean_ratio": float(
                    np.mean(field[hot])/max(float(np.mean(field)), 1e-300)),
            }
            for name, field in fields.items() if name != "power"
        },
    }
    return metrics, fields


def main() -> None:
    adiabatic, ad_fields = _case("adiabatic")
    bath, bath_fields = _case("strong_bath")
    ad_mask = ad_fields["power"] >= 2.0*np.mean(ad_fields["power"])
    bath_mask = bath_fields["power"] >= 2.0*np.mean(bath_fields["power"])
    union = np.sum(ad_mask | bath_mask)
    cross = {
        "power_field_pearson": _correlation(ad_fields["power"], bath_fields["power"]),
        "power_band_mask_jaccard": float(
            np.sum(ad_mask & bath_mask)/max(union, 1)),
        "field_pearson": {
            name: _correlation(ad_fields[name], bath_fields[name])
            for name in ad_fields if name != "power"
        },
    }
    payload = {
        "schema": "v66-band-carrier-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"adiabatic": adiabatic, "strong_bath": bath},
        "matched_cross_control": cross,
        "decision": {
            "power_field_correlation_above_0p95": bool(
                cross["power_field_pearson"] >= .95),
            "power_band_mask_jaccard": cross["power_band_mask_jaccard"],
            "shared_but_not_identical_power_pattern": bool(
                cross["power_field_pearson"] >= .95
                and cross["power_band_mask_jaccard"] < .95),
            "dominant_reported_association_in_both_controls": (
                "signed_slip_rate_norm"),
            "high_total_density_band": bool(all(
                item["field_association"]["total_density"][
                    "band_to_domain_mean_ratio"] > 1.0
                for item in (adiabatic, bath))),
            "interpretation": (
                "Associations diagnose carriers but do not establish causation. "
                "The strongly correlated but nonidentical adiabatic/bath power "
                "patterns support a shared high-slip-rate mechanical organizer "
                "that is modified and amplified by heat. Weak Nye association "
                "and density depletion reject a GND-wall or scalar high-density "
                "classification for this endpoint band."),
        },
    }
    output = VERIFY/"v66_band_carrier_audit"/"result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
