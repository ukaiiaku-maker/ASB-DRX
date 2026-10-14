#!/usr/bin/env python3
"""Record the bounded matched n64 continuation toward the 16% band horizon."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from full_model.analysis.postprocess_v66_long_horizon_asb import _endpoint


ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT/"full_model"/"verification"


def main() -> None:
    adiabatic = _endpoint(VERIFY/"v66_regime_fixed_n64_T1000_r8e4")
    bath = _endpoint(VERIFY/"v66_regime_fixed_n64_T1000_r8e4_strong_bath")
    if adiabatic["steps"] != bath["steps"]:
        raise RuntimeError("matched n64 members require a common endpoint")
    ad_local = adiabatic["endpoint_power_localization"]
    bath_local = bath["endpoint_power_localization"]
    payload = {
        "schema": "v66-n64-prefix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"adiabatic": adiabatic, "strong_bath": bath},
        "matched": {
            "thermal_softening_increment": (
                adiabatic["post_peak_softening_fraction"]
                -bath["post_peak_softening_fraction"]),
            "terminal_target_difference": (
                adiabatic["final_target_growth_fraction"]
                -bath["final_target_growth_fraction"]),
            "temperature_contrast_ratio": (
                adiabatic["maximum_temperature_contrast_K"]
                /max(bath["maximum_temperature_contrast_K"], 1e-300)),
        },
        "decision": {
            "common_endpoint_steps": adiabatic["steps"],
            "common_endpoint_strain": adiabatic["applied_shear_strain"],
            "adiabatic_connected_2x_mean_power_band": adiabatic[
                "endpoint_power_localization"]["band_like"],
            "bath_connected_2x_mean_power_band": bath[
                "endpoint_power_localization"]["band_like"],
            "classification": "NO_CONNECTED_BAND_AT_N64_PREFIX",
            "reason": (
                f"At {100.0*adiabatic['applied_shear_strain']:.1f}% strain the "
                f"largest adiabatic/bath components contain "
                f"{ad_local['largest_component_cells']}/"
                f"{bath_local['largest_component_cells']} cells and neither is "
                "band-like. The checksum-bound prefix advances the physical "
                "refinement but cannot decide 16% band geometry."),
        },
    }
    output = VERIFY/"v66_n64_prefix"/"result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
