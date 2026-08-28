from __future__ import annotations

import json
from pathlib import Path

from asb_drx.boundary import AnalyticalPeakBoundary
from asb_drx.fixtures import SingleGliderDDDParameterization


def main() -> None:
    fixture = SingleGliderDDDParameterization()
    boundary = AnalyticalPeakBoundary(fixture.law())
    temperatures = (850.0, 900.0, 950.0, 1000.0, 1050.0)
    rates = (4.5, 450.0, 45000.0)
    surface = [point.__dict__ for point in boundary.surface(temperatures, rates)]
    source_upper = [
        boundary.classify(
            fixture.source_density_range_m2[1], temperature, fixture.source_strain_rate_s_inv
        ).__dict__
        for temperature in temperatures
    ]
    report = {
        "schema": "asb-drx-analytical-peak-boundary/v1",
        "scientific_disposition": (
            "prospective arbitrary boundary for the user-authorized generic DDD fixture; "
            "not a transparent-node, ASB, DRX, or material classifier"
        ),
        "definition": "rho equals the closed-form independent EXP-floor strength peak",
        "temperature_axis_K": temperatures,
        "shear_rate_axis_s_inv": rates,
        "density_ratios_for_future_spatial_sweep": (0.5, 1.0, 2.0),
        "surface": surface,
        "ddd_source_upper_density_context": source_upper,
        "known_ddd_observation": (
            "explicit single-glider DDD strength remains monotone through 3e16 m^-2; "
            "this contextual mismatch is not fitted"
        ),
        "excluded_source_field": {
            "field": "analytical_peak_density_m2",
            "value_m2": 1.0e18,
            "reason": "hard-coded in driver rather than evaluated from governing equations",
        },
    }
    output = Path("output/analytical_boundary_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
