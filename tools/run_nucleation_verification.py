from __future__ import annotations

import json
import math
from pathlib import Path

from asb_drx.nucleation import CylindricalNucleationParameters, evaluate_candidate


def main() -> None:
    parameters = CylindricalNucleationParameters(0.1, 1.0e8, 1.0e-9, 1.0e20)
    probability = parameters.event_probability(1000.0, 1.0e-12, 1.0e-3)
    decision = evaluate_candidate(
        parameters,
        candidate_radius_m=1.5 * parameters.critical_radius_m,
        minimum_resolved_radius_m=0.8 * parameters.critical_radius_m,
        candidate_orientation_rad=math.radians(12.0),
        parent_orientation_rad=0.0,
        minimum_misorientation_rad=math.radians(5.0),
        symmetry_order=4,
        temperature_K=1000.0,
        eligible_area_m2=1.0e-12,
        interval_s=1.0e-3,
        uniform_draw=0.5 * probability,
    )
    report = {
        "schema": "asb-drx-nucleation-decision-verification/v1",
        "scientific_disposition": (
            "generic analytical/decision fixture; prefactor and represented thickness are not calibrated"
        ),
        "critical_radius_m": parameters.critical_radius_m,
        "escape_radius_m": parameters.escape_radius_m,
        "barrier_J": parameters.barrier_J,
        "barrier_over_kBT": parameters.barrier_J / (1.380649e-23 * 1000.0),
        "event_probability": probability,
        "accepted_fixture_decision": decision.accepted,
        "accepted_fixture_reason": decision.reason,
    }
    output = Path("output/nucleation_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
