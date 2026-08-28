from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from asb_drx.grains import (
    GrainCriteria,
    GrainRecord,
    GrainTrackerState,
    update_grain_tracker,
)


def main() -> None:
    dx_m = 1.0e-6
    criteria = GrainCriteria(
        purity_threshold=0.8,
        minimum_area_m2=9.0 * dx_m**2,
        minimum_persistence_steps=3,
        retirement_grace_steps=2,
        minimum_misorientation_rad=math.radians(5.0),
        symmetry_order=4,
    )
    state = GrainTrackerState(
        (
            GrainRecord(0, 0.0, None, "root-0", 0.0),
            GrainRecord(1, math.radians(12.0), 0, "root-0/child-1", 1.0),
            GrainRecord(2, math.radians(2.0), 0, "root-0/child-2", 1.0),
        )
    )
    fields = np.zeros((3, 12, 12), dtype=float)
    fields[0] = 1.0
    fields[0, 3:7, 4:8] = 0.0
    fields[1, 3:7, 4:8] = 1.0
    metrics = None
    for step in range(1, 4):
        state, metrics = update_grain_tracker(fields, state, float(step), dx_m, criteria)
    report = {
        "schema": "asb-drx-grain-metric-verification/v1",
        "scientific_disposition": (
            "generic classification fixture; criteria are not material parameters and "
            "promotion is not a nucleation or DRX kinetics model"
        ),
        "criteria": asdict(criteria),
        "metrics": asdict(metrics),
        "records": [asdict(record) for record in state.records],
    }
    output = Path("output/grain_metric_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
