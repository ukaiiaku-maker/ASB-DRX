#!/usr/bin/env python3
"""Same-state front-energy comparison under a bounded elastic-mismatch change."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import (
    driving_at_time, geometric_envelope,
)
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.analysis.run_v41_front_direction_energy import compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    args = parser.parse_args()
    context = resolved_bicrystal(
        grid=args.grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    state, metadata = load_stage(args.checkpoint, context)
    controls = I3Controls(
        mura_enabled=False, front_enabled=True, trial_dt_s=1.5625e-5,
        front_dt_s=1.5625e-5, front_exp_n=1.0,
        driving_pressure_a_to_b_Pa=0.0, applied_pressure_a_to_b_Pa=0.0,
        deterministic_front_rate_law="complete_dissipation")
    cases = []
    for name, shear in (("retained_hold", .01),
                        ("prepared_zero_mean_shear", 0.0)):
        driving = driving_at_time(args.grid, shear, "hold", 0.0, 0.0)
        trials = []
        for fraction in (.0625, .03125, .015625, .0078125):
            for direction in (1, -1):
                proposal = geometric_envelope(
                    state.eta, fraction=fraction, direction=direction)
                _, audit = run_i3_cycle(context, state, proposal, driving,
                                        replace(controls))
                row = compact(
                    audit, direction, fraction, "complete_dissipation")
                # The V41 archive owns the full endpoint/component traces.
                # V42 retains only decision scalars for this matched physical
                # alternative instead of duplicating large field arrays.
                row.pop("directional_endpoints", None)
                trials.append(row)
        cases.append({
            "name": name, "mean_shear_strain": shear, "trials": trials,
            "published_trial_count": sum(row["state_published"] for row in trials),
            "downhill_trial_count": sum(
                row["actual_complete_candidate_delta_helmholtz_J"] <= 0.0
                for row in trials),
        })
    payload = {
        "schema": "asb-drx/v42/front-physical-alternative/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "stage_metadata": {key: metadata.get(key) for key in (
            "source_sha", "stage", "grid", "macro_dt_s", "interval_index",
            "completed_intervals", "physical_time_s")},
        "zero_applied_front_work": True,
        "preparation_scope": (
            "same internal checkpoint evaluated in two prescribed mean-strain "
            "states; this is a low-elastic-mismatch counterfactual, not a free "
            "reset or a work-accounted unloading trajectory"),
        "cases": cases,
        "classification": "BOUNDED_ELASTIC_MISMATCH_COUNTERFACTUAL_COMPLETED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
