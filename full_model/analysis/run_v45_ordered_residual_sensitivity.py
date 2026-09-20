#!/usr/bin/env python3
"""Tolerance sensitivity of the first near-extinction ordered inventory."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time


def run(tolerance):
    grid = 16; duration = .5*7.8125e-6
    context = resolved_bicrystal(
        grid=grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    context["wall_parameters"] = replace(
        context["wall_parameters"],
        ordering_implicit_residual_tolerance=tolerance)
    driving = driving_at_time(grid, .01, "hold", 0.0, duration)
    _, audit = run_i3_cycle(
        context, context["state"], context["state"].eta.copy(), driving,
        I3Controls(mura_enabled=True, front_enabled=False,
                   trial_dt_s=duration, front_dt_s=duration,
                   mura_transport_operator="compatible_dealiased"))
    mura = audit["mura"]
    return {
        "requested_tolerance": tolerance,
        "ordered_line_after_m": mura["ordering_generation_audit"][
            "ordered_line_after_m"],
        "endpoint_remainder_relative": mura[
            "ordering_endpoint_remainder_relative"],
        "integration_method": mura["ordering_integration_method"],
        "dispatch": mura["ordering_stiff_dispatch"],
        "complete_elapsed_time_s": mura[
            "ordering_complete_elapsed_time_s"],
        "discarded_reaction_time_s": mura[
            "ordering_discarded_reaction_time_s"],
    }


def main():
    rows = [run(value) for value in (2e-8, 2e-10)]
    scale = max(abs(row["ordered_line_after_m"]) for row in rows)
    relative = abs(rows[0]["ordered_line_after_m"]
                   -rows[1]["ordered_line_after_m"])/max(scale, 1e-300)
    payload = {
        "schema": "asb-drx/v45/ordered-residual-sensitivity/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "operation": "first Mura half from analytic n16 state",
        "rows": rows, "tolerance_relative_change": relative,
        "spatial_observation": {
            "selected_sub8_n128_ordered_line_m": 1.2449770168873084e-05,
            "selected_sub8_n192_ordered_line_m": 3.6616043634772345e-06,
            "classification": (
                "TOLERANCE_INSENSITIVE_NEAR_EXTINCTION_SIGNAL_BUT_"
                "SPATIALLY_UNRESOLVED_NOT_A_WALL"),
        },
        "physical_wall_signal_claimed": False,
    }
    output = Path("full_model/verification/v45_ordered_residual_sensitivity.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tolerance_relative_change": relative}, sort_keys=True))


if __name__ == "__main__":
    main()
