#!/usr/bin/env python3
"""Replay the first operation that creates the near-extinct ordered pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, choices=(128, 192), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    duration = .5*7.8125e-6
    context = resolved_bicrystal(
        grid=args.grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    driving = driving_at_time(args.grid, .01, "hold", 0.0, duration)
    _, audit = run_i3_cycle(
        context, context["state"], context["state"].eta.copy(), driving,
        I3Controls(mura_enabled=True, front_enabled=False,
                   trial_dt_s=duration, front_dt_s=duration,
                   mura_transport_operator="compatible_dealiased"))
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {
        "schema": "asb-drx/v44/ordered-first-generation/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source, "grid": args.grid,
        "operation": "first Mura half from analytic initial state",
        "operator_exposure_s": duration,
        "transport_operator": audit["mura"]["transport_operator"],
        "transport_nonlinear_product_rule": audit["mura"][
            "transport_nonlinear_product_rule"],
        "scalar_balance_residual_m": audit["mura"][
            "transport_maximum_scalar_balance_residual_m"],
        "alignment_balance_residual_m": audit["mura"][
            "transport_maximum_alignment_balance_residual_m"],
        "generation": audit["mura"]["ordering_generation_audit"],
        "integration": {
            key: audit["mura"][key] for key in (
                "ordering_integration_method", "ordering_stiff_dispatch",
                "ordering_complete_elapsed_time_s",
                "ordering_discarded_reaction_time_s",
                "ordering_endpoint_remainder_relative",
                "ordering_finite_time_kinetic_accuracy_certified_by_this_solve",
                "ordering_maximum_attempt_exposure",
                "ordering_solver_evaluations")},
        "interpretation": (
            "records the creation operation; no wall accuracy claim is made "
            "from the near-extinction ordered inventory"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"grid": args.grid,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "ordered_line_after_m": result["generation"][
                          "ordered_line_after_m"]}, sort_keys=True))


if __name__ == "__main__":
    main()
