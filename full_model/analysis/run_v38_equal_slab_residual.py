#!/usr/bin/env python3
"""Finite-slab attribution of the exactly equal-owner front residual."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v37_priority_a_controls import (
    compare_complete_recurrent_state,
    equalize_complete_recurrent_state,
    run_interval,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-um", type=float, nargs="+",
                        default=(3.2, 4.8, 6.4))
    parser.add_argument("--spacing-nm", type=float, default=50.0)
    parser.add_argument("--interface-width-um", type=float, default=0.4)
    args = parser.parse_args()
    rows = []
    for length_um in args.length_um:
        grid = round(length_um*1000.0/args.spacing_nm)
        context = resolved_bicrystal(
            grid=grid, length_m=length_um*1e-6,
            interface_width_m=args.interface_width_um*1e-6,
            child_line_fraction=1.0, temperature_K=1100.0)
        context["extensive_parameters"] = replace(
            context["extensive_parameters"],
            ordering_internal_substep_s=5e-13,
            ordering_internal_max_substeps=8192)
        raw = context["state"]
        state = equalize_complete_recurrent_state(context, raw)
        comparison = compare_complete_recurrent_state(state)
        _, off = run_interval(
            context, state, direction=1, mura_enabled=False,
            front_enabled=True)
        after_mura, mura_only = run_interval(
            context, state, direction=1, mura_enabled=True,
            front_enabled=False)
        _, recurrent = run_interval(
            context, state, direction=1, mura_enabled=True,
            front_enabled=True)
        rows.append({
            "grid": grid, "length_m": length_um*1e-6,
            "spacing_m": context["spacing_m"],
            "interface_width_m": context["interface_width_m"],
            "nominal_interface_separation_m": 0.5*length_um*1e-6,
            "equalized_recurrent_state": comparison,
            "mura_off": off, "mura_only": mura_only,
            "recurrent_mura_front": recurrent,
            "accepted_displacement_difference_recurrent_minus_off_m": (
                recurrent["accepted_contour_displacement_m"]
                -off["accepted_contour_displacement_m"]),
            "post_mura_state_retained_for_audit": bool(after_mura is not None),
        })
    result = {
        "schema": "asb-drx/v38/equal-owner-finite-slab/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "rows": rows,
        "interpretation_rule": (
            "decay with slab separation supports finite-interface interaction; "
            "a persistent Mura-off residual indicates representation/capillarity; "
            "a recurrent-only residual identifies supported history evolution"),
        "equal_state_projection_used": False,
        "fitted_drift_subtracted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
