#!/usr/bin/env python3
"""Compare the fused and historical Laplacians on one copied evolved macro."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v39_common_horizon import load_stage
from full_model.analysis.run_v48_physical_continuation import observables
from full_model.analysis.run_v49_physical_continuation import (
    advance_consistent_midpoint_segment, driving,
)
from full_model.production import extensive_wall


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_l2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))


def inventory_totals(state):
    result = {}
    density = state.mechanical.density
    for name in density.__dataclass_fields__:
        result[name] = float(np.sum(getattr(density, name), dtype=np.longdouble))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt-s", type=float, default=4.8828125e-7)
    args = parser.parse_args()
    context = resolved_bicrystal(
        grid=128, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    initial, metadata = load_stage(args.checkpoint, context)
    t0 = float(metadata["physical_time_s"])
    kwargs = dict(
        grid=128, initial_tensor_shear=float(metadata["initial_tensor_shear"]),
        protocol=metadata["protocol"], rate=float(metadata["strain_rate_s"]),
        physical_time=t0, load_origin=float(metadata["load_origin_time_s"]),
        requested_duration=args.dt_s)
    fused = extensive_wall._laplacian
    rows = {}
    states = {}
    for label, operator in (("composed_oracle",
                             extensive_wall._laplacian_composed_oracle),
                            ("fused", fused)):
        extensive_wall._laplacian = operator
        started = time.perf_counter()
        transaction = advance_consistent_midpoint_segment(
            context, initial, **kwargs)
        wall = time.perf_counter()-started
        state = transaction["candidate"]
        endpoint_drive = driving(
            128, kwargs["initial_tensor_shear"], kwargs["protocol"],
            kwargs["rate"], t0+args.dt_s-kwargs["load_origin"])
        rows[label] = {
            "wall_seconds": wall,
            "accepted_duration_s": transaction["accepted_duration_s"],
            "discarded_shortened_candidates": transaction[
                "discarded_shortened_candidates"],
            "ordering_linear_iterations": transaction["audit"]["mura"].get(
                "ordering_linear_iterations"),
            "ordering_internal_substeps": transaction["audit"]["mura"].get(
                "ordering_internal_substeps"),
            "first_law_residual_J": transaction["first_law_residual_J"],
            "observables": observables(context, state, endpoint_drive),
            "inventory_totals": inventory_totals(state),
        }
        states[label] = state
    extensive_wall._laplacian = fused
    old = states["composed_oracle"].mechanical
    new = states["fused"].mechanical
    payload = {
        "schema": "asb-drx/v52/fused-laplacian-evolved-overlap/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint),
        "physical_time_start_s": t0,
        "physical_time_end_s": t0+args.dt_s,
        "rows": rows,
        "fused_vs_composed": {
            "wall_speedup": (rows["composed_oracle"]["wall_seconds"]
                             /rows["fused"]["wall_seconds"]),
            "eta_relative_l2": relative_l2(
                states["fused"].eta, states["composed_oracle"].eta),
            "beta_p_relative_l2": relative_l2(
                new.common.beta_p, old.common.beta_p),
            "family_nye_relative_l2": relative_l2(
                new.common.family_nye_m1, old.common.family_nye_m1),
            "temperature_increment_relative_l2": relative_l2(
                new.common.temperature_K-initial.mechanical.common.temperature_K,
                old.common.temperature_K-initial.mechanical.common.temperature_K),
            "observable_differences": {
                key: float(rows["fused"]["observables"][key]
                           -rows["composed_oracle"]["observables"][key])
                for key in rows["fused"]["observables"]
            },
            "inventory_total_relative_differences": {
                key: float((rows["fused"]["inventory_totals"][key]
                            -rows["composed_oracle"]["inventory_totals"][key])
                           /max(abs(rows["composed_oracle"][
                               "inventory_totals"][key]), 1e-300))
                for key in rows["fused"]["inventory_totals"]
            },
        },
        "front_enabled": False,
        "subcell_geometry_enabled": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "output_sha256": digest(args.output),
        **payload["fused_vs_composed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
