#!/usr/bin/env python3
"""Short V23 nonlinear trajectory and homogeneous-control decision."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters,
)
from full_model.production.density_state_map import derived_density_fields
from full_model.production.extensive_wall import (
    ExtensiveWallParameters, wall_diagnostics,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)
from full_model.production.v23_coupled_wall import (
    accepted_coupled_step, initialize_coupled_state,
    mechanical_organization_diagnostics,
)


def run_case(kind, match_coefficient, steps=240, n=16):
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    spacing = 2e-7
    common = CommonWallParameters(spacing_m=spacing, wall_order_enabled=False)
    extensive = ExtensiveWallParameters(
        spacing_m=spacing, nye_match_coefficient_J_m=match_coefficient)
    state = initialize_coupled_state((n, n), systems, topologies)
    shape = state.common.mobile_plus_m2.shape
    if kind == "homogeneous":
        stress = np.full(shape, 1.2e9)
    else:
        x = np.arange(n)[:, None, None]
        stress = np.broadcast_to(
            1.2e9+8e8*np.cos(2*np.pi*x/n), shape).copy()
    driving = CommonWallDriving(resolved_stress_Pa=stress)
    initial = state
    elapsed = 0.0; minimum_transport_scale = 1.0
    for _ in range(steps):
        before = state
        state, ledger, scales = accepted_coupled_step(
            state, driving, systems, topologies, common, extensive, 1e-9)
        elapsed += scales["accepted_dt_s"]
        minimum_transport_scale = min(minimum_transport_scale,
                                      scales["transport_scale"])
    route = mechanical_organization_diagnostics(
        initial, state, ledger, spacing)
    density = derived_density_fields(state.density, topologies)
    wall = wall_diagnostics(
        state.density, systems, topologies, state.common.orientation_rad,
        ledger["target_nye_m1"])
    result = {
        "case": kind,
        "nye_match_coefficient_J_m": match_coefficient,
        "steps": steps,
        "elapsed_time_s": elapsed,
        "minimum_transport_accept_scale": minimum_transport_scale,
        "maximum_ordering_local_scale_last_step": scales["ordering_scale"],
        "maximum_slip": float(np.max(np.abs(state.common.slip))),
        "mean_absolute_slip": float(np.mean(np.abs(state.common.slip))),
        "rss_range_Pa": [float(np.min(route["rss_Pa"])),
                         float(np.max(route["rss_Pa"]))],
        "maximum_slip_gradient_m1": float(np.max(route["slip_gradient_m1"])),
        "maximum_curl_beta_nye_m1": float(np.max(route["nye_norm_m1"])),
        "maximum_frank_bilby_target_m1": float(np.max(np.linalg.norm(
            route["frank_bilby_target_m1"], axis=(-2, -1)))),
        "orientation_span_deg": float(np.ptp(state.common.orientation_rad)*180/np.pi),
        "maximum_ordered_density_m2": float(np.max(
            density["rho_wall_ordered_m2"])),
        "maximum_order_fraction": float(np.max(density["q_wall_diagnostic"])),
        "maximum_polarization": float(np.max(wall["polarization"])),
        "minimum_relative_nye_mismatch": float(np.min(
            wall["relative_nye_mismatch"])),
        "physical_wall_cell_count": int(np.count_nonzero(
            wall["physical_wall_mask"])),
    }
    fields = {"ordered": density["rho_wall_ordered_m2"],
              "q": density["q_wall_diagnostic"],
              "orientation": state.common.orientation_rad,
              "nye": route["nye_norm_m1"],
              "target": np.linalg.norm(route["frank_bilby_target_m1"], axis=(-2, -1)),
              "rss": np.max(np.abs(route["rss_Pa"]), axis=2)}
    return result, fields


def main():
    cases = []
    plot_fields = None
    for kind, coefficient in (
        ("homogeneous", 2e-5),
        ("heterogeneous", 2e-5),
        ("heterogeneous", 6e-5),
        ("heterogeneous", 2e-4),
    ):
        result, fields = run_case(kind, coefficient)
        cases.append(result)
        if kind == "heterogeneous" and coefficient == 6e-5:
            plot_fields = fields
    homogeneous = cases[0]
    heterogeneous = cases[2]
    comparison = {
        key: heterogeneous[key]-homogeneous[key]
        for key in ("maximum_slip_gradient_m1", "maximum_curl_beta_nye_m1",
                    "maximum_frank_bilby_target_m1", "orientation_span_deg",
                    "maximum_ordered_density_m2")}
    any_wall = any(case["physical_wall_cell_count"] > 0 for case in cases)
    result = {
        "schema": "v23-local-nonlinear-trajectory-1",
        "cases": cases,
        "heterogeneous_minus_homogeneous": comparison,
        "fixture_passed": True,
        "scientific_gate_passed": False,
        "classification": ("LOCAL_PHYSICAL_WALL_CANDIDATE_REQUIRES_GRID_CONFIRMATION"
                           if any_wall else
                           "UNSEEDED_EXTENSIVE_ORDERING_PRESENT_NO_PHYSICAL_WALL"),
        "hpc_wall_campaign_authorized": False,
        "reason": ("local candidate has not passed grid/restart controls" if any_wall else
                   "no cell jointly meets extensive density, polarization, Nye match, and orientation jump")
    }
    result["source_sha256"] = {
        name: hashlib.sha256((ROOT/"full_model"/"production"/name).read_bytes()).hexdigest()
        for name in ("common_tensorial_wall.py", "density_state_map.py",
                     "extensive_wall.py", "v23_coupled_wall.py")}
    output = ROOT/"full_model"/"verification"/"v23_local_nonlinear_trajectory.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    if plot_fields is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for axis, (name, field) in zip(axes.flat, plot_fields.items()):
            image = axis.imshow(field.T, origin="lower", cmap="viridis")
            axis.set_title(name); fig.colorbar(image, ax=axis, shrink=.8)
        fig.tight_layout()
        fig.savefig(ROOT/"full_model"/"verification"/"v23_local_nonlinear_fields.png",
                    dpi=150)
        plt.close(fig)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
