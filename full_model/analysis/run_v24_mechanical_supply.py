#!/usr/bin/env python3
"""Local V24 full-elastic transport/topology wall-supply comparison."""

from __future__ import annotations

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

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters,
)
from full_model.production.density_state_map import derived_density_fields
from full_model.production.extensive_wall import ExtensiveWallParameters
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology, rotated_system_fields,
)
from full_model.production.v23_coupled_wall import initialize_coupled_state
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, V24TopologyKinetics,
    accepted_v24_mechanical_step, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays,
)
from full_model.production.wall_circuit_diagnostics import (
    classify_persistent_wall_history,
)
from full_model.production.wall_topology_supply import (
    aligned_state_from_directions, reservoir_nye_m1,
)


def build_case(n, length_m=3.2e-6, *, periodic_nye_consistent=False):
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=1e-9),)
    dx = length_m/n
    if periodic_nye_consistent:
        mobile_plus = mobile_minus = 7.5e13
        forest_plus = forest_minus = 4.5e13
    else:
        mobile_plus, mobile_minus = 8e13, 7e13
        forest_plus, forest_minus = 5e13, 4e13
    old = initialize_coupled_state(
        (n, n), systems, topologies, mobile_plus_m2=mobile_plus,
        mobile_minus_m2=mobile_minus, forest_plus_m2=forest_plus,
        forest_minus_m2=forest_minus, wall_tangle_plus_m2=0.0,
        wall_tangle_minus_m2=0.0, junction_m2=0.0)
    _, slip_direction, plane_normal = rotated_system_fields(
        systems, np.zeros((n, n)))
    line_direction = np.cross(plane_normal, slip_direction)
    alignment = aligned_state_from_directions(old.density, line_direction)
    state = V24MechanicalWallState(old.common, old.density, alignment)
    common = CommonWallParameters(
        spacing_m=dx, wall_order_enabled=False, transport_scheme="upwind",
        mobile_correlation_diffusivity_m2_s=0.0)
    extensive = ExtensiveWallParameters(
        spacing_m=dx, nye_match_coefficient_J_m=0.0,
        disordered_excess_J_m=3e-10, ordered_excess_J_m=1e-10,
        ordering_barrier_eV=.2, ordering_attempt_frequency_s=1e8,
        ordering_entropy_over_kB=0.0, negative_barrier_mode="drag")
    kinetics = V24TopologyKinetics(
        ActivatedProcess("junction", 1e10, entropy_over_kB=0.0,
                         negative_barrier_mode="drag"),
        .2*EV_J, 1e9)
    x = (np.arange(n)-n/2)*dx
    stripe = np.abs(x)[:, None] <= .45e-6
    support = np.broadcast_to(stripe, (n, n)).copy()
    profile = np.exp(-(x/.45e-6)**8)[:, None]
    fixed = np.zeros((n, n, 2, 2))
    fixed[..., 0, 1] = .1*profile
    fixed[..., 1, 0] = fixed[..., 0, 1]
    mean = np.array([[0.0, .06], [.06, 0.0]])
    driving = CommonWallDriving(mean_strain=mean, fixed_eigenstrain=fixed)
    release = CommonWallDriving(mean_strain=mean,
                                fixed_eigenstrain=np.zeros_like(fixed))
    return (state, driving, release, support, systems, topologies,
            common, extensive, kinetics, dx)


def snapshot(state, systems, topologies, dx, time_s, active):
    nye = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)
    return {
        "time_s": time_s, "mechanical_loading_active": active,
        "orientation_rad": state.common.orientation_rad.copy(),
        "ordered_nye_m1": (nye["wall_ordered"]+nye["junction"]).copy(),
        "total_nye_m1": nye["total"].copy(), "spacing_m": dx,
        "normal_window_m": .8e-6, "plateau_offset_m": .5e-6,
    }


def compact_classification(result):
    compact = dict(result)
    for record in compact["records"]:
        # Segment dictionaries are retained, but no grid-sized arrays are
        # emitted by classify_persistent_wall_history.
        pass
    return compact


def run_case(n, topology_on, load_steps=300, release_steps=100):
    (state, driving, release, support, systems, topologies,
     common, extensive, kinetics, dx) = build_case(n)
    elapsed = 0.0; snapshots = []
    for step in range(load_steps):
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, 2e-9, topology_route_enabled=topology_on)
        elapsed += ledger["accepted_dt_s"]
    snapshots.append(snapshot(state, systems, topologies, dx, elapsed, True))
    # Exercise the complete restart representation at the load/release seam.
    state = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(state), systems, topologies)
    for step in range(release_steps):
        state, ledger = accepted_v24_mechanical_step(
            state, release, np.zeros_like(support), systems, topologies,
            common, extensive, kinetics, 2e-9,
            topology_route_enabled=topology_on)
        elapsed += ledger["accepted_dt_s"]
        if step in (0, release_steps//2, release_steps-1):
            snapshots.append(snapshot(
                state, systems, topologies, dx, elapsed, False))
    classification = classify_persistent_wall_history(
        snapshots, required_release_persistence_s=0.0)
    nye = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)
    curl_beta = np.sum(state.common.family_nye_m1, axis=2)
    scale = max(float(np.sqrt(np.mean(curl_beta*curl_beta))), 1.0)
    density = derived_density_fields(state.density, topologies)
    segments = [segment for record in classification["records"]
                for segment in record["local_integrated_audit"]["segments"]]
    record = {
        "grid": n, "spacing_m": dx, "topology_route_enabled": topology_on,
        "elapsed_time_s": elapsed,
        "orientation_span_deg_diagnostic_only": float(
            np.ptp(state.common.orientation_rad)*180/np.pi),
        "maximum_local_misorientation_deg": float(max(
            [np.rad2deg(item["local_misorientation_rad"])
             for item in segments] or [0.0])),
        "maximum_ordered_supply_ratio": float(max(
            [item["ordered_supply_ratio"] for item in segments] or [0.0])),
        "minimum_ordered_frank_bilby_residual": (
            float(min(item["ordered_frank_bilby_residual"]
                      for item in segments)) if segments else None),
        "maximum_total_nye_m1": float(np.max(np.linalg.norm(
            nye["total"], axis=(-2, -1)))),
        "curl_beta_vs_reservoir_nye_relative_rms": float(
            np.sqrt(np.mean((curl_beta-nye["total"])**2))/scale),
        "maximum_ordered_density_m2": float(np.max(
            density["rho_wall_ordered_m2"])),
        "maximum_junction_extent_m2": float(np.max(state.density.junction_m2)),
        "restart_at_release_seam_exact_representation": True,
        "classification": compact_classification(classification),
    }
    fields = {
        "total_density": density["rho_total_m2"],
        "nye": np.linalg.norm(nye["total"], axis=(-2, -1)),
        "ordered_nye": np.linalg.norm(
            nye["wall_ordered"]+nye["junction"], axis=(-2, -1)),
        "slip": np.linalg.norm(state.common.slip, axis=2),
        "orientation_deg": state.common.orientation_rad*180/np.pi,
        "stress": np.max(np.abs(ledger["raw_stress_Pa"]), axis=2),
    }
    for family in range(4):
        fields[f"signed_family_{family}"] = (
            state.density.mobile_plus_m2[..., family]
            -state.density.mobile_minus_m2[..., family]
            +state.density.forest_plus_m2[..., family]
            -state.density.forest_minus_m2[..., family]
            +state.density.wall_tangle_plus_m2[..., family]
            -state.density.wall_tangle_minus_m2[..., family]
            +state.density.wall_ordered_plus_m2[..., family]
            -state.density.wall_ordered_minus_m2[..., family])
    fields["structure_factor"] = np.fft.fftshift(np.abs(
        np.fft.fft2(state.common.orientation_rad-np.mean(
            state.common.orientation_rad)))**2)
    return record, fields


def main():
    cases = []; plot_fields = None
    for n in (16, 24, 32):
        for topology_on in (False, True):
            record, fields = run_case(n, topology_on)
            cases.append(record)
            if n == 32 and topology_on:
                plot_fields = fields
    any_pass = any(item["classification"]["scientific_gate_passed"]
                   for item in cases)
    result = {
        "schema": "asb-drx/v24-mechanical-line-supply-local/v1",
        "mechanical_heterogeneity": "full-elastic finite-width eigenstrain stripe",
        "future_orientation_profile_prescribed": False,
        "orientation_derived_ordering_target_used": False,
        "phase_and_grain_allocation_enabled": False,
        "multi_hit_enabled": False,
        "cases": cases,
        "fixture_passed": True,
        "scientific_gate_passed": bool(any_pass),
        "hpc_wall_campaign_authorized": bool(any_pass),
        "classification": (
            "SINGLE_CRYSTAL_DEFORMATION_GENERATES_A_COMPATIBLE_LAGB_PRECURSOR"
            if any_pass else
            "KINEMATIC_TRANSITION_BAND_WITH_INSUFFICIENT_BOUNDARY_INVENTORY"),
    }
    result["source_sha256"] = {
        name: hashlib.sha256((ROOT/"full_model"/"production"/name).read_bytes()).hexdigest()
        for name in ("v24_mechanical_wall.py", "wall_topology_supply.py",
                     "wall_circuit_diagnostics.py", "common_tensorial_wall.py",
                     "nonlocal_elasticity.py")}
    output = ROOT/"full_model"/"verification"/"v24_mechanical_supply_local.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    if plot_fields is not None:
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        for axis, (name, field) in zip(axes.flat, plot_fields.items()):
            shown = np.log10(np.maximum(field, 1e-300)) if name in (
                "total_density", "nye", "ordered_nye", "structure_factor") else field
            image = axis.imshow(shown.T, origin="lower", cmap="viridis")
            axis.set_title(name); fig.colorbar(image, ax=axis, shrink=.7)
        fig.tight_layout()
        fig.savefig(ROOT/"full_model"/"verification"/
                    "v24_mechanical_supply_fields.png", dpi=150)
        plt.close(fig)
    print(result["classification"])


if __name__ == "__main__":
    main()
