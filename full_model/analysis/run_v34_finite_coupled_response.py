#!/usr/bin/env python3
"""Finite-amplitude I3 transactions on the authoritative common state.

This module is intentionally an integration runner, not a second constitutive
model.  It composes three production transactions:

* ``accepted_v24_mechanical_step(..., mura_work_budget_mode="energy_limited")``
  for complete-affinity family selection and the one-flux Mura update;
* ``accept_coupled_front_candidate`` for the topology-aware signed sweep; and
* ``evaluate_common_front_transaction`` for complete-energy acceptance before
  publication of any phase-supported material owner.

The V35 checkpoint schema carries reservoir-resolved scalar inventories and
line moments for parent, child, and processed wake owners.  Accepted front
partitions transfer both from the same event, so the next Mura step consumes
the actual post-front state and repeated cycles are representable.  Phase
trials are supplied by the production phase proposer; this file never
manufactures or translates a front on its own.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_front_state import (
    apply_mechanical_increment, attach_reservoir_moment_owners,
    initialize_common_front, reconstruct_common, reconstruct_mechanical_state,
    state_arrays as common_front_arrays,
    state_from_checkpoint as common_front_from_checkpoint,
    state_metadata_json as common_front_metadata_json,
)
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallState, resolved_driving_components,
)
from full_model.production.complete_front_energy import (
    evaluate_common_front_transaction, evaluate_complete_directional_kinetics,
    evaluate_complete_front_energy,
)
from full_model.production.coupled_front_production import (
    accept_coupled_front_candidate, initialize_coupled_front_runtime,
    runtime_arrays, runtime_from_checkpoint, runtime_metadata_json,
)
from full_model.production.density_state_map import (
    derived_density_fields, from_v22_common_state,
)
from full_model.production.moving_front import (
    DefectState, initialize_declared_boundary_front,
    state_arrays as sparse_front_arrays,
    state_from_checkpoint as sparse_front_from_checkpoint,
    state_metadata_json as sparse_front_metadata_json,
)
from full_model.production.tensorial_nye import rotated_system_fields
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, accepted_v24_mechanical_step,
    mechanical_checkpoint_arrays, mechanical_from_checkpoint_arrays,
)
from full_model.production.wall_topology_supply import (
    aligned_state_from_directions,
)


SCHEMA = "asb-drx/v35-finite-common-state-multicycle/v2"


@dataclass(frozen=True)
class I3State:
    mechanical: V24MechanicalWallState
    common_front: object
    front_runtime: object
    eta: np.ndarray


@dataclass(frozen=True)
class I3Controls:
    mura_enabled: bool = True
    front_enabled: bool = True
    prescribed_temperature: bool = False
    driving_pressure_a_to_b_Pa: float = 0.0
    applied_pressure_a_to_b_Pa: float = 0.0
    geometric_probe_pressure_Pa: float | None = None
    trial_dt_s: float = 2.0e-9
    front_dt_s: float = 1.0e-6
    transmission_fraction: float = 0.5
    boundary_storage_fraction: float = 0.05
    neutral_sink_fraction: float = 0.02
    signed_sink_fraction: float = 0.0
    front_activation_h0_eV: float = 0.35
    front_exp_a: float = 2.0
    front_exp_n: float = 1.5
    front_exp_floor: float = 0.10
    front_attempt_frequency_s: float = 1.0e8
    front_activation_entropy_kB: float = 0.0
    front_event_volume_b3: float = 1.0
    front_jump_length_b: float = 1.0
    front_symmetric_availability: float = 1.0
    deterministic_front_rate_law: str = "complete_dissipation"


def _owner_with_line_scale(state, factor):
    line_names = {
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
        "junction_m2", "alignment_m2", "family_nye_m1",
    }
    return CommonWallState(**{
        item.name: (np.asarray(getattr(state, item.name))*float(factor)
                    if item.name in line_names
                    else np.asarray(getattr(state, item.name)).copy())
        for item in fields(CommonWallState)
    })


def _defect(owner):
    return DefectState(
        owner.mobile_plus_m2.copy(), owner.mobile_minus_m2.copy(),
        (owner.forest_plus_m2+owner.forest_minus_m2).copy(),
        np.sum(owner.wall_plus_m2+owner.wall_minus_m2, axis=2))


def resolved_bicrystal(grid=64, length_m=1.0e-5,
                       interface_width_m=3.0e-7,
                       child_line_fraction=0.35, temperature_K=1100.0):
    """Construct an existing periodic bicrystal and one shared physical state."""
    if grid < 16 or interface_width_m < 2.0*length_m/grid:
        raise ValueError("bicrystal interface must be resolved by at least two cells")
    (base, _, _, support, systems, topologies, parameters, extensive,
     kinetics, spacing) = build_case(
        grid, length_m=length_m, periodic_nye_consistent=True)
    parameters = replace(parameters, bath_temperature_K=float(temperature_K))
    # Existing-boundary production trajectories use the V39 bounded stiff
    # integration of the unchanged ordering law.  The explicit complete-time
    # implementation remains available as the short-horizon reference oracle.
    extensive = replace(
        extensive, ordering_integration_method="implicit_backward_euler",
        # Low-exposure production calls use a bounded resolved reference; the
        # dedicated oracle audit below V39 retains the stricter 0.5 ps spacing.
        ordering_internal_substep_s=5e-11,
        ordering_internal_max_substeps=8192)
    parent = replace(
        base.common,
        temperature_K=np.full((grid, grid), float(temperature_K)))
    child = _owner_with_line_scale(parent, child_line_fraction)
    x = np.arange(grid)*spacing
    left, right = .25*length_m, .75*length_m
    fraction_1d = .5*(np.tanh((x-left)/interface_width_m)
                      -np.tanh((x-right)/interface_width_m))
    chi = np.broadcast_to(np.clip(fraction_1d[:, None], 0.0, 1.0),
                          (grid, grid)).copy()
    eta = np.stack((1.0-chi, chi), axis=2)
    front = initialize_declared_boundary_front(
        _defect(parent), _defect(child), chi, 0, 1)
    adapter = initialize_common_front(front, parent)
    # initialize_common_front obtains all signed line owners from the declared
    # parent/child defect states.  Preserve the intended non-line child fields.
    adapter = replace(adapter, child=replace(
        adapter.child,
        temperature_K=child.temperature_K.copy(),
        orientation_rad=child.orientation_rad.copy()))
    owner_densities = []
    owner_alignments = []
    for owner in (adapter.parent, adapter.child, adapter.wake):
        density = from_v22_common_state(
            owner, ordered_fraction=owner.wall_order)
        _, slip_direction, plane_normal = rotated_system_fields(
            systems, owner.orientation_rad)
        alignment = aligned_state_from_directions(
            density, np.cross(plane_normal, slip_direction))
        owner_densities.append(density)
        owner_alignments.append(alignment)
    adapter = attach_reservoir_moment_owners(
        adapter, *owner_densities, *owner_alignments, systems, topologies)
    mechanical = reconstruct_mechanical_state(
        adapter, spacing, systems, topologies)
    runtime = initialize_coupled_front_runtime(
        adapter.front, eta[..., 1]-eta[..., 0], normal_axis=0,
        periodic=True)
    return {
        "state": I3State(mechanical, adapter, runtime, eta),
        "systems": systems, "topologies": topologies,
        "wall_parameters": parameters,
        "extensive_parameters": extensive,
        "topology_kinetics": kinetics, "capture_support": support,
        "spacing_m": spacing,
        "represented_thickness_m": 2.0*parameters.burgers_m,
        "interface_width_m": float(interface_width_m),
    }


def _inventory(common, topologies, cell_volume_m3):
    density = from_v22_common_state(common, ordered_fraction=0.0)
    derived = derived_density_fields(density, topologies)
    signed = sum(
        np.asarray(getattr(density, stem+"_plus_m2"))
        -np.asarray(getattr(density, stem+"_minus_m2"))
        for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"))
    return {
        "total_line_m": float(np.sum(derived["rho_total_m2"],
                                     dtype=np.longdouble)*cell_volume_m3),
        "signed_family_line_m": [float(value) for value in
            np.sum(signed, axis=(0, 1), dtype=np.longdouble)*cell_volume_m3],
        "junction_extent_m": [float(value) for value in
            np.sum(common.junction_m2, axis=(0, 1),
                   dtype=np.longdouble)*cell_volume_m3],
    }


def _delta(after, before):
    return {
        "total_line_m": after["total_line_m"]-before["total_line_m"],
        "signed_family_line_m": [a-b for a, b in zip(
            after["signed_family_line_m"], before["signed_family_line_m"])],
        "junction_extent_m": [a-b for a, b in zip(
            after["junction_extent_m"], before["junction_extent_m"])],
    }


def _energy_options(context, driving):
    p = context["wall_parameters"]
    return dict(
        wall_parameters=p, mean_strain=driving.mean_strain,
        topologies=context["topologies"], systems=context["systems"],
        phase_barrier_J_m3=5.0e6, phase_gradient_J_m=5.0e-7,
        boundary_line_energy_J_m=p.line_energy_J_m,
        boundary_junction_energy_J_m=p.junction_energy_J_m,
        reference_temperature_K=p.bath_temperature_K)


def run_i3_cycle(context, state, eta_trial, driving, controls=I3Controls()):
    """Run one immutable Mura/front cycle and return state plus complete audit."""
    eta_trial = np.asarray(eta_trial, dtype=float)
    if eta_trial.shape != state.eta.shape:
        raise ValueError("the production phase proposal is not grid matched")
    spacing = context["spacing_m"]
    thickness = context["represented_thickness_m"]
    cell_volume = spacing*spacing*thickness
    common0, _ = reconstruct_common(state.common_front, spacing)
    inventory0 = _inventory(common0, context["topologies"], cell_volume)
    front_state = state.common_front
    mechanical = reconstruct_mechanical_state(
        state.common_front, spacing, context["systems"],
        context["topologies"])
    mura_ledger = None
    if controls.mura_enabled:
        mechanical, mura_ledger = accepted_v24_mechanical_step(
            mechanical, driving, context["capture_support"],
            context["systems"], context["topologies"],
            context["wall_parameters"], context["extensive_parameters"],
            context["topology_kinetics"], controls.trial_dt_s,
            topology_route_enabled=False,
            mura_work_budget_mode="energy_limited")
        if mura_ledger["mura_work_budget"]["family_selection_rule"] != (
                "complete_discrete_full_event_affinity_then_joint_backtrack"):
            raise RuntimeError("I3 requires complete-affinity Mura selection")
        if controls.prescribed_temperature:
            # Heat remains in the independently evaluated Mura ledger.  Only
            # the thermal state is clamped by this boundary-condition control.
            mechanical = replace(mechanical, common=replace(
                mechanical.common,
                temperature_K=state.mechanical.common.temperature_K.copy()))
        front_state = apply_mechanical_increment(
            front_state, mechanical, spacing, context["systems"],
            context["topologies"])

    runtime = state.front_runtime
    accepted_eta = state.eta.copy()
    front_decision = None
    energy_decision = None
    directional_kinetics = None
    front_published = False
    before_front = front_state
    burgers = float(context["wall_parameters"].burgers_m)
    kinetic_event_volume = controls.front_event_volume_b3*burgers**3
    kinetic_event_length = controls.front_jump_length_b*burgers
    if controls.front_enabled:
        process = ActivatedProcess(
            "v34-i3-existing-boundary",
            controls.front_attempt_frequency_s*controls.front_symmetric_availability,
            entropy_over_kB=controls.front_activation_entropy_kB,
            negative_barrier_mode="drag")
        # The first call only constructs and topology-checks the supplied
        # geometric candidate. It has zero channel availability and applies
        # no pressure, work, or direction selection. The second call below
        # uses rates derived from the complete candidate endpoint energies.
        sparse_candidate, runtime_candidate, accepted_eta, front_decision = (
            accept_coupled_front_candidate(
                front_state.front, runtime, state.eta, eta_trial,
                spacing_m=spacing, represented_thickness_m=thickness,
                dt_s=controls.front_dt_s,
                temperature_K=mechanical.common.temperature_K,
                line_energy_J_m=context["wall_parameters"].line_energy_J_m,
                process=process, h0_J=controls.front_activation_h0_eV*EV_J,
                critical_pressure_Pa=1.0e9, exp_a=controls.front_exp_a,
                exp_n=controls.front_exp_n, exp_floor=controls.front_exp_floor,
                driving_pressure_a_to_b_Pa=0.0,
                applied_pressure_a_to_b_Pa=0.0,
                mobility_enabled=True, periodic=True,
                transmission_fraction=controls.transmission_fraction,
                boundary_storage_fraction=controls.boundary_storage_fraction,
                neutral_sink_fraction=controls.neutral_sink_fraction,
                signed_sink_fraction=controls.signed_sink_fraction,
                support_component_reconnection=True,
                topology_backtracking_enabled=True,
                kinetic_event_volume_m3=kinetic_event_volume,
                kinetic_event_length_m=kinetic_event_length,
                proposal_probe_only=True,
                deterministic_rate_law=(
                    controls.deterministic_front_rate_law)))
        if front_decision.accepted:
            directional_kinetics = evaluate_complete_directional_kinetics(
                front_state, sparse_candidate, state.eta, accepted_eta,
                event_volume_m3=kinetic_event_volume, spacing_m=spacing,
                cell_volume_m3=cell_volume,
                represented_thickness_m=thickness,
                transmission_fraction=controls.transmission_fraction,
                boundary_storage_fraction=controls.boundary_storage_fraction,
                neutral_sink_fraction=controls.neutral_sink_fraction,
                signed_sink_fraction=controls.signed_sink_fraction,
                energy_kwargs=_energy_options(context, driving),
                external_work_density_Pa=(
                    controls.applied_pressure_a_to_b_Pa),
                prescribed_temperature=controls.prescribed_temperature)
            sparse_candidate, runtime_candidate, accepted_eta, front_decision = (
                accept_coupled_front_candidate(
                    front_state.front, runtime, state.eta, eta_trial,
                    spacing_m=spacing, represented_thickness_m=thickness,
                    dt_s=controls.front_dt_s,
                    temperature_K=mechanical.common.temperature_K,
                    line_energy_J_m=context["wall_parameters"].line_energy_J_m,
                    process=process, h0_J=controls.front_activation_h0_eV*EV_J,
                    critical_pressure_Pa=1.0e9, exp_a=controls.front_exp_a,
                    exp_n=controls.front_exp_n,
                    exp_floor=controls.front_exp_floor,
                    driving_pressure_a_to_b_Pa=(
                        controls.driving_pressure_a_to_b_Pa),
                    applied_pressure_a_to_b_Pa=(
                        controls.applied_pressure_a_to_b_Pa),
                    mobility_enabled=True, periodic=True,
                    transmission_fraction=controls.transmission_fraction,
                    boundary_storage_fraction=controls.boundary_storage_fraction,
                    neutral_sink_fraction=controls.neutral_sink_fraction,
                    signed_sink_fraction=controls.signed_sink_fraction,
                    support_component_reconnection=True,
                    topology_backtracking_enabled=True,
                    kinetic_free_energy_a_to_b_J=(
                        directional_kinetics.a_to_b_event_J),
                    kinetic_free_energy_b_to_a_J=(
                        directional_kinetics.b_to_a_event_J),
                    kinetic_event_volume_m3=kinetic_event_volume,
                    kinetic_event_length_m=kinetic_event_length,
                    actual_reverse_edge=directional_kinetics.actual_reverse_edge,
                    deterministic_rate_law=(
                        controls.deterministic_front_rate_law)))
        transaction = evaluate_common_front_transaction(
            front_state, sparse_candidate, state.eta, accepted_eta,
            spacing_m=spacing, cell_volume_m3=cell_volume,
            represented_thickness_m=thickness,
            transmission_fraction=controls.transmission_fraction,
            boundary_storage_fraction=controls.boundary_storage_fraction,
            neutral_sink_fraction=controls.neutral_sink_fraction,
            signed_sink_fraction=controls.signed_sink_fraction,
            energy_kwargs=_energy_options(context, driving),
            external_work_J=(controls.applied_pressure_a_to_b_Pa
                             *front_decision.accepted_signed_volume_m3),
            prescribed_temperature=controls.prescribed_temperature)
        energy_decision = transaction.decision
        if front_decision.accepted and energy_decision.accepted:
            front_state = transaction.published_state
            runtime = runtime_candidate
            front_published = True
        elif (front_decision.classification == "STATIONARY_GEOMETRY"
              and energy_decision.accepted):
            accepted_eta = accepted_eta.copy()
            runtime = runtime_candidate
        else:
            front_state = before_front
            runtime = state.front_runtime
            accepted_eta = state.eta.copy()

    # The next Mura interval is reconstructed only from evolved owner
    # inventories and moments.  This is an exact support-weighted read, not a
    # Nye inversion or a manufactured realignment.
    mechanical = reconstruct_mechanical_state(
        front_state, spacing, context["systems"], context["topologies"])

    common1, _ = reconstruct_common(front_state, spacing)
    inventory1 = _inventory(common1, context["topologies"], cell_volume)
    boundary0 = float(np.sum(
        state.common_front.boundary_plus_m2
        +state.common_front.boundary_minus_m2,
        dtype=np.longdouble)*cell_volume)
    boundary0 += float(np.sum(
        state.common_front.boundary_junction_m2,
        dtype=np.longdouble)*cell_volume)
    boundary1 = float(np.sum(
        front_state.boundary_plus_m2+front_state.boundary_minus_m2,
        dtype=np.longdouble)*cell_volume)
    boundary1 += float(np.sum(
        front_state.boundary_junction_m2,
        dtype=np.longdouble)*cell_volume)
    before_energy = evaluate_complete_front_energy(
        state.common_front, state.eta, spacing_m=spacing,
        represented_thickness_m=thickness,
        **_energy_options(context, driving))
    after_energy = evaluate_complete_front_energy(
        front_state, accepted_eta, spacing_m=spacing,
        represented_thickness_m=thickness,
        **_energy_options(context, driving))
    phase_change = accepted_eta-state.eta
    mura_budget = None if mura_ledger is None else mura_ledger["mura_work_budget"]
    mura_balance = None if mura_ledger is None else mura_ledger["mura_balance_ledger"]
    diagnostics = {
        "schema": SCHEMA,
        "single_cycle_only": False,
        "next_mura_state_from_owned_reservoir_moments": True,
        "mura_enabled": controls.mura_enabled,
        "front_enabled": controls.front_enabled,
        "prescribed_temperature": controls.prescribed_temperature,
        "geometric_probe_pressure_Pa": (
            None if controls.geometric_probe_pressure_Pa is None else
            float(controls.geometric_probe_pressure_Pa)),
        "geometric_probe_is_nonphysical_and_unledgered": True,
        "kinetic_event_volume_m3": float(kinetic_event_volume),
        "kinetic_event_length_m": float(kinetic_event_length),
        "kinetic_normalization_is_grid_independent": True,
        "physical_site_event_measure": ({
            "physical_site_count": front_decision.physical_site_count,
            "expected_events_a_to_b": front_decision.expected_events_a_to_b,
            "expected_events_b_to_a": front_decision.expected_events_b_to_a,
            "expected_signed_event_count": (
                front_decision.expected_signed_event_count),
            "expected_signed_swept_volume_m3": (
                front_decision.expected_signed_swept_volume_m3),
            "expected_normal_velocity_m_s": (
                front_decision.expected_normal_velocity_m_s),
            "site_count_per_interface_area_m2": (
                front_decision.site_count_per_interface_area_m2),
            "event_count_per_interface_area": (
                front_decision.event_count_per_interface_area),
        } if front_decision is not None else None),
        "sweep": ({
            "positive_m3": front_decision.positive_swept_volume_m3,
            "negative_m3": front_decision.negative_swept_volume_m3,
            "absolute_m3": front_decision.absolute_swept_volume_m3,
            "net_m3": (front_decision.positive_swept_volume_m3
                       -front_decision.negative_swept_volume_m3),
            "components": list(front_decision.component_motion),
        } if front_decision is not None and front_published else {
            "positive_m3": 0.0, "negative_m3": 0.0,
            "absolute_m3": 0.0, "net_m3": 0.0, "components": []}),
        "candidate_sweep_published": front_published,
        "phase": {
            "maximum_abs_change": float(np.max(np.abs(phase_change))),
            "rms_change": float(np.sqrt(np.mean(phase_change*phase_change))),
            "child_fraction_change": float(np.mean(
                accepted_eta[..., 1]-state.eta[..., 1])),
        },
        "inventory_before": inventory0,
        "inventory_after": inventory1,
        "actual_inventory_change": _delta(inventory1, inventory0),
        "actual_boundary_inventory_change_m": boundary1-boundary0,
        "actual_material_sink_change_m": (
            front_state.ledger.sink_line_m
            -state.common_front.ledger.sink_line_m),
        "actual_processed_line_change_m": (
            front_state.ledger.processed_line_m
            -state.common_front.ledger.processed_line_m),
        "maximum_abs_slip": float(np.max(np.abs(
            common1.slip-common0.slip))),
        "maximum_abs_beta_p": float(np.max(np.abs(
            common1.beta_p-common0.beta_p))),
        "temperature_range_K": [float(np.min(common1.temperature_K)),
                                float(np.max(common1.temperature_K))],
        "complete_energy": {
            "before": asdict(before_energy), "after": asdict(after_energy),
            "delta_helmholtz_J": after_energy.helmholtz_J-before_energy.helmholtz_J,
            "front_decision": (None if energy_decision is None
                               else energy_decision.as_dict()),
        },
        "front_decision": (None if front_decision is None
                           else asdict(front_decision)),
        "complete_directional_kinetics": (
            None if directional_kinetics is None else {
                "a_to_b_event_J": directional_kinetics.a_to_b_event_J,
                "b_to_a_event_J": directional_kinetics.b_to_a_event_J,
                "forward_signed_volume_m3": (
                    directional_kinetics.forward_signed_volume_m3),
                "opposite_signed_volume_m3": (
                    directional_kinetics.opposite_signed_volume_m3),
                "a_to_b_endpoint": asdict(
                    directional_kinetics.a_to_b_endpoint),
                "b_to_a_endpoint": asdict(
                    directional_kinetics.b_to_a_endpoint),
                "actual_reverse_edge": (
                    directional_kinetics.actual_reverse_edge),
                "reverse_edge_status": (
                    directional_kinetics.reverse_edge_status),
                "opposite_evaluated_from_actual_state": True,
            }),
        "mura": (None if mura_ledger is None else {
            "accepted_dt_s": float(mura_ledger["accepted_dt_s"]),
            "event_scale": float(mura_ledger["mura_event_scale"]),
            "family_event_scales": [float(x) for x in
                                     mura_ledger["mura_family_event_scales"]],
            "family_selection_rule": mura_budget["family_selection_rule"],
            "unreacted_remainder_semantics": mura_budget[
                "unreacted_remainder_semantics"],
            "first_law_residual_J_m3_cells": float(mura_balance[
                "global_work_minus_heat_storage_residual_J_m3_cells"]),
            "minimum_heat_increment_J_m3": float(np.min(
                mura_balance["deposited_heat_increment_J_m3"])),
            "plastic_work_increment_J_m3_cells": float(np.sum(
                mura_balance["plastic_work_increment_J_m3"])),
            "deposited_heat_increment_J_m3_cells": float(np.sum(
                mura_balance["deposited_heat_increment_J_m3"])),
            "stored_line_energy_increment_J_m3_cells": float(np.sum(
                mura_balance["stored_line_energy_increment_J_m3"])),
            "ordering_integration_method": mura_ledger[
                "ordering_thermodynamics"].get("integration_method"),
            "ordering_stiff_dispatch": mura_ledger[
                "ordering_thermodynamics"].get("stiff_dispatch"),
            "ordering_complete_elapsed_time_s": float(mura_ledger[
                "ordering_thermodynamics"].get(
                    "complete_elapsed_time_s", mura_ledger["accepted_dt_s"])),
            "ordering_discarded_reaction_time_s": float(mura_ledger[
                "ordering_thermodynamics"].get(
                    "discarded_reaction_time_s", 0.0)),
            "ordering_endpoint_remainder_relative": mura_ledger[
                "ordering_thermodynamics"].get(
                    "asymptotic_endpoint_inventory_change_bound_relative"),
            "ordering_endpoint_diagnostic_semantics": mura_ledger[
                "ordering_thermodynamics"].get(
                    "asymptotic_endpoint_diagnostic_semantics"),
            "ordering_finite_time_kinetic_accuracy_certified_by_this_solve": (
                mura_ledger["ordering_thermodynamics"].get(
                    "finite_time_kinetic_accuracy_certified_by_this_solve")),
            "ordering_maximum_attempt_exposure": mura_ledger[
                "ordering_thermodynamics"].get("maximum_attempt_exposure"),
            "ordering_solver_evaluations": int(mura_ledger[
                "ordering_thermodynamics"].get("implicit_nfev", 0)),
        }),
    }
    return I3State(mechanical, front_state, runtime, accepted_eta), diagnostics


def compare_response_family(context, initial, forward_trial, reverse_trial,
                            driving, controls=I3Controls()):
    """Evaluate all I3 controls from byte-identical initial values."""
    definitions = {
        "front_only": (forward_trial, replace(controls, mura_enabled=False)),
        "mura_only": (initial.eta, replace(controls, front_enabled=False)),
        "combined": (forward_trial, controls),
        "prescribed_temperature": (
            forward_trial, replace(controls, prescribed_temperature=True)),
        "reversed_contrast": (reverse_trial, replace(
            controls,
            driving_pressure_a_to_b_Pa=-controls.driving_pressure_a_to_b_Pa,
            applied_pressure_a_to_b_Pa=-controls.applied_pressure_a_to_b_Pa)),
    }
    states = {}; cases = {}
    for name, (trial, case_controls) in definitions.items():
        states[name], cases[name] = run_i3_cycle(
            context, initial, trial, driving, case_controls)

    # Reciprocal measures are independently recomputed observables, not
    # inferred residuals: Mura's effect on the accepted front sweep and the
    # front's effect on the next resolved stress/speed fields.
    combined_common, _ = reconstruct_common(
        states["combined"].common_front, context["spacing_m"])
    mura_only_common, _ = reconstruct_common(
        states["mura_only"].common_front, context["spacing_m"])
    combined_drive = resolved_driving_components(
        combined_common, driving, context["systems"], context["topologies"],
        context["wall_parameters"])
    mura_only_drive = resolved_driving_components(
        mura_only_common, driving, context["systems"], context["topologies"],
        context["wall_parameters"])
    interactions = {
        "mura_to_front_net_sweep_change_m3": (
            cases["combined"]["sweep"]["net_m3"]
            -cases["front_only"]["sweep"]["net_m3"]),
        "front_to_next_mura_raw_stress_rms_change_Pa": float(np.sqrt(np.mean(
            (combined_drive["raw_stress_Pa"]
             -mura_only_drive["raw_stress_Pa"])**2))),
        "front_to_next_mura_speed_rms_change_m_s": float(np.sqrt(np.mean(
            (combined_drive["speed_m_s"]-mura_only_drive["speed_m_s"])**2))),
        "next_mura_extent_not_executed": False,
    }
    return states, {"schema": SCHEMA, "cases": cases,
                    "reciprocal_interactions": interactions}


def run_i3_intervals(context, initial, phase_trials, driving,
                     controls=I3Controls()):
    """Execute a declared sequence of production Mura/front intervals."""
    state = initial
    records = []
    for index, trial in enumerate(phase_trials):
        if callable(trial):
            eta_trial, interval_controls = trial(index, state, controls)
        elif isinstance(trial, tuple):
            eta_trial, interval_controls = trial
        else:
            eta_trial, interval_controls = trial, controls
        state, audit = run_i3_cycle(
            context, state, eta_trial, driving, interval_controls)
        records.append({
            "interval": index,
            "front_classification": audit["front_decision"]["classification"],
            "front_published": audit["candidate_sweep_published"],
            "signed_sweep_m3": audit["sweep"]["net_m3"],
            "mura_event_scale": audit["mura"]["event_scale"],
            "mura_family_event_scales": audit["mura"][
                "family_event_scales"],
            "maximum_abs_slip_increment": audit["maximum_abs_slip"],
            "maximum_abs_beta_increment": audit["maximum_abs_beta_p"],
        })
    return state, {
        "schema": SCHEMA, "interval_count": len(records),
        "accepted_front_intervals": sum(
            record["front_published"] for record in records),
        "cumulative_signed_sweep_m3": float(sum(
            record["signed_sweep_m3"] for record in records)),
        "records": records,
    }


def checkpoint_payload(state, context):
    """Return a lossless one-cycle restart payload (no filesystem side effect)."""
    payload = {
        "i3_metadata_json": np.asarray(json.dumps({
            "schema": SCHEMA,
            "common_front": common_front_metadata_json(state.common_front),
            "sparse_front": sparse_front_metadata_json(state.common_front.front),
            "runtime": runtime_metadata_json(state.front_runtime),
        }, sort_keys=True)),
        "eta": np.asarray(state.eta),
    }
    payload.update({"mechanical__"+k: v for k, v in
                    mechanical_checkpoint_arrays(state.mechanical).items()})
    payload.update({"common__"+k: v for k, v in
                    common_front_arrays(state.common_front).items()})
    payload.update({"sparse__"+k: v for k, v in
                    sparse_front_arrays(state.common_front.front).items()})
    payload.update({"runtime__"+k: v for k, v in
                    runtime_arrays(state.front_runtime).items()})
    return payload


def state_from_payload(payload, context):
    metadata = json.loads(str(np.asarray(payload["i3_metadata_json"]).item()))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported I3 checkpoint schema")
    select = lambda prefix: {
        key.removeprefix(prefix): np.asarray(value)
        for key, value in payload.items() if key.startswith(prefix)}
    sparse = sparse_front_from_checkpoint(
        metadata["sparse_front"], select("sparse__"))
    common = common_front_from_checkpoint(
        metadata["common_front"], select("common__"), sparse)
    runtime = runtime_from_checkpoint(
        metadata["runtime"], select("runtime__"))
    mechanical = mechanical_from_checkpoint_arrays(
        select("mechanical__"), context["systems"], context["topologies"])
    return I3State(mechanical, common, runtime,
                   np.asarray(payload["eta"]).copy())


def write_result(path, result):
    """Write an explicitly requested result; campaign controllers own paths."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


def main():
    parser = argparse.ArgumentParser(
        description=("Run one V34 complete-affinity/common-front I3 cycle "
                     "using phase proposals produced by the qualified solver."))
    parser.add_argument("--phase-trials-npz", type=Path, required=True,
                        help="NPZ containing forward_eta and reverse_eta")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--length-m", type=float, default=1.0e-5)
    parser.add_argument("--interface-width-m", type=float, default=3.0e-7)
    parser.add_argument("--temperature-K", type=float, default=1100.0)
    parser.add_argument("--child-line-fraction", type=float, default=.35)
    parser.add_argument("--mean-shear-strain", type=float, default=.01)
    parser.add_argument("--trial-dt-s", type=float, default=2.0e-9)
    parser.add_argument("--front-dt-s", type=float, default=1.0e-6)
    parser.add_argument("--driving-pressure-Pa", type=float, required=True)
    parser.add_argument("--applied-pressure-Pa", type=float, default=0.0,
                        help="external front work density; ledgered separately")
    parser.add_argument("--geometric-probe-pressure-Pa", type=float,
                        help=("unledgered proposal-only pressure; final rate "
                              "is recomputed from complete event energies"))
    args = parser.parse_args()
    context = resolved_bicrystal(
        args.grid, args.length_m, args.interface_width_m,
        args.child_line_fraction, args.temperature_K)
    with np.load(args.phase_trials_npz, allow_pickle=False) as archive:
        if "forward_eta" not in archive or "reverse_eta" not in archive:
            raise ValueError("phase trial archive requires forward_eta and reverse_eta")
        forward = np.asarray(archive["forward_eta"]).copy()
        reverse = np.asarray(archive["reverse_eta"]).copy()
    fixed = np.zeros((args.grid, args.grid, 2, 2))
    driving = CommonWallDriving(
        mean_strain=np.array([[0.0, args.mean_shear_strain],
                              [args.mean_shear_strain, 0.0]]),
        fixed_eigenstrain=fixed)
    controls = I3Controls(
        driving_pressure_a_to_b_Pa=args.driving_pressure_Pa,
        applied_pressure_a_to_b_Pa=args.applied_pressure_Pa,
        geometric_probe_pressure_Pa=args.geometric_probe_pressure_Pa,
        trial_dt_s=args.trial_dt_s, front_dt_s=args.front_dt_s)
    _, result = compare_response_family(
        context, context["state"], forward, reverse, driving, controls)
    result["configuration"] = {
        "grid": args.grid, "length_m": args.length_m,
        "interface_width_m": args.interface_width_m,
        "temperature_K": args.temperature_K,
        "child_line_fraction": args.child_line_fraction,
        "mean_shear_strain": args.mean_shear_strain,
        "trial_dt_s": args.trial_dt_s, "front_dt_s": args.front_dt_s,
        "driving_pressure_Pa": args.driving_pressure_Pa,
        "applied_pressure_Pa": args.applied_pressure_Pa,
        "geometric_probe_pressure_Pa": args.geometric_probe_pressure_Pa,
        "phase_trials_npz": str(args.phase_trials_npz.resolve()),
    }
    write_result(args.output, result)


if __name__ == "__main__":
    main()
