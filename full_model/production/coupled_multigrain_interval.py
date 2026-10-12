"""One complete mechanics/thermal/front interval for adaptive production.

This module contains the physical interval map used by the V64 controller.
It preserves the established operator order and constitutive laws while making
the complete update callable on an immutable parent endpoint.  Trial ledgers,
heat, histories, and material states therefore remain private until the outer
accuracy controller publishes an accepted fine path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Mapping

import numpy as np

from .common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components, wall_residual,
)
from .coupled_interval_control import CoupledEndpoint, CoupledInterval
from .multigrain_common_state import (
    audit_multigrain_nye, reconstruct_multigrain_common,
)
from .multigrain_production import (
    MultiGrainProductionRuntime, _owner_drivings_from_common_stress,
    advance_energy_qualified_mechanics, advance_multigrain_front_interval,
    multigrain_instantaneous_dissipation_fields,
)


@dataclass(frozen=True)
class CoupledMultigrainContext:
    systems: tuple
    topologies: tuple
    wall_parameters: object
    front_kinetics: object
    spacing_m: float
    represented_thickness_m: float
    shear_rate_s: float
    maximum_front_substep_s: float
    maximum_mechanical_subdivisions: int
    energy_kwargs: Mapping[str, object]
    heat_deposition_mode: str = "local_realized_event"
    front_enabled: bool = True


def endpoint_observables(state, load: float,
                         context: CoupledMultigrainContext) -> dict:
    spacing = float(context.spacing_m)
    strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    driving = CommonWallDriving(mean_strain=strain)
    common, _ = reconstruct_multigrain_common(state, spacing)
    nye = audit_multigrain_nye(state, spacing)
    instantaneous = multigrain_instantaneous_dissipation_fields(
        state, driving=driving, systems=context.systems,
        topologies=context.topologies, wall_parameters=context.wall_parameters)
    owner_drivings = _owner_drivings_from_common_stress(
        state, driving, context.systems, context.wall_parameters)
    owner_speeds = []
    owner_slip_rates = []
    for support, owner, owner_driving in zip(
            state.supports, state.owners, owner_drivings):
        resolved = resolved_driving_components(
            owner, owner_driving, context.systems, context.topologies,
            context.wall_parameters)
        residual = wall_residual(
            owner, owner_driving, context.systems, context.topologies,
            context.wall_parameters)
        weight = np.asarray(support)[..., None]
        owner_speeds.append(weight*np.asarray(resolved["speed_m_s"]))
        owner_slip_rates.append(weight*np.asarray(residual.state_rate.slip))
    signed_total = sum(
        state.supports[index][..., None]*sum(
            np.asarray(getattr(owner, plus))-np.asarray(getattr(owner, minus))
            for plus, minus in (("mobile_plus_m2", "mobile_minus_m2"),
                                ("forest_plus_m2", "forest_minus_m2"),
                                ("wall_plus_m2", "wall_minus_m2")))
        for index, owner in enumerate(state.owners))
    return {
        "supports": np.asarray(state.supports),
        "common_temperature_K": np.asarray(common.temperature_K),
        "common_beta_p": np.asarray(common.beta_p),
        "signed_total_density_m2": np.asarray(signed_total),
        "exact_reconstructed_nye_m1": np.asarray(nye.exact_reconstructed_m1),
        "common_stress_tensor_Pa": np.asarray(
            owner_drivings[0].fixed_stress_tensor_Pa),
        "instantaneous_plastic_power_W_m3": np.asarray(
            instantaneous["plastic_power_W_m3"]),
        "instantaneous_signed_slip_rate_s": np.asarray(
            instantaneous["signed_slip_rate_s"]),
        "owner_weighted_speed_m_s": np.stack(owner_speeds, axis=2),
        "owner_weighted_signed_slip_rate_s": np.stack(
            owner_slip_rates, axis=2),
    }


def make_endpoint(state, runtime, load: float,
                  context: CoupledMultigrainContext) -> CoupledEndpoint:
    return CoupledEndpoint(
        state, runtime, float(load), endpoint_observables(state, load, context))


def advance_complete_multigrain_interval(
        parent: CoupledEndpoint, duration_s: float,
        context: CoupledMultigrainContext) -> CoupledInterval:
    """Apply exactly one complete coupled physical interval.

    Mechanical and moving-front operators both represent the same physical
    interval.  Only the front runtime owns the physical clock, so their
    durations are never added to one another.
    """
    dt = float(duration_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("complete coupled interval must be positive and finite")
    gamma0 = float(parent.load)
    gamma1 = gamma0+float(context.shear_rate_s)*dt
    strain0 = np.array([[0.0, .5*gamma0], [.5*gamma0, 0.0]])
    strain1 = np.array([[0.0, .5*gamma1], [.5*gamma1, 0.0]])
    energy_before = dict(context.energy_kwargs, mean_strain=strain0)
    mechanical_started = time.perf_counter()
    mechanical = advance_energy_qualified_mechanics(
        parent.state, mean_strain_before=strain0,
        mean_strain_candidate=strain1, systems=context.systems,
        topologies=context.topologies, wall_parameters=context.wall_parameters,
        dt_s=dt, represented_thickness_m=context.represented_thickness_m,
        energy_kwargs=energy_before, maximum_relative_first_law_residual=.05,
        maximum_subdivisions=context.maximum_mechanical_subdivisions)
    mechanical_wall_time = time.perf_counter()-mechanical_started
    mechanical_heat = sum(
        (np.asarray(item.irreversible_heat_J_m3_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    mechanical_work_density = sum(
        (np.asarray(item.plastic_work_J_m3_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    conduction = sum(
        (np.asarray(item.thermal_conduction_J_m3_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    bath = sum(
        (np.asarray(item.thermal_bath_exchange_J_m3_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    runtime = replace(parent.runtime, ledger=replace(
        parent.runtime.ledger,
        cumulative_mechanical_external_work_J=(
            parent.runtime.ledger.cumulative_mechanical_external_work_J
            +mechanical.external_work_J),
        cumulative_mechanical_internal_energy_change_J=(
            parent.runtime.ledger.cumulative_mechanical_internal_energy_change_J
            +mechanical.internal_energy_change_J),
        cumulative_mechanical_first_law_residual_J=(
            parent.runtime.ledger.cumulative_mechanical_first_law_residual_J
            +mechanical.first_law_residual_J),
        maximum_mechanical_relative_first_law_residual=max(
            parent.runtime.ledger.maximum_mechanical_relative_first_law_residual,
            mechanical.relative_first_law_residual)))
    energy_after = dict(context.energy_kwargs, mean_strain=strain1)
    front_started = time.perf_counter()
    if context.front_enabled:
        state, runtime, fronts = advance_multigrain_front_interval(
            mechanical.state, runtime, kinetics=context.front_kinetics,
            dt_s=dt, maximum_substep_s=context.maximum_front_substep_s,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            wall_parameters=context.wall_parameters,
            energy_kwargs=energy_after,
            applied_shear_rate_s=context.shear_rate_s,
            systems=context.systems,
            heat_deposition_mode=context.heat_deposition_mode)
        if not all(item.accepted and not item.contour_cfl_clipped
                   for item in fronts):
            raise RuntimeError(
                "complete coupled interval contains a rejected or clipped front trial")
    else:
        state = mechanical.state
        fronts = ()
        runtime = MultiGrainProductionRuntime(runtime.interfaces, replace(
            runtime.ledger, intervals=runtime.ledger.intervals+1,
            physical_time_s=runtime.ledger.physical_time_s+dt,
            applied_shear_strain=(runtime.ledger.applied_shear_strain
                                  +context.shear_rate_s*dt),
            stationary_intervals=runtime.ledger.stationary_intervals+1))
    front_wall_time = time.perf_counter()-front_started
    front_sources = [
        np.asarray(item.heat_source_J_by_cell, dtype=float)
        for item in fronts if item.heat_source_J_by_cell is not None]
    front_heat = sum(
        front_sources,
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    if not np.isclose(
            runtime.ledger.physical_time_s-parent.runtime.ledger.physical_time_s,
            dt, rtol=0.0, atol=64*np.finfo(float).eps*max(
                abs(runtime.ledger.physical_time_s),
                abs(parent.runtime.ledger.physical_time_s), dt, 1e-300)):
        raise RuntimeError("complete coupled map failed its physical clock")
    if not np.isclose(runtime.ledger.applied_shear_strain
                      -parent.runtime.ledger.applied_shear_strain,
                      context.shear_rate_s*dt, rtol=0.0,
                      atol=64*np.finfo(float).eps*max(abs(gamma1), 1.0)):
        raise RuntimeError("complete coupled map failed its load clock")
    observable_started = time.perf_counter()
    endpoint = make_endpoint(state, runtime, gamma1, context)
    observable_wall_time = time.perf_counter()-observable_started
    integrals = {
        "mechanical_external_work_J": mechanical.external_work_J,
        "mechanical_internal_energy_change_J": mechanical.internal_energy_change_J,
        "mechanical_irreversible_heat_J": sum(
            item.irreversible_heat_J for item in mechanical.operator_decisions),
        "mechanical_plastic_work_J": sum(
            item.external_plastic_work_J for item in mechanical.operator_decisions),
        "front_heat_J": float(np.sum(front_heat, dtype=np.longdouble)),
        "mechanical_heat_J_m3": mechanical_heat,
        "mechanical_work_J_m3": mechanical_work_density,
        "front_heat_J_by_cell": front_heat,
        "thermal_conduction_J_m3": conduction,
        "thermal_bath_exchange_J_m3": bath,
    }
    audit = {
        "mechanical_subdivision_depth": mechanical.subdivisions,
        "mechanical_internal_substeps": sum(
            item.maximum_substeps for item in mechanical.operator_decisions),
        "mechanical_consumed_interval_s": sum(
            item.consumed_interval_s for item in mechanical.operator_decisions),
        "mechanical_relative_first_law_residual": (
            mechanical.relative_first_law_residual),
        "front_subintervals": len(fronts),
        "front_rejected_subintervals": sum(not item.accepted for item in fronts),
        "front_maximum_unclipped_fraction": max(
            (item.maximum_unclipped_fraction for item in fronts), default=0.0),
        "front_directions": ({} if not fronts else
                             fronts[-1].direction_by_interface),
        "front_selected_velocities_m_s": ({} if not fronts else
                                           fronts[-1].selected_velocity_by_interface_m_s),
        "front_force_rate_closure_classifications": [
            item.force_rate_closure_classification for item in fronts],
        "front_force_rate_closure_brackets": [
            item.force_rate_closure_bracket for item in fronts
            if item.force_rate_closure_bracket is not None],
        "cost_breakdown_s": {
            "mechanical_map": mechanical_wall_time,
            "front_map": front_wall_time,
            "endpoint_observables": observable_wall_time,
            "accounted_total": (mechanical_wall_time+front_wall_time
                                +observable_wall_time),
        },
    }
    return CoupledInterval(endpoint, dt, integrals, audit)
