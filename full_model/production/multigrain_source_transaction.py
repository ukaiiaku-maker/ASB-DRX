"""Complete-energy Hall--Petch pair-source transaction for common owners."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .arrhenius_dislocation_sources import apply_conjugate_pair_source_event
from .common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components,
    wall_total_free_energy_density_J_m3,
)
from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .multigrain_common_state import reconstruct_multigrain_common
from .multigrain_production import (
    _masked_owner_update, _owner_drivings_from_common_stress,
)


@dataclass(frozen=True)
class MultiGrainSourceDecision:
    accepted: bool
    classification: str
    candidate_state: object
    event_count: float
    created_line_cell_sum_m2: float
    signed_content_change_m2: float
    line_balance_residual_m2: float
    resolved_work_J: float
    defect_storage_change_J: float
    generated_heat_J: float
    delta_helmholtz_before_heat_J: float
    internal_energy_change_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    energy_resolution_floor_J: float
    heat_source_J_by_cell: np.ndarray


def evaluate_multigrain_pair_source_transaction(
        state, *, mean_strain, site_field, interval_s, source_parameters,
        systems, wall_parameters, represented_thickness_m,
        energy_kwargs=None, source_mode="unspecified",
        maximum_relative_first_law_residual=.05):
    """Propose and qualify one coupled neutral-pair/source-slip event.

    The event is performed at fixed total strain, so external boundary work is
    zero. ``tau * delta_gamma`` is an independently evaluated release of
    recoverable elastic work. Line-storage increase is subtracted locally; the
    nonnegative remainder is deposited as heat. The complete common energy is
    then evaluated without using its residual to define that heat.
    """
    dt = float(interval_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("source interval must be positive and finite")
    site = np.asarray(site_field, dtype=float)
    if site.shape != state.supports.shape[1:]:
        raise ValueError("source site field has wrong grid")
    spacing = float(wall_parameters.spacing_m)
    thickness = float(represented_thickness_m)
    cell_volume = spacing*spacing*thickness
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common_energy = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before_energy = evaluate_complete_multigrain_energy(state, **common_energy)
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=np.asarray(mean_strain)),
        systems, wall_parameters)
    proposed_owners = []
    ledgers = []
    work_density = np.zeros(site.shape)
    defect_change_density = np.zeros(site.shape)
    for support, owner, driving in zip(state.supports, state.owners, drivings):
        components = resolved_driving_components(
            owner, driving, systems, (), wall_parameters)
        proposed, ledger = apply_conjugate_pair_source_event(
            owner, site, components["raw_stress_Pa"], dt, systems,
            source_parameters, spacing_m=spacing, source_mode=source_mode,
            orientation_spin_weight=wall_parameters.orientation_spin_weight,
            active_mask=np.asarray(support) > 256*np.finfo(float).eps)
        proposed = _masked_owner_update(
            owner, proposed,
            np.asarray(support) > 256*np.finfo(float).eps,
            systems, spacing)
        proposed_owners.append(proposed)
        ledgers.append(ledger)
        weight = np.asarray(support)
        work_density += weight*ledger.resolved_work_density_J_m3
        before_defect = wall_total_free_energy_density_J_m3(
            owner, wall_parameters, (), systems)
        after_defect = wall_total_free_energy_density_J_m3(
            proposed, wall_parameters, (), systems)
        defect_change_density += weight*(after_defect-before_defect)

    unheated = replace(state, owners=tuple(proposed_owners))
    unheated.validate()
    unheated_energy = evaluate_complete_multigrain_energy(
        unheated, **common_energy)
    heat_density = work_density-defect_change_density
    local_scale = np.maximum.reduce((
        np.abs(work_density), np.abs(defect_change_density),
        np.ones_like(work_density)))
    local_heat_valid = bool(np.all(
        heat_density >= -1e-12*local_scale))
    heat_density = np.maximum(heat_density, 0.0)
    common_before, _ = reconstruct_multigrain_common(unheated, spacing)
    temperature = (np.asarray(common_before.temperature_K)
                   +heat_density/wall_parameters.volumetric_heat_capacity_J_m3_K)
    heated_owners = tuple(
        replace(owner, temperature_K=temperature.copy())
        for owner in unheated.owners)
    candidate = replace(unheated, owners=heated_owners)
    candidate.validate()
    candidate_energy = evaluate_complete_multigrain_energy(
        candidate, **common_energy)

    work_J = float(np.sum(work_density, dtype=np.longdouble)*cell_volume)
    defect_J = float(np.sum(
        defect_change_density, dtype=np.longdouble)*cell_volume)
    heat_J = float(np.sum(heat_density, dtype=np.longdouble)*cell_volume)
    delta_helmholtz = unheated_energy.helmholtz_J-before_energy.helmholtz_J
    delta_internal = candidate_energy.internal_J-before_energy.internal_J
    residual = delta_internal  # fixed total strain: external boundary work is zero
    scale = max(abs(work_J), abs(defect_J), abs(heat_J),
                abs(delta_helmholtz), 1e-300)
    resolution_floor = (4096.0*np.finfo(float).eps
                        *max(abs(before_energy.internal_J),
                             abs(candidate_energy.internal_J), 1e-300))
    resolved = scale >= resolution_floor
    relative = abs(residual)/scale
    event_count = float(sum(np.sum(item.events_by_family, dtype=np.longdouble)
                            for item in ledgers))
    line_change = float(sum(item.line_content_change_m2 for item in ledgers))
    signed_change = float(sum(item.signed_content_change_m2 for item in ledgers))
    line_residual = float(sum(item.line_balance_residual_m2 for item in ledgers))
    downhill = delta_helmholtz <= 1e-12*scale
    accepted = bool(
        resolved and local_heat_valid and downhill
        and relative <= float(maximum_relative_first_law_residual))
    if not resolved:
        classification = "INACTIVE_BELOW_ENERGY_RESOLUTION"
    elif not local_heat_valid:
        classification = "REJECTED_INSUFFICIENT_RESOLVED_WORK"
    elif not downhill:
        classification = "REJECTED_UPHILL_COMPLETE_HELMHOLTZ"
    elif relative > float(maximum_relative_first_law_residual):
        classification = "REJECTED_FIRST_LAW_CLOSURE"
    else:
        classification = "ACCEPTED_CONJUGATE_PAIR_SOURCE"
    return MultiGrainSourceDecision(
        accepted, classification, candidate if accepted else state,
        event_count, line_change, signed_change, line_residual,
        work_J, defect_J, heat_J, delta_helmholtz, delta_internal,
        residual, relative, resolution_floor, heat_density*cell_volume)
