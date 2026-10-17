"""Conservative neutral-pair emission from explicit GB mobile reservoirs.

This atomic projection is default-off.  It transfers only equal positive and
negative family content from the mobile-channel row of an existing interface
reservoir into its two incident lattice owners.  The emitted pair therefore
has zero net Burgers vector.  Its conjugate grain-spanning slip updates beta_p
and Nye through the authoritative kinematic map.  Complete-energy pricing is
required before a caller may publish the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .tensorial_nye import (
    TensorialKinematicState, accept_slip_increment,
    nye_from_plastic_distortion, plastic_distortion_from_slip,
)
from .arrhenius_dislocation_sources import (
    ArrheniusSourceParameters, hall_petch_nucleation_hazard_Hz,
)
from .common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components,
)
from .complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
    evaluate_complete_multigrain_local_fields,
)
from .multigrain_common_state import reconstruct_multigrain_common
from .multigrain_production import _owner_drivings_from_common_stress


@dataclass(frozen=True)
class InterfaceMobilePairEmissionProjection:
    candidate_state: object
    removed_interface_pair_cell_sum_m2: float
    created_lattice_pair_cell_sum_m2: float
    line_balance_residual_cell_sum_m2: float
    signed_content_change_cell_sum_m2: float
    maximum_mura_relative_residual: float
    emitted_pair_density_m2_by_cell: np.ndarray


@dataclass(frozen=True)
class InterfaceMobilePairEmissionDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    projection: InterfaceMobilePairEmissionProjection
    maximum_kinetic_fraction: float
    resolved_work_J: float
    defect_boundary_storage_change_J: float
    delta_helmholtz_before_heat_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def project_interface_mobile_pair_emission(
        state, *, fraction, resolved_stress_by_owner, systems, spacing_m,
        burgers_m, source_grain_size_m, emission_weight_by_owner=None,
        preserve_owner_mean_orientation=True):
    """Transfer neutral interface pairs to incident mobile populations.

    Interface row zero is the declared mobile-line channel.  At every cell and
    family, at most ``min(boundary_plus, boundary_minus)`` is transferable.
    The intensive owner increment is chosen so its support-weighted extensive
    sum equals the interface debit exactly.
    """
    state.validate()
    spacing = float(spacing_m)
    burgers = float(burgers_m)
    grain_size = float(source_grain_size_m)
    if any(not math.isfinite(x) or x <= 0.0
           for x in (spacing, burgers, grain_size)):
        raise ValueError("spacing, Burgers magnitude, and grain size must be positive")
    stresses = tuple(np.asarray(value, dtype=float)
                     for value in resolved_stress_by_owner)
    if (len(stresses) != len(state.owners)
            or any(value.shape != state.owners[0].mobile_plus_m2.shape
                   or np.any(~np.isfinite(value)) for value in stresses)):
        raise ValueError("resolved stresses require owner x grid x family")
    raw_extent = np.asarray(fraction, dtype=float)
    target_shape = ((len(state.interfaces),)
                    +state.owners[0].mobile_plus_m2.shape)
    if raw_extent.ndim == 0:
        extents = np.full(target_shape, float(raw_extent))
    else:
        try:
            extents = np.broadcast_to(raw_extent, target_shape)
        except ValueError as exc:
            raise ValueError(
                "emission fraction must broadcast to interface x grid x family") from exc
    if np.any(~np.isfinite(extents)) or np.any((extents < 0.0)|(extents > 1.0)):
        raise ValueError("emission fraction must lie in [0,1]")
    if emission_weight_by_owner is None:
        weights = tuple(np.ones_like(value) for value in stresses)
    else:
        weights = tuple(np.asarray(value, dtype=float)
                        for value in emission_weight_by_owner)
        if (len(weights) != len(state.owners)
                or any(value.shape != stresses[0].shape
                       or np.any(~np.isfinite(value)) or np.any(value < 0.0)
                       for value in weights)):
            raise ValueError("emission weights require nonnegative owner x grid x family")

    owner_index = {grain: i for i, grain in enumerate(state.grain_ids)}
    additions = [np.zeros_like(owner.mobile_plus_m2)
                 for owner in state.owners]
    next_interfaces = []
    emitted_by_cell = np.zeros(state.supports.shape[1:])
    removed_pairs = 0.0
    for edge_index, component in enumerate(state.interfaces):
        ia = owner_index[component.grain_a_id]
        ib = owner_index[component.grain_b_id]
        plus = np.asarray(component.boundary_plus_m2, dtype=float)
        minus = np.asarray(component.boundary_minus_m2, dtype=float)
        eta_a = np.asarray(state.supports[ia])[..., None]
        eta_b = np.asarray(state.supports[ib])[..., None]
        weighted_a = eta_a*weights[ia]
        weighted_b = eta_b*weights[ib]
        weight_sum = weighted_a+weighted_b
        active = weight_sum > 256.0*np.finfo(float).eps
        neutral = extents[edge_index]*np.minimum(
            plus[..., 0, :], minus[..., 0, :])
        neutral = np.where(active, neutral, 0.0)
        extensive_a = np.divide(
            neutral*weighted_a, weight_sum, out=np.zeros_like(neutral),
            where=active)
        extensive_b = neutral-extensive_a
        additions[ia] += np.divide(
            extensive_a, eta_a, out=np.zeros_like(neutral), where=eta_a > 0.0)
        additions[ib] += np.divide(
            extensive_b, eta_b, out=np.zeros_like(neutral), where=eta_b > 0.0)
        next_plus = plus.copy()
        next_minus = minus.copy()
        next_plus[..., 0, :] -= neutral
        next_minus[..., 0, :] -= neutral
        next_interfaces.append(replace(
            component, boundary_plus_m2=next_plus,
            boundary_minus_m2=next_minus))
        removed_pairs += float(np.sum(neutral, dtype=np.longdouble))
        emitted_by_cell += np.sum(neutral, axis=2)

    if removed_pairs == 0.0:
        return InterfaceMobilePairEmissionProjection(
            state, 0.0, 0.0, 0.0, 0.0, 0.0, emitted_by_cell)

    owners = []
    created_pairs = 0.0
    signed_change = 0.0
    maximum_mura = 0.0
    for support, owner, stress, addition in zip(
            state.supports, state.owners, stresses, additions):
        created_pairs += float(np.sum(
            np.asarray(support)[..., None]*addition,
            dtype=np.longdouble))
        # A pair density 1/d^2 corresponds to one grain-spanning source event;
        # the conjugate shear per event is b/d.
        slip_increment = np.sign(stress)*addition*burgers*grain_size
        kinematics = TensorialKinematicState(
            np.asarray(owner.slip), np.asarray(owner.beta_p),
            np.asarray(owner.alignment_m2), np.asarray(owner.family_nye_m1))
        updated = accept_slip_increment(
            kinematics, slip_increment, systems,
            np.asarray(owner.orientation_rad), spacing)
        dbeta = updated.beta_p-np.asarray(owner.beta_p)
        orientation = (np.asarray(owner.orientation_rad)
                       +.5*(dbeta[..., 1, 0]-dbeta[..., 0, 1]))
        if preserve_owner_mean_orientation:
            orientation += (float(np.mean(owner.orientation_rad))
                            -float(np.mean(orientation)))
        next_owner = replace(
            owner,
            mobile_plus_m2=np.asarray(owner.mobile_plus_m2)+addition,
            mobile_minus_m2=np.asarray(owner.mobile_minus_m2)+addition,
            slip=updated.slip, beta_p=updated.beta_p,
            family_nye_m1=updated.family_nye_m1,
            alignment_m2=updated.alignment_m2,
            orientation_rad=orientation)
        owners.append(next_owner)
        signed_change += float(np.sum(
            np.asarray(support)[..., None]*(
                (next_owner.mobile_plus_m2-next_owner.mobile_minus_m2)
                -(owner.mobile_plus_m2-owner.mobile_minus_m2)),
            dtype=np.longdouble))
        beta_nye_increment = nye_from_plastic_distortion(dbeta, spacing)
        # Audit the manufactured accepted increment directly. Subtracting the
        # updated and retained large Nye tensors loses several digits on mature
        # states even though accept_slip_increment constructs this identity
        # family by family.
        expected_family_increment = np.stack([
            nye_from_plastic_distortion(
                plastic_distortion_from_slip(
                    np.where(
                        np.arange(len(systems))[None, None, :] == family,
                        slip_increment, 0.0),
                    systems, np.asarray(owner.orientation_rad)),
                spacing)
            for family in range(len(systems))], axis=2)
        mura_residual = (np.sum(
            expected_family_increment.astype(np.longdouble), axis=2,
            dtype=np.longdouble)-beta_nye_increment.astype(np.longdouble))
        mura_scale = max(float(np.sqrt(np.mean(
            beta_nye_increment*beta_nye_increment))), 1.0)
        maximum_mura = max(
            maximum_mura,
            float(np.sqrt(np.mean(mura_residual*mura_residual))/mura_scale))

    candidate = replace(
        state, owners=tuple(owners), interfaces=tuple(next_interfaces))
    candidate.validate()
    line_balance = 2.0*(created_pairs-removed_pairs)
    return InterfaceMobilePairEmissionProjection(
        candidate, removed_pairs, created_pairs, line_balance, signed_change,
        maximum_mura, emitted_by_cell)


def evaluate_interface_mobile_pair_emission(
        state, *, mean_strain, interval_s, systems, wall_parameters,
        represented_thickness_m, source_parameters=ArrheniusSourceParameters(),
        energy_kwargs=None, maximum_fraction_per_interval=.05,
        maximum_relative_first_law_residual=.05,
        maximum_mura_relative_residual=1.0e-10):
    """Price one Hall--Petch-activated neutral GB-reservoir emission event."""
    dt = float(interval_s)
    maximum_fraction = float(maximum_fraction_per_interval)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("emission interval must be positive finite")
    if (not math.isfinite(maximum_fraction)
            or not 0.0 < maximum_fraction <= 1.0):
        raise ValueError("maximum emission fraction must lie in (0,1]")
    if (not math.isfinite(float(maximum_mura_relative_residual))
            or float(maximum_mura_relative_residual) <= 0.0):
        raise ValueError("Mura tolerance must be positive finite")
    spacing = float(wall_parameters.spacing_m)
    thickness = float(represented_thickness_m)
    cell_volume = spacing*spacing*thickness
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common_energy = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before_energy = evaluate_complete_multigrain_energy(
        state, **common_energy)
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=np.asarray(mean_strain)),
        systems, wall_parameters)
    raw_stresses = tuple(resolved_driving_components(
        owner, driving, systems, (), wall_parameters)["raw_stress_Pa"]
        for owner, driving in zip(state.owners, drivings))
    hazards = tuple(hall_petch_nucleation_hazard_Hz(
        stress, np.asarray(owner.temperature_K)[..., None], source_parameters)
        for owner, stress in zip(state.owners, raw_stresses))
    owner_index = {grain: i for i, grain in enumerate(state.grain_ids)}
    fractions = []
    for component in state.interfaces:
        ia = owner_index[component.grain_a_id]
        ib = owner_index[component.grain_b_id]
        eta_a = np.asarray(state.supports[ia])[..., None]
        eta_b = np.asarray(state.supports[ib])[..., None]
        support = eta_a+eta_b
        rate = np.divide(
            eta_a*hazards[ia]+eta_b*hazards[ib], support,
            out=np.zeros_like(hazards[ia]), where=support > 0.0)
        fractions.append(np.minimum(-np.expm1(-rate*dt), maximum_fraction))
    fraction_field = np.asarray(fractions)
    projection = project_interface_mobile_pair_emission(
        state, fraction=fraction_field,
        resolved_stress_by_owner=raw_stresses, systems=systems,
        spacing_m=spacing, burgers_m=source_parameters.burgers_m,
        source_grain_size_m=source_parameters.grain_size_m,
        emission_weight_by_owner=hazards,
        preserve_owner_mean_orientation=True)
    if projection.removed_interface_pair_cell_sum_m2 == 0.0:
        zeros = np.zeros(state.supports.shape[1:])
        return InterfaceMobilePairEmissionDecision(
            False, "INACTIVE_NO_NEUTRAL_INTERFACE_CAPACITY", state, state,
            projection, float(np.max(fraction_field, initial=0.0)),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zeros)

    unheated = projection.candidate_state
    local_options = {
        key: value for key, value in options.items()
        if key in {"topologies", "systems", "boundary_line_energy_J_m",
                   "boundary_junction_energy_J_m"}}
    before_local = evaluate_complete_multigrain_local_fields(
        state, spacing_m=spacing, wall_parameters=wall_parameters,
        mean_strain=np.asarray(mean_strain), **local_options)
    after_local = evaluate_complete_multigrain_local_fields(
        unheated, spacing_m=spacing, wall_parameters=wall_parameters,
        mean_strain=np.asarray(mean_strain), **local_options)
    storage_change_density = (
        after_local.defect_storage_J_m3-before_local.defect_storage_J_m3
        +after_local.signed_junction_storage_J_m3
        -before_local.signed_junction_storage_J_m3
        +after_local.boundary_excess_J_m3-before_local.boundary_excess_J_m3)
    work_density = np.zeros(state.supports.shape[1:])
    for support, before, after, stress in zip(
            state.supports, state.owners, unheated.owners, raw_stresses):
        work_density += np.asarray(support)*np.sum(
            stress*(np.asarray(after.slip)-np.asarray(before.slip)), axis=2)
    raw_heat_density = work_density-storage_change_density
    local_scale = np.maximum.reduce((
        np.abs(work_density), np.abs(storage_change_density),
        np.ones_like(work_density)))
    raw_heat_sum = float(np.sum(raw_heat_density, dtype=np.longdouble))
    heat_valid = bool(
        raw_heat_sum >= -1e-12*float(np.sum(local_scale, dtype=np.longdouble)))
    positive_work = np.maximum(work_density, 0.0)
    positive_work_sum = float(np.sum(positive_work, dtype=np.longdouble))
    heat_density = np.zeros_like(raw_heat_density)
    if heat_valid and raw_heat_sum > 0.0 and positive_work_sum > 0.0:
        heat_density = raw_heat_sum*positive_work/positive_work_sum
    common, _ = reconstruct_multigrain_common(unheated, spacing)
    temperature = (np.asarray(common.temperature_K)
                   +heat_density/wall_parameters.volumetric_heat_capacity_J_m3_K)
    candidate = replace(unheated, owners=tuple(
        replace(owner, temperature_K=temperature.copy())
        for owner in unheated.owners))
    candidate.validate()
    unheated_energy = evaluate_complete_multigrain_energy(
        unheated, **common_energy)
    after_energy = evaluate_complete_multigrain_energy(
        candidate, **common_energy)
    work_J = float(np.sum(work_density, dtype=np.longdouble)*cell_volume)
    storage_J = float(np.sum(
        storage_change_density, dtype=np.longdouble)*cell_volume)
    heat_J = float(np.sum(heat_density, dtype=np.longdouble)*cell_volume)
    delta_helmholtz = unheated_energy.helmholtz_J-before_energy.helmholtz_J
    residual = after_energy.internal_J-before_energy.internal_J
    scale = max(abs(work_J), abs(storage_J), abs(delta_helmholtz),
                abs(heat_J), 1e-300)
    relative = abs(residual)/scale
    balance_scale = max(
        projection.removed_interface_pair_cell_sum_m2, 1.0)
    balances_close = bool(
        abs(projection.line_balance_residual_cell_sum_m2)
        <= 1e-11*balance_scale
        and abs(projection.signed_content_change_cell_sum_m2)
        <= 1e-11*balance_scale
        and projection.maximum_mura_relative_residual
        < float(maximum_mura_relative_residual))
    downhill = delta_helmholtz <= 1e-12*scale
    accepted = bool(
        balances_close and heat_valid and downhill
        and relative <= maximum_relative_first_law_residual)
    if not balances_close:
        classification = "REJECTED_LINE_BURGERS_OR_MURA_BALANCE"
    elif not heat_valid:
        classification = "REJECTED_INSUFFICIENT_RESOLVED_WORK"
    elif not downhill:
        classification = "REJECTED_UPHILL_COMPLETE_HELMHOLTZ"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_COMPLETE_FIRST_LAW"
    else:
        classification = "ACCEPTED_INTERFACE_MOBILE_PAIR_EMISSION"
    return InterfaceMobilePairEmissionDecision(
        accepted, classification, candidate, candidate if accepted else state,
        projection, float(np.max(fraction_field, initial=0.0)),
        work_J, storage_J, delta_helmholtz, heat_J, residual, relative,
        heat_density*cell_volume)
