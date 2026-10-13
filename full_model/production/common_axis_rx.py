"""Common-state absorption--rotation operator for a planar tilt boundary.

This is a production-callable restricted crystallographic operator.  It couples
the authoritative ``CommonWallState`` to one planar, common-axis low-angle
boundary.  Compatible signed lattice line is removed from a declared capture
band, stored as structural boundary content, and the adjacent region's
orientation is derived from ``theta=b*N``.  It cannot allocate a grain label.

The complete transaction includes the exact common defect energy, one
Read--Shockley intrinsic boundary energy, extrinsic line energy, and a declared
quadratic accommodation energy.  Isotropic elastic energy is unchanged by the
pure orientation update at fixed beta_p and strain; anisotropic accommodation
is represented by the explicit comparator stiffness until a full tensorial
boundary solver replaces this common-axis reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .arrhenius_dislocation_sources import apply_sign_balanced_pair_source
from .common_tensorial_wall import (
    accepted_euler_step,
    wall_total_free_energy_density_J_m3,
)
from .structural_boundary import (
    StructuralBoundaryParameters,
    exp_floor_event_rate_Hz,
    read_shockley_energy_J_m2,
)


@dataclass(frozen=True)
class CommonAxisBoundaryState:
    fixed_orientation_rad: float
    reference_misorientation_rad: float
    structural_signed_m1: float
    extrinsic_plus_m1: float = 0.0
    extrinsic_minus_m1: float = 0.0
    exposure_s: float = 0.0
    cumulative_dissipation_J_m2: float = 0.0

    def validate(self, parameters):
        values = (
            self.fixed_orientation_rad, self.reference_misorientation_rad,
            self.structural_signed_m1, self.extrinsic_plus_m1,
            self.extrinsic_minus_m1, self.exposure_s,
            self.cumulative_dissipation_J_m2,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("boundary state must be finite")
        if any(value < 0.0 for value in (
                self.reference_misorientation_rad, self.extrinsic_plus_m1,
                self.extrinsic_minus_m1, self.exposure_s,
                self.cumulative_dissipation_J_m2)):
            raise ValueError("unsigned boundary state must be nonnegative")
        theta = abs(parameters.burgers_m*self.structural_signed_m1)
        if theta > parameters.transition_misorientation_rad*(1.0+1e-12):
            raise ValueError("common-axis reduction is restricted to LAGB content")


@dataclass(frozen=True)
class CommonAxisRXDecision:
    common_state: object
    boundary_state: CommonAxisBoundaryState
    accepted: bool
    classification: str
    attempted_absorbed_plus_m1: float
    attempted_absorbed_minus_m1: float
    accepted_fraction: float
    structural_change_m1: float
    extrinsic_change_m1: float
    annihilated_line_m1: float
    line_balance_residual_m1: float
    delta_bulk_defect_J_m2: float
    delta_intrinsic_boundary_J_m2: float
    delta_extrinsic_J_m2: float
    delta_accommodation_J_m2: float
    delta_helmholtz_J_m2: float
    dissipation_J_m2: float
    event_rate_Hz: float


@dataclass(frozen=True)
class CommonAxisRXInterval:
    common_state: object
    boundary_state: CommonAxisBoundaryState
    source_ledger: object | None
    mechanical_residual: object
    mechanical_acceptance_scale: float
    absorption_decision: CommonAxisRXDecision


def _boundary_energy(state, parameters, accommodation_stiffness_J_m2_rad2):
    theta = parameters.burgers_m*state.structural_signed_m1
    intrinsic = read_shockley_energy_J_m2(theta, parameters)
    extrinsic = parameters.extrinsic_line_energy_J_m*(
        state.extrinsic_plus_m1+state.extrinsic_minus_m1)
    accommodation = .5*float(accommodation_stiffness_J_m2_rad2)*(
        theta-state.reference_misorientation_rad)**2
    return intrinsic, extrinsic, accommodation


def _bulk_energy_per_boundary_area(common, wall_parameters, topologies,
                                   systems, spacing_m, boundary_length_m):
    density = wall_total_free_energy_density_J_m3(
        common, wall_parameters, topologies, systems)
    # Unit represented thickness cancels between cell volume and boundary area.
    return float(np.sum(density, dtype=np.longdouble)
                 *float(spacing_m)**2/float(boundary_length_m))


def _candidate(common, boundary, *, absorbed_plus_m1, absorbed_minus_m1,
               capture_mask, rotating_mask, family, compatible, parameters,
               spacing_m, capture_width_m):
    plus = float(absorbed_plus_m1)
    minus = float(absorbed_minus_m1)
    area_cells = int(np.count_nonzero(capture_mask))
    if area_cells == 0:
        raise ValueError("capture mask is empty")
    # Uniform density decrement in the capture band has integrated line per
    # boundary area deltaN = delta_rho*capture_width.
    decrement_plus = plus/float(capture_width_m)
    decrement_minus = minus/float(capture_width_m)
    mp = np.asarray(common.mobile_plus_m2).copy()
    mm = np.asarray(common.mobile_minus_m2).copy()
    mp[..., family][capture_mask] -= decrement_plus
    mm[..., family][capture_mask] -= decrement_minus
    if np.any(mp < -1e-6) or np.any(mm < -1e-6):
        raise ValueError("absorption exceeds capture-band content")
    mp = np.maximum(mp, 0.0); mm = np.maximum(mm, 0.0)

    structural_before = boundary.structural_signed_m1
    extra_plus = boundary.extrinsic_plus_m1
    extra_minus = boundary.extrinsic_minus_m1
    annihilated = 0.0
    if compatible:
        incoming = plus-minus
        structural_after = structural_before+incoming
        neutral = min(plus, minus)
        extra_plus += neutral; extra_minus += neutral
        annihilated = max(
            abs(structural_before)+abs(incoming)-abs(structural_after), 0.0)
    else:
        structural_after = structural_before
        extra_plus += plus; extra_minus += minus
    theta_change = parameters.burgers_m*(structural_after-structural_before)
    orientation = np.asarray(common.orientation_rad).copy()
    # Superpose the boundary-mediated rigid rotation on any compatible plastic
    # spin already accumulated inside the region.  Resetting the absolute
    # orientation would erase production kinematics and hide its energy cost.
    orientation[rotating_mask] += theta_change
    candidate_common = replace(
        common, mobile_plus_m2=mp, mobile_minus_m2=mm,
        orientation_rad=orientation)
    candidate_boundary = replace(
        boundary, structural_signed_m1=structural_after,
        extrinsic_plus_m1=extra_plus, extrinsic_minus_m1=extra_minus)
    return candidate_common, candidate_boundary, annihilated


def advance_common_axis_absorption(
        common, boundary, *, capture_mask, rotating_mask, family,
        compatible, parameters: StructuralBoundaryParameters,
        wall_parameters, systems, topologies=(), dt_s,
        accommodation_stiffness_J_m2_rad2=0.0,
        maximum_reservoir_fraction=0.05, maximum_backtracks=20,
        energy_tolerance_J_m2=1e-13):
    """Advance one energy-qualified stationary absorption reaction."""
    boundary.validate(parameters)
    dt = float(dt_s)
    spacing = float(wall_parameters.spacing_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite interval required")
    capture = np.asarray(capture_mask, dtype=bool)
    rotating = np.asarray(rotating_mask, dtype=bool)
    if capture.shape != common.orientation_rad.shape or rotating.shape != capture.shape:
        raise ValueError("capture and rotating masks must match the common grid")
    if not 0 <= int(family) < common.mobile_plus_m2.shape[2]:
        raise ValueError("invalid Burgers family")
    fraction_cap = float(maximum_reservoir_fraction)
    if not 0.0 < fraction_cap <= 1.0:
        raise ValueError("maximum reservoir fraction must lie in (0,1]")
    # For a straight vertical band, area/length gives its physical width and
    # remains valid for a mask spanning a noninteger number of rows.
    boundary_length = common.orientation_rad.shape[1]*spacing
    capture_width = np.count_nonzero(capture)*spacing**2/boundary_length
    if capture_width <= 0.0:
        raise ValueError("capture mask is empty")
    plus_capacity = float(np.min(common.mobile_plus_m2[..., family][capture]))*capture_width
    minus_capacity = float(np.min(common.mobile_minus_m2[..., family][capture]))*capture_width
    maximum_plus = fraction_cap*plus_capacity
    maximum_minus = fraction_cap*minus_capacity

    before_bulk = _bulk_energy_per_boundary_area(
        common, wall_parameters, topologies, systems, spacing, boundary_length)
    before_parts = _boundary_energy(
        boundary, parameters, accommodation_stiffness_J_m2_rad2)

    # A small exact transaction supplies the line affinity consumed by the
    # immutable EXP-floor kinetics.  No rate is inferred from a final residual.
    probe_plus = max(maximum_plus*1e-4, 0.0)
    probe_minus = max(maximum_minus*1e-4, 0.0)
    if probe_plus+probe_minus == 0.0:
        return CommonAxisRXDecision(
            common, boundary, False, "NO_COMPATIBLE_RESERVOIR", 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    probe_common, probe_boundary, _ = _candidate(
        common, boundary, absorbed_plus_m1=probe_plus,
        absorbed_minus_m1=probe_minus, capture_mask=capture,
        rotating_mask=rotating, family=family, compatible=compatible,
        parameters=parameters, spacing_m=spacing,
        capture_width_m=capture_width)
    probe_bulk = _bulk_energy_per_boundary_area(
        probe_common, wall_parameters, topologies, systems, spacing,
        boundary_length)
    probe_parts = _boundary_energy(
        probe_boundary, parameters, accommodation_stiffness_J_m2_rad2)
    probe_delta = probe_bulk-before_bulk+sum(
        after-before for after, before in zip(probe_parts, before_parts))
    affinity = -probe_delta/max(probe_plus+probe_minus, 1e-300)
    temperature = float(np.mean(common.temperature_K[capture]))
    rate = exp_floor_event_rate_Hz(affinity, temperature, parameters)
    kinetic_fraction = min(1.0-math.exp(-rate*dt), 1.0)
    attempted_plus = kinetic_fraction*maximum_plus
    attempted_minus = kinetic_fraction*maximum_minus

    extent = 1.0
    for _ in range(int(maximum_backtracks)+1):
        plus = extent*attempted_plus
        minus = extent*attempted_minus
        candidate_common, candidate_boundary, annihilated = _candidate(
            common, boundary, absorbed_plus_m1=plus,
            absorbed_minus_m1=minus, capture_mask=capture,
            rotating_mask=rotating, family=family, compatible=compatible,
            parameters=parameters, spacing_m=spacing,
            capture_width_m=capture_width)
        theta = abs(parameters.burgers_m*candidate_boundary.structural_signed_m1)
        if theta > parameters.transition_misorientation_rad:
            extent *= .5
            continue
        candidate_bulk = _bulk_energy_per_boundary_area(
            candidate_common, wall_parameters, topologies, systems, spacing,
            boundary_length)
        candidate_parts = _boundary_energy(
            candidate_boundary, parameters,
            accommodation_stiffness_J_m2_rad2)
        components = (
            candidate_bulk-before_bulk,
            candidate_parts[0]-before_parts[0],
            candidate_parts[1]-before_parts[1],
            candidate_parts[2]-before_parts[2],
        )
        delta = sum(components)
        if delta <= float(energy_tolerance_J_m2):
            structural_change = (candidate_boundary.structural_signed_m1
                                 -boundary.structural_signed_m1)
            extrinsic_change = (
                candidate_boundary.extrinsic_plus_m1
                +candidate_boundary.extrinsic_minus_m1
                -boundary.extrinsic_plus_m1-boundary.extrinsic_minus_m1)
            balance = (plus+minus-(
                abs(candidate_boundary.structural_signed_m1)
                -abs(boundary.structural_signed_m1))
                -extrinsic_change-annihilated)
            candidate_boundary = replace(
                candidate_boundary, exposure_s=boundary.exposure_s+dt,
                cumulative_dissipation_J_m2=(
                    boundary.cumulative_dissipation_J_m2+max(-delta, 0.0)))
            classification = ("COMPATIBLE_ABSORPTION_ROTATION" if compatible
                              and abs(structural_change) > 0.0 else
                              "EXTRINSIC_ABSORPTION")
            return CommonAxisRXDecision(
                candidate_common, candidate_boundary, True, classification,
                attempted_plus, attempted_minus, extent, structural_change,
                extrinsic_change, annihilated, balance, *components, delta,
                max(-delta, 0.0), rate)
        extent *= .5
    return CommonAxisRXDecision(
        common, boundary, False, "ENERGY_REJECTED", attempted_plus,
        attempted_minus, 0.0, 0.0, 0.0, 0.0, 0.0,
        probe_bulk-before_bulk, probe_parts[0]-before_parts[0],
        probe_parts[1]-before_parts[1], probe_parts[2]-before_parts[2],
        probe_delta, 0.0, rate)


def advance_common_axis_rx_interval(
        common, boundary, *, driving, systems, topologies,
        wall_parameters, structural_parameters, capture_mask, rotating_mask,
        family, compatible, dt_s, pair_source_rate_m2_s=None,
        source_mode="disabled", source_family_weights=None,
        accommodation_stiffness_J_m2_rad2=0.0,
        maximum_reservoir_fraction=0.05):
    """Compose source, common mechanics/heat, and structural reaction once.

    This is the V66 production integration point.  All three suboperators act
    on the same published common state and physical interval.  A disabled
    source leaves the state unchanged.  The absorption transaction publishes
    only when its complete reduced Helmholtz change is nonpositive.
    """
    source_ledger = None
    sourced = common
    if pair_source_rate_m2_s is not None:
        sourced, source_ledger = apply_sign_balanced_pair_source(
            common, pair_source_rate_m2_s, dt_s,
            family_weights=source_family_weights, source_mode=source_mode)
    mechanical, residual, scale = accepted_euler_step(
        sourced, driving, systems, topologies, wall_parameters, dt_s)
    absorption = advance_common_axis_absorption(
        mechanical, boundary, capture_mask=capture_mask,
        rotating_mask=rotating_mask, family=family, compatible=compatible,
        parameters=structural_parameters, wall_parameters=wall_parameters,
        systems=systems, topologies=topologies, dt_s=dt_s,
        accommodation_stiffness_J_m2_rad2=(
            accommodation_stiffness_J_m2_rad2),
        maximum_reservoir_fraction=maximum_reservoir_fraction)
    return CommonAxisRXInterval(
        absorption.common_state, absorption.boundary_state, source_ledger,
        residual, scale, absorption)
