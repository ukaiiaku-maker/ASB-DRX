"""Energy-complete low-angle structural-boundary reactions.

This module is the V66 reduced physics kernel.  It deliberately does not create
a grain label.  For a straight tilt boundary, compatible signed line content
``N`` (m^-1) determines the small misorientation through ``theta = b N``.
That structural content is represented once by the Read--Shockley energy.
Sign-balanced or crystallographically incompatible absorbed content is stored
as extrinsic line content and is represented once by its line energy.

The scalar reduction is not a general Frank--Bilby solver.  It is an exact,
unit-testable limit used to decide whether absorption--rotation and conservative
network redistribution are thermodynamically viable before tensorial production
integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StructuralBoundaryParameters:
    burgers_m: float = 2.48e-10
    transition_misorientation_rad: float = math.radians(15.0)
    high_angle_energy_J_m2: float = 0.5
    bulk_line_energy_J_m: float = 1.5e-9
    bulk_correlation_energy_J_m: float = 8.0e-11
    reference_density_m2: float = 1.0e12
    extrinsic_line_energy_J_m: float = 1.5e-9
    reservoir_width_m: float = 5.0e-7
    activation_enthalpy_eV: float = 1.50
    barrier_floor_eV: float = 0.05
    attempt_frequency_Hz: float = 1.0e12
    activation_segment_m: float = 2.48e-10

    def __post_init__(self):
        positive = (
            "burgers_m", "transition_misorientation_rad",
            "high_angle_energy_J_m2", "bulk_line_energy_J_m",
            "bulk_correlation_energy_J_m", "reference_density_m2",
            "extrinsic_line_energy_J_m", "reservoir_width_m",
            "activation_enthalpy_eV", "barrier_floor_eV",
            "attempt_frequency_Hz", "activation_segment_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.barrier_floor_eV > self.activation_enthalpy_eV:
            raise ValueError("barrier floor cannot exceed activation enthalpy")


@dataclass(frozen=True)
class StructuralBoundaryState:
    """Straight-boundary state per unit boundary area.

    ``bulk_*_m2`` are compatible signed lattice reservoirs within the declared
    reservoir width. ``structural_signed_m1`` is the net compatible boundary
    content. ``extrinsic_*_m1`` contains absorbed line not used to define the
    boundary misorientation.
    """

    bulk_plus_m2: float
    bulk_minus_m2: float
    structural_signed_m1: float
    extrinsic_plus_m1: float = 0.0
    extrinsic_minus_m1: float = 0.0

    def validate(self):
        nonnegative = (
            self.bulk_plus_m2, self.bulk_minus_m2,
            self.extrinsic_plus_m1, self.extrinsic_minus_m1,
        )
        if any(not math.isfinite(float(value)) or value < 0.0
               for value in nonnegative):
            raise ValueError("density and extrinsic content must be finite and nonnegative")
        if not math.isfinite(float(self.structural_signed_m1)):
            raise ValueError("structural content must be finite")


@dataclass(frozen=True)
class AbsorptionLedger:
    before: StructuralBoundaryState
    candidate: StructuralBoundaryState
    absorbed_plus_m1: float
    absorbed_minus_m1: float
    structural_change_m1: float
    extrinsic_line_change_m1: float
    annihilated_line_m1: float
    line_balance_residual_m1: float
    energy_before_J_m2: float
    energy_candidate_J_m2: float
    delta_helmholtz_J_m2: float
    accepted: bool
    classification: str


@dataclass(frozen=True)
class NetworkState:
    """Three orientations separated by two compatible tilt boundaries."""

    orientation_rad: tuple[float, float, float]

    def validate(self):
        values = tuple(float(value) for value in self.orientation_rad)
        if len(values) != 3 or any(not math.isfinite(value) for value in values):
            raise ValueError("three finite orientations are required")


@dataclass(frozen=True)
class NetworkTransferLedger:
    before: NetworkState
    candidate: NetworkState
    transferred_structural_m1: float
    structural_sum_before_m1: float
    structural_sum_candidate_m1: float
    balance_residual_m1: float
    energy_before_J_m2: float
    energy_candidate_J_m2: float
    delta_helmholtz_J_m2: float
    accepted: bool


def read_shockley_energy_J_m2(theta_rad, parameters):
    """Read--Shockley energy, continuously capped at the HAGB transition."""
    theta = np.abs(np.asarray(theta_rad, dtype=float))
    if np.any(~np.isfinite(theta)):
        raise ValueError("misorientation must be finite")
    theta_m = parameters.transition_misorientation_rad
    ratio = np.minimum(theta/theta_m, 1.0)
    value = np.zeros_like(ratio)
    positive = ratio > 0.0
    value[positive] = parameters.high_angle_energy_J_m2*ratio[positive]*(
        1.0-np.log(ratio[positive]))
    return float(value) if value.ndim == 0 else value


def read_shockley_slope_J_m2_rad(theta_rad, parameters):
    """One-sided magnitude derivative below the HAGB transition."""
    theta = np.abs(np.asarray(theta_rad, dtype=float))
    if np.any(~np.isfinite(theta)) or np.any(theta <= 0.0):
        raise ValueError("strictly positive finite misorientation required")
    theta_m = parameters.transition_misorientation_rad
    ratio = theta/theta_m
    value = np.where(
        ratio < 1.0,
        -parameters.high_angle_energy_J_m2/theta_m*np.log(ratio),
        0.0,
    )
    return float(value) if value.ndim == 0 else value


def bulk_energy_density_J_m3(total_density_m2, parameters):
    """Production common line plus positive rho-log-rho correlation branch."""
    rho = np.asarray(total_density_m2, dtype=float)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("bulk density must be finite and nonnegative")
    out = parameters.bulk_line_energy_J_m*rho
    positive = rho > 0.0
    out = np.asarray(out)
    out[positive] += parameters.bulk_correlation_energy_J_m*rho[positive]*np.log(
        rho[positive]/parameters.reference_density_m2)
    return float(out) if out.ndim == 0 else out


def boundary_misorientation_rad(state, parameters):
    state.validate()
    return parameters.burgers_m*state.structural_signed_m1


def structural_boundary_energy_J_m2(state, parameters):
    """Complete reduced Helmholtz energy per unit boundary area."""
    state.validate()
    bulk = parameters.reservoir_width_m*bulk_energy_density_J_m3(
        state.bulk_plus_m2+state.bulk_minus_m2, parameters)
    intrinsic = read_shockley_energy_J_m2(
        boundary_misorientation_rad(state, parameters), parameters)
    extrinsic = parameters.extrinsic_line_energy_J_m*(
        state.extrinsic_plus_m1+state.extrinsic_minus_m1)
    return float(bulk+intrinsic+extrinsic)


def absorb_lattice_content(
        state, *, absorbed_plus_m1, absorbed_minus_m1, compatible,
        parameters, energy_tolerance_J_m2=1.0e-14):
    """Propose a conservative lattice-to-boundary absorption transaction.

    Compatible net content changes the structural boundary.  Equal opposite
    content is extrinsic and cannot rotate it.  If incoming net content opposes
    existing structural content, their exact reduction is reported as
    annihilation.  Incompatible content is entirely extrinsic.
    """
    state.validate()
    plus = float(absorbed_plus_m1)
    minus = float(absorbed_minus_m1)
    if any(not math.isfinite(value) or value < 0.0 for value in (plus, minus)):
        raise ValueError("absorbed line content must be finite and nonnegative")
    width = parameters.reservoir_width_m
    if plus > state.bulk_plus_m2*width+1e-12 or minus > state.bulk_minus_m2*width+1e-12:
        raise ValueError("absorption exceeds the lattice reservoir")

    structural_before = state.structural_signed_m1
    extra_plus = state.extrinsic_plus_m1
    extra_minus = state.extrinsic_minus_m1
    annihilated = 0.0
    if compatible:
        incoming_signed = plus-minus
        structural_after = structural_before+incoming_signed
        neutral = min(plus, minus)
        extra_plus += neutral
        extra_minus += neutral
        annihilated = max(
            abs(structural_before)+abs(incoming_signed)-abs(structural_after),
            0.0)
    else:
        structural_after = structural_before
        extra_plus += plus
        extra_minus += minus

    candidate = StructuralBoundaryState(
        bulk_plus_m2=max(state.bulk_plus_m2-plus/width, 0.0),
        bulk_minus_m2=max(state.bulk_minus_m2-minus/width, 0.0),
        structural_signed_m1=structural_after,
        extrinsic_plus_m1=extra_plus,
        extrinsic_minus_m1=extra_minus,
    )
    candidate.validate()
    structural_line_change = abs(structural_after)-abs(structural_before)
    extrinsic_change = (
        candidate.extrinsic_plus_m1+candidate.extrinsic_minus_m1
        -state.extrinsic_plus_m1-state.extrinsic_minus_m1)
    # An annihilation consumes two line populations per opposing pair.  The
    # expression above is already the total lost line (twice the pair count).
    balance = plus+minus-structural_line_change-extrinsic_change-annihilated
    before_energy = structural_boundary_energy_J_m2(state, parameters)
    candidate_energy = structural_boundary_energy_J_m2(candidate, parameters)
    delta = candidate_energy-before_energy
    accepted = delta <= float(energy_tolerance_J_m2)
    if plus == 0.0 and minus == 0.0:
        classification = "no_incoming_content"
    elif not compatible:
        classification = "incompatible_extrinsic_storage"
    elif plus > 0.0 and minus > 0.0 and abs(plus-minus) <= 1e-12*max(plus, minus):
        classification = "sign_balanced_no_rotation"
    elif annihilated > 0.0:
        classification = "compatible_structural_annihilation"
    else:
        classification = "compatible_absorption_rotation"
    return AbsorptionLedger(
        state, candidate, plus, minus,
        structural_after-structural_before, extrinsic_change, annihilated,
        balance, before_energy, candidate_energy, delta, accepted,
        classification)


def network_structural_content_m1(state, parameters):
    state.validate()
    orientation = np.asarray(state.orientation_rad, dtype=float)
    return np.diff(orientation)/parameters.burgers_m


def network_energy_J_m2(state, parameters, boundary_area_weights=(1.0, 1.0)):
    content = network_structural_content_m1(state, parameters)
    weights = np.asarray(boundary_area_weights, dtype=float)
    if weights.shape != (2,) or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("two positive finite boundary area weights required")
    return float(np.sum(weights*read_shockley_energy_J_m2(
        parameters.burgers_m*content, parameters)))


def transfer_network_structural_content(
        state, transferred_m1, parameters, *, boundary_area_weights=(1.0, 1.0),
        energy_tolerance_J_m2=1.0e-14):
    """Transfer structural line from boundary 2 to boundary 1.

    Outer orientations are held fixed.  The middle orientation is derived from
    the transferred signed line, so the two boundary changes are equal and
    opposite and their signed structural sum is exactly conserved.
    """
    state.validate()
    transfer = float(transferred_m1)
    if not math.isfinite(transfer):
        raise ValueError("transfer must be finite")
    before_content = network_structural_content_m1(state, parameters)
    candidate_content = before_content+np.array((transfer, -transfer))
    left, _, right = map(float, state.orientation_rad)
    middle_left = left+parameters.burgers_m*candidate_content[0]
    middle_right = right-parameters.burgers_m*candidate_content[1]
    compatibility_error = middle_left-middle_right
    if abs(compatibility_error) > 1e-12*max(abs(middle_left), abs(middle_right), 1.0):
        raise ValueError("input network is not orientation-compatible")
    candidate = NetworkState((left, .5*(middle_left+middle_right), right))
    before_energy = network_energy_J_m2(
        state, parameters, boundary_area_weights)
    candidate_energy = network_energy_J_m2(
        candidate, parameters, boundary_area_weights)
    before_sum = float(np.sum(before_content))
    candidate_sum = float(np.sum(candidate_content))
    delta = candidate_energy-before_energy
    return NetworkTransferLedger(
        state, candidate, transfer, before_sum, candidate_sum,
        candidate_sum-before_sum, before_energy, candidate_energy, delta,
        delta <= float(energy_tolerance_J_m2))


def exp_floor_event_rate_Hz(affinity_J_m, temperature_K, parameters):
    """Immutable Arrhenius EXP-floor rate driven by a line affinity.

    ``affinity_J_m * activation_segment_m`` is the event work.  A positive
    affinity lowers the activation barrier but never below the declared floor.
    """
    affinity = np.asarray(affinity_J_m, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    affinity, temperature = np.broadcast_arrays(affinity, temperature)
    if np.any(~np.isfinite(affinity)) or np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("finite affinity and positive temperature required")
    eV_J = 1.602176634e-19
    kB_J_K = 1.380649e-23
    work_eV = affinity*parameters.activation_segment_m/eV_J
    barrier_eV = np.maximum(
        parameters.barrier_floor_eV,
        parameters.activation_enthalpy_eV-work_eV)
    rate = parameters.attempt_frequency_Hz*np.exp(
        -barrier_eV*eV_J/(kB_J_K*temperature))
    return float(rate) if rate.ndim == 0 else rate
