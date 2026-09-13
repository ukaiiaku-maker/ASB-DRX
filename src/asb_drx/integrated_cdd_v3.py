"""Integrated four-family Arrhenius mechanics and staggered signed CDD.

This is a new Mission-v3 production candidate.  It is deliberately a declared
small-strain antiplane reduction: plastic distortion is accumulated additively
and lattice rotation is advanced from total spin minus plastic spin.  It must
not be interpreted as a finite-strain multiplicative constitutive model.  The
historical Bertin model is not called, and no grain/phase allocation or
collective DD closure is present.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .arrhenius_v3 import ArrheniusMechanism
from .cdd_flux_v3 import (
    logarithmic_mean,
    variational_chemical_potentials_J_m,
    variational_correlation_energy_J_m3,
    variational_correlation_fluxes,
    variational_correlation_imex_fluxes,
)
from .staggered_cdd import (
    StaggeredFluxLedger,
    StaggeredSignedState,
    advance_staggered_flux_with_ledger,
    cell_centered_slip,
    face_traction_from_cells,
    work_conjugacy_residual_J_m3,
)


@dataclass(frozen=True)
class IntegratedCDDParameters:
    glide: ArrheniusMechanism
    shear_modulus_Pa: float
    volumetric_heat_capacity_J_m3_K: float
    line_energy_J_m: float
    slip_dyads_crystal: np.ndarray
    correlation_mobility_m_Pa_s: float
    backstress_coefficient: float
    diffusion_coefficient: float
    reference_density_m2: float
    density_floor_m2: float = 1.0e8
    glide_event_length_m: float | None = None
    junction_line_energy_J_m: float | None = None
    wall_line_energy_J_m: float | None = None
    correlation_integration: str = "imex"
    multiplication_per_slip_m2: float = 0.0
    annihilation: ArrheniusMechanism | None = None
    locking: ArrheniusMechanism | None = None
    unlocking: ArrheniusMechanism | None = None
    wall_capture: ArrheniusMechanism | None = None

    def __post_init__(self) -> None:
        for name in (
            "shear_modulus_Pa", "volumetric_heat_capacity_J_m3_K",
            "line_energy_J_m", "correlation_mobility_m_Pa_s",
            "reference_density_m2", "density_floor_m2",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("backstress_coefficient", "diffusion_coefficient"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        dyads = np.asarray(self.slip_dyads_crystal, dtype=float)
        if dyads.shape != (4, 3, 3) or np.any(~np.isfinite(dyads)):
            raise ValueError("slip_dyads_crystal must have shape (4,3,3)")
        if np.max(np.abs(np.trace(dyads, axis1=1, axis2=2))) > 1.0e-12:
            raise ValueError("slip dyads must be isochoric")
        if self.correlation_integration not in ("explicit", "imex"):
            raise ValueError("correlation_integration must be 'explicit' or 'imex'")
        if (
            self.glide_event_length_m is None
            or not math.isfinite(self.glide_event_length_m)
            or self.glide_event_length_m <= 0.0
        ):
            raise ValueError("glide_event_length_m must be explicitly positive")
        for name in ("junction_line_energy_J_m", "wall_line_energy_J_m"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, self.line_energy_J_m)
            elif not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.multiplication_per_slip_m2)
            or self.multiplication_per_slip_m2 < 0.0
        ):
            raise ValueError("multiplication_per_slip_m2 must be nonnegative")
        object.__setattr__(self, "slip_dyads_crystal", dyads.copy())


@dataclass(frozen=True)
class IntegratedCDDState:
    signed: StaggeredSignedState
    plastic_distortion: np.ndarray
    orientation: np.ndarray
    temperature_K: np.ndarray
    applied_shear: float = 0.0
    time_s: float = 0.0
    accepted_steps: int = 0

    def __post_init__(self) -> None:
        families, cells = self.signed.mobile_plus_m2.shape
        if families != 4:
            raise ValueError("the integrated state requires four Burgers families")
        beta_p = np.asarray(self.plastic_distortion, dtype=float)
        rotation = np.asarray(self.orientation, dtype=float)
        temperature = np.asarray(self.temperature_K, dtype=float)
        if beta_p.shape != (cells, 3, 3) or rotation.shape != beta_p.shape:
            raise ValueError("plastic distortion and orientation must have shape (cells,3,3)")
        if temperature.shape != (cells,) or np.any(temperature <= 0.0):
            raise ValueError("temperature_K must be positive and cellwise")
        if np.any(~np.isfinite(beta_p)):
            raise ValueError("plastic distortion must be finite")
        if np.max(np.abs(np.trace(beta_p, axis1=1, axis2=2))) > 2.0e-11:
            raise ValueError("plastic distortion must be isochoric")
        if np.any(~np.isfinite(rotation)):
            raise ValueError("orientation must be finite")
        identity = np.eye(3)
        for index in range(cells):
            if not np.allclose(rotation[index].T @ rotation[index], identity, atol=2.0e-11):
                raise ValueError("orientation must be a proper rotation")
            if np.linalg.det(rotation[index]) <= 0.0:
                raise ValueError("orientation must be a proper rotation")
        if not math.isfinite(self.applied_shear) or not math.isfinite(self.time_s):
            raise ValueError("applied shear and time must be finite")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time and accepted steps must be nonnegative")
        object.__setattr__(self, "plastic_distortion", beta_p.copy())
        object.__setattr__(self, "orientation", rotation.copy())
        object.__setattr__(self, "temperature_K", temperature.copy())

    @property
    def physical_grain_count(self) -> int:
        return 1


@dataclass(frozen=True)
class IntegratedCDDLedger:
    external_work_J_m3: float
    elastic_energy_change_J_m3: float
    plastic_work_J_m3: float
    correlation_energy_change_J_m3: float
    mobile_correlation_energy_change_J_m3: float
    long_range_energy_change_J_m3: float
    stored_line_energy_change_J_m3: float
    mobile_line_energy_change_J_m3: float
    junction_energy_change_J_m3: float
    wall_energy_change_J_m3: float
    transport_free_energy_change_J_m3: float
    reaction_free_energy_change_J_m3: float
    transport_heat_J_m3: float
    reaction_heat_J_m3: float
    heat_J_m3: float
    thermal_energy_change_J_m3: float
    total_energy_residual_J_m3: float
    work_projection_residual_J_m3: float
    flux: StaggeredFluxLedger
    reaction: "IntegratedReactionLedger"


@dataclass(frozen=True)
class IntegratedReactionLedger:
    line_content_before_m_inv: float
    pair_generated_m_inv: float
    pair_annihilated_m_inv: float
    locked_transfer_m_inv: float
    unlocked_transfer_m_inv: float
    wall_capture_m_inv: float
    line_content_after_m_inv: float
    line_balance_residual_m_inv: float
    maximum_signed_burgers_residual_m2: float


@dataclass(frozen=True)
class _StoredEnergy:
    mobile_correlation_J_m3: float
    long_range_J_m3: float
    mobile_line_J_m3: float
    junction_J_m3: float
    wall_J_m3: float

    @property
    def total_J_m3(self) -> float:
        return (
            self.mobile_correlation_J_m3 + self.long_range_J_m3
            + self.mobile_line_J_m3
            + self.junction_J_m3 + self.wall_J_m3
        )

    @property
    def correlation_J_m3(self) -> float:
        return self.mobile_correlation_J_m3 + self.long_range_J_m3


def _stored_energy(
    signed: StaggeredSignedState,
    parameters: IntegratedCDDParameters,
) -> _StoredEnergy:
    """Explicit mobile, junction, wall, and total-signed long-range storage."""

    cells = signed.mobile_plus_m2.shape[1]
    common = dict(
        reference_density_m2=parameters.reference_density_m2,
        density_floor_m2=parameters.density_floor_m2,
        stationary_plus_m2=signed.locked_plus_m2 + signed.wall_plus_m2,
        stationary_minus_m2=signed.locked_minus_m2 + signed.wall_minus_m2,
    )
    mobile_correlation = float(np.mean(variational_correlation_energy_J_m3(
        signed.mobile_plus_m2, signed.mobile_minus_m2,
        np.full(cells, parameters.shear_modulus_Pa), signed.burgers_m,
        backstress_coefficient=0.0,
        diffusion_coefficient=parameters.diffusion_coefficient,
        **common,
    )))
    long_range = float(np.mean(variational_correlation_energy_J_m3(
        signed.mobile_plus_m2, signed.mobile_minus_m2,
        np.full(cells, parameters.shear_modulus_Pa), signed.burgers_m,
        backstress_coefficient=parameters.backstress_coefficient,
        diffusion_coefficient=0.0,
        **common,
    )))
    return _StoredEnergy(
        mobile_correlation,
        long_range,
        parameters.line_energy_J_m * float(np.mean(
            signed.mobile_plus_m2 + signed.mobile_minus_m2
        )),
        parameters.junction_line_energy_J_m * float(np.mean(
            signed.locked_plus_m2 + signed.locked_minus_m2
        )),
        parameters.wall_line_energy_J_m * float(np.mean(
            signed.wall_plus_m2 + signed.wall_minus_m2
        )),
    )


def apply_integrated_reactions(
    state: StaggeredSignedState,
    slip_increment_cell: np.ndarray,
    temperature_K: np.ndarray,
    dt_s: float,
    parameters: IntegratedCDDParameters,
) -> tuple[StaggeredSignedState, IntegratedReactionLedger]:
    """Bounded pair/source and stationary-reservoir reaction substep."""

    slip = np.asarray(slip_increment_cell, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    if slip.shape != state.mobile_plus_m2.shape:
        raise ValueError("slip increment must match signed populations")
    if temperature.shape != (slip.shape[1],):
        raise ValueError("reaction temperature must be cellwise")
    before_arrays = (
        state.mobile_plus_m2, state.mobile_minus_m2,
        state.locked_plus_m2, state.locked_minus_m2,
        state.wall_plus_m2, state.wall_minus_m2,
    )
    before = state.dx_m * float(sum(np.sum(x) for x in before_arrays))
    signed_before = state.total_signed_density_m2
    pair_each = 0.5 * parameters.multiplication_per_slip_m2 * np.abs(slip)
    mobile_plus = state.mobile_plus_m2 + pair_each
    mobile_minus = state.mobile_minus_m2 + pair_each
    locked_plus = state.locked_plus_m2.copy()
    locked_minus = state.locked_minus_m2.copy()
    wall_plus = state.wall_plus_m2.copy()
    wall_minus = state.wall_minus_m2.copy()
    generated = state.dx_m * float(2.0 * np.sum(pair_each))
    annihilated = 0.0
    locked_amount = 0.0
    unlocked_amount = 0.0
    wall_amount = 0.0

    def rate(mechanism: ArrheniusMechanism | None, family: int, cell: int) -> float:
        if mechanism is None:
            return 0.0
        density = max(
            float(mobile_plus[family, cell] + mobile_minus[family, cell]),
            parameters.density_floor_m2,
        )
        return mechanism.one_way_rate_s_inv(
            0.0, float(temperature[cell]), density
        )

    for family, cell in np.ndindex(slip.shape):
        annihilation_fraction = -math.expm1(
            -rate(parameters.annihilation, family, cell) * dt_s
        )
        removed_each = annihilation_fraction * min(
            mobile_plus[family, cell], mobile_minus[family, cell]
        )
        mobile_plus[family, cell] -= removed_each
        mobile_minus[family, cell] -= removed_each
        annihilated += 2.0 * removed_each

        unlock_fraction = -math.expm1(
            -rate(parameters.unlocking, family, cell) * dt_s
        )
        unlock_plus = unlock_fraction * locked_plus[family, cell]
        unlock_minus = unlock_fraction * locked_minus[family, cell]
        locked_plus[family, cell] -= unlock_plus
        locked_minus[family, cell] -= unlock_minus
        mobile_plus[family, cell] += unlock_plus
        mobile_minus[family, cell] += unlock_minus
        unlocked_amount += unlock_plus + unlock_minus

        lock_rate = rate(parameters.locking, family, cell)
        wall_rate = rate(parameters.wall_capture, family, cell)
        total_rate = lock_rate + wall_rate
        if total_rate > 0.0:
            transfer_fraction = -math.expm1(-total_rate * dt_s)
            lock_fraction = transfer_fraction * lock_rate / total_rate
            wall_fraction = transfer_fraction * wall_rate / total_rate
            lock_plus = lock_fraction * mobile_plus[family, cell]
            lock_minus = lock_fraction * mobile_minus[family, cell]
            capture_plus = wall_fraction * mobile_plus[family, cell]
            capture_minus = wall_fraction * mobile_minus[family, cell]
            mobile_plus[family, cell] -= lock_plus + capture_plus
            mobile_minus[family, cell] -= lock_minus + capture_minus
            locked_plus[family, cell] += lock_plus
            locked_minus[family, cell] += lock_minus
            wall_plus[family, cell] += capture_plus
            wall_minus[family, cell] += capture_minus
            locked_amount += lock_plus + lock_minus
            wall_amount += capture_plus + capture_minus

    advanced = StaggeredSignedState(
        mobile_plus, mobile_minus, state.face_slip, state.burgers_m, state.dx_m,
        locked_plus, locked_minus, wall_plus, wall_minus,
    )
    after = float(sum(np.sum(x) for x in (
        mobile_plus, mobile_minus, locked_plus, locked_minus,
        wall_plus, wall_minus,
    )))
    return advanced, IntegratedReactionLedger(
        before, generated, state.dx_m * annihilated,
        state.dx_m * locked_amount, state.dx_m * unlocked_amount,
        state.dx_m * wall_amount, state.dx_m * after,
        state.dx_m * after - (before + generated - state.dx_m * annihilated),
        float(np.max(np.abs(advanced.total_signed_density_m2 - signed_before))),
    )


@dataclass(frozen=True)
class IntegratedCDDStep:
    state: IntegratedCDDState
    ledger: IntegratedCDDLedger
    resolved_shear_Pa: np.ndarray
    plus_face_flux_m_inv_s: np.ndarray
    minus_face_flux_m_inv_s: np.ndarray
    accepted_dt_s: float
    halvings: int


def _macro_plastic_shear(state: IntegratedCDDState, parameters: IntegratedCDDParameters) -> np.ndarray:
    """Accumulated laboratory xz plastic distortion."""

    return state.plastic_distortion[:, 0, 2]


def _rotated_slip_dyads_and_weights(
    state: IntegratedCDDState,
    parameters: IntegratedCDDParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Cellwise lab-frame dyads and their antiplane work projections."""

    cells = state.signed.mobile_plus_m2.shape[1]
    dyads = np.empty((4, cells, 3, 3))
    for family, cell in np.ndindex((4, cells)):
        rotation = state.orientation[cell]
        dyads[family, cell] = (
            rotation @ parameters.slip_dyads_crystal[family] @ rotation.T
        )
    return dyads, dyads[:, :, 0, 2]


def _common_stress_Pa(state: IntegratedCDDState, parameters: IntegratedCDDParameters) -> float:
    return parameters.shear_modulus_Pa * (
        state.applied_shear - float(np.mean(_macro_plastic_shear(state, parameters)))
    )


def frozen_mobile_rhs_m2_s(
    mobile_plus_m2: np.ndarray,
    mobile_minus_m2: np.ndarray,
    resolved_shear_Pa: np.ndarray,
    temperature_K: np.ndarray,
    burgers_m: float,
    dx_m: float,
    parameters: IntegratedCDDParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Semidiscrete explicit flux RHS used for the frozen-state spectrum."""

    plus = np.asarray(mobile_plus_m2, dtype=float)
    minus = np.asarray(mobile_minus_m2, dtype=float)
    stress = np.asarray(resolved_shear_Pa, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    if plus.shape != minus.shape or plus.shape != stress.shape or plus.shape[0] != 4:
        raise ValueError("frozen fields must share a four-family grid")
    if temperature.shape != (plus.shape[1],):
        raise ValueError("temperature must be cellwise")
    stress_face = 0.5 * (stress + np.roll(stress, -1, axis=1))
    temperature_face = 0.5 * (temperature + np.roll(temperature, -1))

    def physical_flux(population: np.ndarray, stress_sign: float) -> np.ndarray:
        face_population = np.maximum(
            logarithmic_mean(
                population + parameters.density_floor_m2,
                np.roll(population, -1, axis=1) + parameters.density_floor_m2,
            ) - parameters.density_floor_m2,
            0.0,
        )
        velocity = np.empty_like(population)
        for family, cell in np.ndindex(population.shape):
            velocity[family, cell] = parameters.glide.glide_velocity_m_s(
                stress_sign * float(stress_face[family, cell]),
                float(temperature_face[cell]), parameters.glide_event_length_m,
            )
        return face_population * velocity

    flux_plus = physical_flux(plus, 1.0)
    flux_minus = physical_flux(minus, -1.0)
    corr_plus, corr_minus = variational_correlation_fluxes(
        plus, minus,
        np.full_like(plus, parameters.correlation_mobility_m_Pa_s),
        np.full(plus.shape[1], parameters.shear_modulus_Pa),
        burgers_m, dx_m,
        backstress_coefficient=parameters.backstress_coefficient,
        diffusion_coefficient=parameters.diffusion_coefficient,
        reference_density_m2=parameters.reference_density_m2,
        density_floor_m2=parameters.density_floor_m2,
        stationary_plus_m2=np.zeros_like(plus),
        stationary_minus_m2=np.zeros_like(minus),
    )
    return (
        -(flux_plus + corr_plus - np.roll(flux_plus + corr_plus, 1, axis=1)) / dx_m,
        -(flux_minus + corr_minus - np.roll(flux_minus + corr_minus, 1, axis=1)) / dx_m,
    )


def frozen_mode_eigenvalues_s_inv(
    grid_points: int,
    domain_m: float,
    density_per_population_m2: float,
    resolved_shear_by_family_Pa: np.ndarray,
    temperature_K: float,
    burgers_m: float,
    parameters: IntegratedCDDParameters,
    *,
    relative_increment: float = 1.0e-6,
) -> dict[int, np.ndarray]:
    """Numerical 8-population Fourier symbol through the Nyquist mode."""

    if grid_points < 4 or grid_points % 2:
        raise ValueError("an even grid with at least four points is required")
    if domain_m <= 0.0 or density_per_population_m2 <= 0.0:
        raise ValueError("domain and density must be positive")
    stresses = np.asarray(resolved_shear_by_family_Pa, dtype=float)
    if stresses.shape != (4,):
        raise ValueError("exactly four resolved family stresses are required")
    dx_m = domain_m / grid_points
    base_plus = np.full((4, 1), density_per_population_m2)
    base_minus = base_plus.copy()
    amplitude = relative_increment * density_per_population_m2
    base_chemical = np.concatenate(variational_chemical_potentials_J_m(
        base_plus, base_minus, np.asarray([parameters.shear_modulus_Pa]),
        burgers_m,
        backstress_coefficient=parameters.backstress_coefficient,
        diffusion_coefficient=parameters.diffusion_coefficient,
        reference_density_m2=parameters.reference_density_m2,
        density_floor_m2=parameters.density_floor_m2,
    ), axis=0)[:, 0]
    hessian = np.zeros((8, 8))
    for column in range(8):
        plus = base_plus.copy()
        minus = base_minus.copy()
        (plus if column < 4 else minus)[column % 4, 0] += amplitude
        shifted = np.concatenate(variational_chemical_potentials_J_m(
            plus, minus, np.asarray([parameters.shear_modulus_Pa]), burgers_m,
            backstress_coefficient=parameters.backstress_coefficient,
            diffusion_coefficient=parameters.diffusion_coefficient,
            reference_density_m2=parameters.reference_density_m2,
            density_floor_m2=parameters.density_floor_m2,
        ), axis=0)[:, 0]
        hessian[:, column] = (shifted - base_chemical) / amplitude
    velocities = np.asarray([
        parameters.glide.glide_velocity_m_s(
            float(stress), temperature_K, parameters.glide_event_length_m
        )
        for stress in stresses
    ])
    advection_velocity = np.concatenate((velocities, -velocities))
    population_mobility = (
        parameters.correlation_mobility_m_Pa_s
        * density_per_population_m2 / burgers_m
    )
    spectra: dict[int, np.ndarray] = {}
    for mode in range(1, grid_points // 2 + 1):
        angle = 2.0 * np.pi * mode / grid_points
        laplacian_symbol = -4.0 * math.sin(0.5 * angle) ** 2 / dx_m**2
        symbol = population_mobility * laplacian_symbol * hessian
        symbol = symbol.astype(complex) - 1j * np.diag(
            advection_velocity * math.sin(angle) / dx_m
        )
        spectra[mode] = np.linalg.eigvals(symbol)
    return spectra


def integrated_cdd_step(
    state: IntegratedCDDState,
    applied_shear_rate_s_inv: float,
    proposed_dt_s: float,
    parameters: IntegratedCDDParameters,
    *,
    maximum_halvings: int = 40,
) -> IntegratedCDDStep:
    """Advance one energy-checked coupled explicit/IMEX interval.

    The stiff logarithmic diffusion is implicit by default; the nonlinear
    polarization and physical Arrhenius fluxes remain explicit. Step rejection,
    rather than clipping or filtering, controls the remaining explicit terms.
    """

    if not math.isfinite(applied_shear_rate_s_inv):
        raise ValueError("applied_shear_rate_s_inv must be finite")
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    signed = state.signed
    old_stress = _common_stress_Pa(state, parameters)
    cells = signed.mobile_plus_m2.shape[1]
    rotated_dyads, weights = _rotated_slip_dyads_and_weights(state, parameters)
    resolved = old_stress * weights
    resolved_face = 0.5 * (resolved + np.roll(resolved, -1, axis=1))
    temperature_face = 0.5 * (
        state.temperature_K + np.roll(state.temperature_K, -1)
    )
    plus_face = logarithmic_mean(
        signed.mobile_plus_m2 + parameters.density_floor_m2,
        np.roll(signed.mobile_plus_m2, -1, axis=1) + parameters.density_floor_m2,
    ) - parameters.density_floor_m2
    minus_face = logarithmic_mean(
        signed.mobile_minus_m2 + parameters.density_floor_m2,
        np.roll(signed.mobile_minus_m2, -1, axis=1) + parameters.density_floor_m2,
    ) - parameters.density_floor_m2
    plus_face = np.maximum(plus_face, 0.0)
    minus_face = np.maximum(minus_face, 0.0)
    physical_plus = np.zeros_like(resolved)
    physical_minus = np.zeros_like(resolved)
    for family, cell in np.ndindex(resolved.shape):
        physical_plus[family, cell] = plus_face[family, cell] * parameters.glide.glide_velocity_m_s(
            float(resolved_face[family, cell]), float(temperature_face[cell]),
            parameters.glide_event_length_m,
        )
        physical_minus[family, cell] = minus_face[family, cell] * parameters.glide.glide_velocity_m_s(
            float(-resolved_face[family, cell]), float(temperature_face[cell]),
            parameters.glide_event_length_m,
        )
    old_energy = _stored_energy(signed, parameters)
    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        correlation_kwargs = dict(
            backstress_coefficient=parameters.backstress_coefficient,
            diffusion_coefficient=parameters.diffusion_coefficient,
            reference_density_m2=parameters.reference_density_m2,
            density_floor_m2=parameters.density_floor_m2,
            stationary_plus_m2=signed.locked_plus_m2 + signed.wall_plus_m2,
            stationary_minus_m2=signed.locked_minus_m2 + signed.wall_minus_m2,
        )
        if parameters.correlation_integration == "imex":
            try:
                correlation_plus, correlation_minus = variational_correlation_imex_fluxes(
                    signed.mobile_plus_m2, signed.mobile_minus_m2,
                    parameters.correlation_mobility_m_Pa_s,
                    parameters.shear_modulus_Pa, signed.burgers_m,
                    signed.dx_m, dt_s, **correlation_kwargs,
                )
            except RuntimeError:
                dt_s *= 0.5
                continue
        else:
            correlation_plus, correlation_minus = variational_correlation_fluxes(
                signed.mobile_plus_m2, signed.mobile_minus_m2,
                np.full_like(
                    signed.mobile_plus_m2,
                    parameters.correlation_mobility_m_Pa_s,
                ),
                np.full(cells, parameters.shear_modulus_Pa),
                signed.burgers_m, signed.dx_m, **correlation_kwargs,
            )
        flux_plus = physical_plus + correlation_plus
        flux_minus = physical_minus + correlation_minus
        try:
            advanced_signed, flux_ledger = advance_staggered_flux_with_ledger(
                signed, flux_plus, flux_minus, dt_s
            )
        except RuntimeError:
            dt_s *= 0.5
            continue
        if flux_ledger.clipping_added_m_inv != 0.0:
            dt_s *= 0.5
            continue
        slip_increment_face = advanced_signed.face_slip - signed.face_slip
        slip_increment_cell = cell_centered_slip(slip_increment_face)
        transported_energy = _stored_energy(advanced_signed, parameters)
        advanced_signed, reaction_ledger = apply_integrated_reactions(
            advanced_signed, slip_increment_cell, state.temperature_K,
            dt_s, parameters,
        )
        macro_increment = np.sum(weights * slip_increment_cell, axis=0)
        applied_increment = applied_shear_rate_s_inv * dt_s
        new_applied = state.applied_shear + applied_increment
        new_mean_plastic = float(np.mean(
            _macro_plastic_shear(state, parameters) + macro_increment
        ))
        new_stress = parameters.shear_modulus_Pa * (new_applied - new_mean_plastic)
        mean_stress = 0.5 * (old_stress + new_stress)
        external = mean_stress * applied_increment
        elastic_change = (new_stress**2 - old_stress**2) / (2.0 * parameters.shear_modulus_Pa)
        plastic_work = mean_stress * float(np.mean(macro_increment))
        new_energy = _stored_energy(advanced_signed, parameters)
        mobile_correlation_change = (
            new_energy.mobile_correlation_J_m3 - old_energy.mobile_correlation_J_m3
        )
        long_range_change = new_energy.long_range_J_m3 - old_energy.long_range_J_m3
        correlation_change = mobile_correlation_change + long_range_change
        mobile_line_change = new_energy.mobile_line_J_m3 - old_energy.mobile_line_J_m3
        junction_change = new_energy.junction_J_m3 - old_energy.junction_J_m3
        wall_change = new_energy.wall_J_m3 - old_energy.wall_J_m3
        line_change = mobile_line_change + junction_change + wall_change
        transport_free_change = transported_energy.total_J_m3 - old_energy.total_J_m3
        reaction_free_change = new_energy.total_J_m3 - transported_energy.total_J_m3
        transport_heat = plastic_work - transport_free_change
        reaction_heat = -reaction_free_change
        heat = transport_heat + reaction_heat
        scale = max(abs(external), abs(elastic_change), abs(plastic_work), abs(correlation_change), 1.0)
        if transport_heat < -1.0e-12 * scale or reaction_heat < -1.0e-12 * scale:
            dt_s *= 0.5
            continue
        transport_heat = max(transport_heat, 0.0)
        reaction_heat = max(reaction_heat, 0.0)
        heat = transport_heat + reaction_heat
        temperature = state.temperature_K + heat / parameters.volumetric_heat_capacity_J_m3_K
        plastic_distortion = state.plastic_distortion.copy()
        orientation = state.orientation.copy()
        for cell in range(cells):
            increment_lp = np.zeros((3, 3))
            for family in range(4):
                increment_lp += (
                    slip_increment_cell[family, cell]
                    * rotated_dyads[family, cell]
                )
            plastic_distortion[cell] += increment_lp
            total_spin_increment = np.zeros((3, 3))
            total_spin_increment[0, 2] = 0.5 * applied_increment
            total_spin_increment[2, 0] = -0.5 * applied_increment
            plastic_spin_increment = 0.5 * (increment_lp - increment_lp.T)
            orientation[cell] = expm(
                total_spin_increment - plastic_spin_increment
            ) @ orientation[cell]
        projection_residual = work_conjugacy_residual_J_m3(
            mean_stress * weights,
            slip_increment_face,
        ) / cells
        thermal_change = parameters.volumetric_heat_capacity_J_m3_K * float(
            np.mean(temperature - state.temperature_K)
        )
        residual = external - elastic_change - correlation_change - line_change - thermal_change
        thermal_roundoff = (
            16.0 * np.finfo(float).eps
            * parameters.volumetric_heat_capacity_J_m3_K
            * float(np.max(temperature))
        )
        if abs(residual) > max(2.0e-11 * scale, thermal_roundoff):
            dt_s *= 0.5
            continue
        advanced = IntegratedCDDState(
            advanced_signed, plastic_distortion, orientation, temperature, new_applied,
            state.time_s + dt_s, state.accepted_steps + 1,
        )
        return IntegratedCDDStep(
            advanced,
            IntegratedCDDLedger(
                external, elastic_change, plastic_work, correlation_change,
                mobile_correlation_change, long_range_change,
                line_change, mobile_line_change, junction_change, wall_change,
                transport_free_change, reaction_free_change,
                transport_heat, reaction_heat, heat, thermal_change, residual,
                projection_residual, flux_ledger, reaction_ledger,
            ),
            resolved, flux_plus, flux_minus, dt_s, halvings,
        )
    raise RuntimeError("no invariant-admissible integrated CDD step found")


INTEGRATED_PARAMETER_CLASSIFICATION = {
    "Arrhenius_form": "immutable_framework",
    "glide_event_length_m": "physical_literature_bound",
    "slip_dyads_crystal": "immutable_framework",
    "correlation_mobility_m_Pa_s": "generic_development_parameter",
    "backstress_coefficient": "generic_development_parameter",
    "diffusion_coefficient": "generic_development_parameter",
    "density_floor_m2": "numerical_regularization",
    "correlation_integration": "numerical_regularization",
    "multiplication_per_slip_m2": "generic_development_parameter",
    "line_energy_J_m": "physical_literature_bound",
    "junction_line_energy_J_m": "physical_literature_bound",
    "wall_line_energy_J_m": "physical_literature_bound",
    "reaction_Arrhenius_mechanisms": "generic_development_parameter",
    "collective_DD_closure": "disabled_ablation_only",
}


CHECKPOINT_SCHEMA = "asb-drx-integrated-cdd-v3/v2-small-strain"


def save_integrated_cdd_checkpoint(path: str | Path, state: IntegratedCDDState) -> None:
    """Persist every authoritative field required for an exact restart."""

    np.savez(
        Path(path), schema=np.array(CHECKPOINT_SCHEMA),
        mobile_plus_m2=state.signed.mobile_plus_m2,
        mobile_minus_m2=state.signed.mobile_minus_m2,
        locked_plus_m2=state.signed.locked_plus_m2,
        locked_minus_m2=state.signed.locked_minus_m2,
        wall_plus_m2=state.signed.wall_plus_m2,
        wall_minus_m2=state.signed.wall_minus_m2,
        face_slip=state.signed.face_slip,
        burgers_m=np.array(state.signed.burgers_m),
        dx_m=np.array(state.signed.dx_m),
        plastic_distortion=state.plastic_distortion,
        orientation=state.orientation,
        temperature_K=state.temperature_K,
        applied_shear=np.array(state.applied_shear),
        time_s=np.array(state.time_s),
        accepted_steps=np.array(state.accepted_steps, dtype=np.int64),
    )


def load_integrated_cdd_checkpoint(path: str | Path) -> IntegratedCDDState:
    with np.load(Path(path), allow_pickle=False) as archive:
        if str(archive["schema"]) != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported integrated CDD checkpoint schema")
        signed = StaggeredSignedState(
            archive["mobile_plus_m2"], archive["mobile_minus_m2"],
            archive["face_slip"], float(archive["burgers_m"]),
            float(archive["dx_m"]), archive["locked_plus_m2"],
            archive["locked_minus_m2"], archive["wall_plus_m2"],
            archive["wall_minus_m2"],
        )
        return IntegratedCDDState(
            signed, archive["plastic_distortion"],
            archive["orientation"], archive["temperature_K"],
            float(archive["applied_shear"]), float(archive["time_s"]),
            int(archive["accepted_steps"]),
        )
