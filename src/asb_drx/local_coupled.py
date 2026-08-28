"""Locally equilibrated antiplane EXP-floor/thermal/phase-field kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .analytical import ExpFloorLaw
from .antiplane import AntiplaneEquilibrium, solve_periodic_antiplane
from .multi_order import interpolation_h
from .spatial_coupled import (
    SpatialCoupledParameters,
    SpatialMechanismControls,
    _chemical_potential_J_m3,
    _energies_J_m,
)


CHECKPOINT_SCHEMA = "asb-drx-local-antiplane-coupled/v1"


@dataclass(frozen=True)
class LocalCoupledState:
    applied_shear: float
    plastic_shear: np.ndarray
    temperature_K: np.ndarray
    forest_density_m2: np.ndarray
    eta_fields: np.ndarray
    time_s: float = 0.0
    accepted_steps: int = 0

    def __post_init__(self) -> None:
        plastic = np.asarray(self.plastic_shear, dtype=float)
        temperature = np.asarray(self.temperature_K, dtype=float)
        density = np.asarray(self.forest_density_m2, dtype=float)
        fields = np.asarray(self.eta_fields, dtype=float)
        if plastic.ndim != 2 or min(plastic.shape) < 8:
            raise ValueError("plastic_shear must be at least an 8x8 field")
        if temperature.shape != plastic.shape or density.shape != (2, *plastic.shape):
            raise ValueError("temperature/density shapes must match the spatial grid")
        if fields.shape != density.shape:
            raise ValueError("eta_fields must match the two density fields")
        if not all(np.all(np.isfinite(array)) for array in (plastic, temperature, density, fields)):
            raise ValueError("all local coupled fields must be finite")
        if np.any(temperature <= 0.0) or np.any(density <= 0.0):
            raise ValueError("temperature and densities must be positive")
        if np.any(fields < 0.0) or np.any(fields > 1.0):
            raise ValueError("eta_fields must be in [0,1]")
        if not np.allclose(np.sum(fields, axis=0), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("eta_fields must sum pointwise to one")
        if not math.isfinite(self.applied_shear) or not math.isfinite(self.time_s):
            raise ValueError("applied_shear and time_s must be finite")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time_s and accepted_steps must be nonnegative")


@dataclass(frozen=True)
class LocalCoupledLedger:
    external_work_J_m3: float
    elastic_change_J_m3: float
    stored_change_J_m3: float
    interface_order_change_J_m3: float
    mechanical_heat_J_m3: float
    phase_heat_J_m3: float
    thermal_change_J_m3: float
    global_closure_error_J_m3: float
    thermal_closure_error_J_m3: float
    bath_heat_J_m3: float
    plastic_work_J_m3: float


@dataclass(frozen=True)
class LocalCoupledStep:
    state: LocalCoupledState
    equilibrium: AntiplaneEquilibrium
    ledger: LocalCoupledLedger
    accepted_dt_s: float
    halvings: int
    storage_limited_fraction: float


def diffuse_temperature_periodic_exact(
    temperature_K: np.ndarray,
    diffusivity_m2_s: float,
    dt_s: float,
    dx_m: float,
) -> np.ndarray:
    """Exact Fourier propagation of periodic constant-property heat diffusion."""
    temperature = np.asarray(temperature_K, dtype=float)
    if temperature.ndim != 2 or not np.all(np.isfinite(temperature)):
        raise ValueError("temperature_K must be a finite two-dimensional field")
    if not math.isfinite(diffusivity_m2_s) or diffusivity_m2_s < 0.0:
        raise ValueError("diffusivity_m2_s must be finite and nonnegative")
    if not math.isfinite(dt_s) or dt_s < 0.0:
        raise ValueError("dt_s must be finite and nonnegative")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
    if diffusivity_m2_s == 0.0 or dt_s == 0.0:
        return np.array(temperature, copy=True)
    ny, nx = temperature.shape
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx_m)[None, :]
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=dx_m)[:, None]
    decay = np.exp(-diffusivity_m2_s * (kx**2 + ky**2) * dt_s)
    return np.fft.ifft2(np.fft.fft2(temperature) * decay).real


def local_coupled_step(
    state: LocalCoupledState,
    applied_shear_rate_s_inv: float,
    dx_m: float,
    proposed_dt_s: float,
    law: ExpFloorLaw,
    parameters: SpatialCoupledParameters,
    *,
    controls: SpatialMechanismControls = SpatialMechanismControls(),
    maximum_halvings: int = 40,
    tolerance_J_m3: float = 1.0e-8,
    relative_ledger_tolerance: float = 1.0e-9,
) -> LocalCoupledStep:
    if not math.isfinite(applied_shear_rate_s_inv):
        raise ValueError("applied_shear_rate_s_inv must be finite")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    if (
        not math.isfinite(relative_ledger_tolerance)
        or relative_ledger_tolerance <= 0.0
        or relative_ledger_tolerance >= 1.0
    ):
        raise ValueError("relative_ledger_tolerance must be finite and in (0,1)")
    fields = np.asarray(state.eta_fields, dtype=float)
    density = np.asarray(state.forest_density_m2, dtype=float)
    temperature = np.asarray(state.temperature_K, dtype=float)
    plastic = np.asarray(state.plastic_shear, dtype=float)
    weights = interpolation_h(fields)
    if not np.allclose(np.sum(weights, axis=0), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("binary interpolated weights must sum to one")
    old_equilibrium = solve_periodic_antiplane(
        state.applied_shear, plastic, parameters.shear_modulus_Pa, dx_m
    )
    rates = np.zeros_like(density)
    for label in range(2):
        for index in np.ndindex(temperature.shape):
            local_stress = float(old_equilibrium.stress_x_Pa[index])
            if local_stress != 0.0:
                obstacle_stress = abs(local_stress) / law.taylor_ratio(
                    float(density[(label, *index)])
                )
                rates[(label, *index)] = math.copysign(
                    law.shear_rate_s_inv(
                        obstacle_stress,
                        float(density[(label, *index)]),
                        float(temperature[index]),
                    ),
                    local_stress,
                )
    _, old_stored, old_interface = _energies_J_m(fields, density, dx_m, parameters)
    area_m2 = fields.shape[1] * fields.shape[2] * dx_m**2
    diffusivity = parameters.thermal_conductivity_W_m_K / parameters.volumetric_heat_capacity_J_m3_K
    dt_s = proposed_dt_s
    last_rejection = "none"
    for halvings in range(maximum_halvings + 1):
        grain_increment = rates * dt_s
        plastic_increment = np.sum(weights * grain_increment, axis=0)
        applied_increment = applied_shear_rate_s_inv * dt_s
        new_plastic = plastic + plastic_increment
        new_applied = state.applied_shear + applied_increment
        new_equilibrium = solve_periodic_antiplane(
            new_applied, new_plastic, parameters.shear_modulus_Pa, dx_m
        )
        midpoint_stress = 0.5 * (
            old_equilibrium.stress_x_Pa + new_equilibrium.stress_x_Pa
        )
        plastic_work_local = midpoint_stress * plastic_increment
        requested_density_increment = parameters.forest_storage_per_plastic_strain_m2 * np.abs(
            grain_increment
        )
        requested_stored_local = parameters.stored_line_energy_J_m * np.sum(
            weights * requested_density_increment, axis=0
        )
        if np.any(plastic_work_local < -tolerance_J_m3):
            last_rejection = "negative_mechanical_heat"
            dt_s *= 0.5
            continue
        # The nominal Kocks--Mecking storage rate is a requested rate, not an
        # independent energy source.  At low stress it can ask for more line
        # energy than the local plastic work supplies.  Limit all grain-wise
        # storage increments at that point by the same factor, preserving their
        # partition while enforcing the local dissipation inequality.
        storage_scale = np.ones_like(plastic_work_local)
        requesting = requested_stored_local > 0.0
        storage_scale[requesting] = np.minimum(
            1.0,
            np.maximum(plastic_work_local[requesting], 0.0)
            / requested_stored_local[requesting],
        )
        density_increment = requested_density_increment * storage_scale[None, :, :]
        new_density = density + density_increment
        phase_old_total, phase_old_stored, _ = _energies_J_m(
            fields, new_density, dx_m, parameters
        )
        mechanical_stored_local = parameters.stored_line_energy_J_m * np.sum(
            weights * density_increment, axis=0
        )
        mechanical_heat_local = plastic_work_local - mechanical_stored_local
        mechanical_heat_local = np.maximum(mechanical_heat_local, 0.0)
        if controls.evolve_temperature:
            source_temperature = temperature + mechanical_heat_local / parameters.volumetric_heat_capacity_J_m3_K
            conducted_temperature = diffuse_temperature_periodic_exact(
                source_temperature, diffusivity, dt_s, dx_m
            )
        else:
            conducted_temperature = temperature
        if np.any(conducted_temperature <= 0.0) or not np.all(np.isfinite(conducted_temperature)):
            last_rejection = "invalid_conducted_temperature"
            dt_s *= 0.5
            continue
        if controls.evolve_phase:
            chemical = _chemical_potential_J_m3(fields, new_density, dx_m, parameters)
            projected = chemical - np.mean(chemical, axis=0, keepdims=True)
            candidate = fields - dt_s * parameters.phase_mobility_m3_J_s * projected
            if (
                np.any(candidate < -1.0e-14)
                or np.any(candidate > 1.0 + 1.0e-14)
                or not np.all(np.isfinite(candidate))
            ):
                last_rejection = "phase_bounds"
                dt_s *= 0.5
                continue
            candidate = np.clip(candidate, 0.0, 1.0)
            candidate /= np.sum(candidate, axis=0, keepdims=True)
        else:
            projected = np.zeros_like(fields)
            candidate = fields
        new_total, new_stored, new_interface = _energies_J_m(
            candidate, new_density, dx_m, parameters
        )
        if new_total > phase_old_total:
            last_rejection = "phase_energy_increase"
            dt_s *= 0.5
            continue
        phase_heat_mean = (phase_old_total - new_total) / area_m2
        dissipation_weight = np.sum(projected**2, axis=0)
        if float(np.mean(dissipation_weight)) > 0.0:
            phase_heat_local = phase_heat_mean * dissipation_weight / float(
                np.mean(dissipation_weight)
            )
        else:
            phase_heat_local = np.full_like(temperature, phase_heat_mean)
        if controls.evolve_temperature:
            final_temperature = conducted_temperature + phase_heat_local / parameters.volumetric_heat_capacity_J_m3_K
            bath_heat = 0.0
        else:
            final_temperature = temperature
            bath_heat = float(np.mean(mechanical_heat_local)) + phase_heat_mean
        external = 0.5 * (
            old_equilibrium.mean_stress_Pa + new_equilibrium.mean_stress_Pa
        ) * applied_increment
        elastic = new_equilibrium.elastic_energy_J_m3 - old_equilibrium.elastic_energy_J_m3
        plastic_work = float(np.mean(plastic_work_local))
        stored_change = (new_stored - old_stored) / area_m2
        interface_change = (new_interface - old_interface) / area_m2
        mechanical_heat = float(np.mean(mechanical_heat_local))
        thermal_change = parameters.volumetric_heat_capacity_J_m3_K * float(
            np.mean(final_temperature - temperature)
        )
        global_closure = (
            external
            - elastic
            - stored_change
            - interface_change
            - mechanical_heat
            - phase_heat_mean
        )
        thermal_closure = thermal_change + bath_heat - mechanical_heat - phase_heat_mean
        plastic_closure = plastic_work - (phase_old_stored - old_stored) / area_m2 - mechanical_heat
        scale = max(
            abs(external), abs(elastic), abs(stored_change), abs(interface_change),
            abs(plastic_work), mechanical_heat, phase_heat_mean, 1.0,
        )
        thermal_floor = (
            16.0
            * np.finfo(float).eps
            * parameters.volumetric_heat_capacity_J_m3_K
            * float(np.max(final_temperature))
        )
        mechanical_floor = 64.0 * np.finfo(float).eps * max(
            abs(old_equilibrium.elastic_energy_J_m3),
            abs(new_equilibrium.elastic_energy_J_m3),
            abs(old_stored) / area_m2,
            abs(old_interface) / area_m2,
            abs(phase_old_total) / area_m2,
            abs(new_total) / area_m2,
            1.0,
        )
        ledger_limit = max(relative_ledger_tolerance * scale, mechanical_floor)
        if (
            abs(global_closure) > ledger_limit
            or abs(plastic_closure) > ledger_limit
            or abs(thermal_closure) > max(1.0e-8 * scale, thermal_floor)
        ):
            last_rejection = (
                "ledger_closure:"
                f"global={global_closure:.16g},plastic={plastic_closure:.16g},"
                f"thermal={thermal_closure:.16g},scale={scale:.16g},"
                f"ledger_limit={ledger_limit:.16g},thermal_floor={thermal_floor:.16g}"
            )
            dt_s *= 0.5
            continue
        return LocalCoupledStep(
            LocalCoupledState(
                new_applied,
                new_plastic,
                final_temperature,
                new_density,
                candidate,
                state.time_s + dt_s,
                state.accepted_steps + 1,
            ),
            new_equilibrium,
            LocalCoupledLedger(
                external, elastic, stored_change, interface_change,
                mechanical_heat, phase_heat_mean, thermal_change,
                global_closure, thermal_closure, bath_heat, plastic_work,
            ),
            dt_s,
            halvings,
            float(np.mean(storage_scale < 1.0 - 8.0 * np.finfo(float).eps)),
        )
    raise RuntimeError(
        "no admissible local coupled step found; "
        f"last_rejection={last_rejection}; final_trial_dt_s={dt_s:.16g}"
    )


def advance_local_coupled(
    state: LocalCoupledState,
    applied_shear_rate_s_inv: float,
    dx_m: float,
    proposed_dt_s: float,
    steps: int,
    law: ExpFloorLaw,
    parameters: SpatialCoupledParameters,
    *,
    controls: SpatialMechanismControls = SpatialMechanismControls(),
) -> tuple[LocalCoupledState, tuple[LocalCoupledStep, ...]]:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    current = state
    accepted = []
    for _ in range(steps):
        step = local_coupled_step(
            current, applied_shear_rate_s_inv, dx_m, proposed_dt_s,
            law, parameters, controls=controls,
        )
        current = step.state
        accepted.append(step)
    return current, tuple(accepted)


def save_local_coupled_checkpoint(path: Path, state: LocalCoupledState) -> None:
    np.savez(
        Path(path),
        schema=np.asarray(CHECKPOINT_SCHEMA),
        applied_shear=np.asarray(state.applied_shear),
        plastic_shear=state.plastic_shear,
        temperature_K=state.temperature_K,
        forest_density_m2=state.forest_density_m2,
        eta_fields=state.eta_fields,
        time_s=np.asarray(state.time_s),
        accepted_steps=np.asarray(state.accepted_steps, dtype=np.int64),
    )


def load_local_coupled_checkpoint(path: Path) -> LocalCoupledState:
    with np.load(Path(path), allow_pickle=False) as payload:
        if str(payload["schema"]) != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported local coupled checkpoint schema")
        return LocalCoupledState(
            float(payload["applied_shear"]),
            np.array(payload["plastic_shear"], copy=True),
            np.array(payload["temperature_K"], copy=True),
            np.array(payload["forest_density_m2"], copy=True),
            np.array(payload["eta_fields"], copy=True),
            float(payload["time_s"]),
            int(payload["accepted_steps"]),
        )
