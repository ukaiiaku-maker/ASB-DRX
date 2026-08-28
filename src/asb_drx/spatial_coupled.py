"""Periodic 2-D common-stress thermomechanical/phase verification kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .analytical import ExpFloorLaw
from .multi_order import interpolation_h, interpolation_h_prime


@dataclass(frozen=True)
class SpatialCoupledParameters:
    shear_modulus_Pa: float
    volumetric_heat_capacity_J_m3_K: float
    thermal_conductivity_W_m_K: float
    stored_line_energy_J_m: float
    forest_storage_per_plastic_strain_m2: float
    pair_penalty_J_m3: float
    gradient_coefficient_J_m: float
    phase_mobility_m3_J_s: float

    def __post_init__(self) -> None:
        for name in (
            "shear_modulus_Pa", "volumetric_heat_capacity_J_m3_K",
            "stored_line_energy_J_m", "pair_penalty_J_m3",
            "gradient_coefficient_J_m", "phase_mobility_m3_J_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("thermal_conductivity_W_m_K", "forest_storage_per_plastic_strain_m2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class SpatialCoupledState:
    stress_Pa: float
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
            raise ValueError("all spatial fields must be finite")
        if np.any(temperature <= 0.0) or np.any(density <= 0.0):
            raise ValueError("temperature and densities must be positive")
        if np.any(fields < 0.0) or np.any(fields > 1.0):
            raise ValueError("eta_fields must be in [0, 1]")
        if not np.allclose(np.sum(fields, axis=0), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("eta_fields must sum pointwise to one")
        for name in ("stress_Pa", "applied_shear", "time_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time_s and accepted_steps must be nonnegative")


@dataclass(frozen=True)
class SpatialCoupledLedger:
    external_work_J_m3: float
    elastic_change_J_m3: float
    stored_change_J_m3: float
    interface_order_change_J_m3: float
    mechanical_heat_J_m3: float
    phase_heat_J_m3: float
    thermal_change_J_m3: float
    global_closure_error_J_m3: float
    thermal_closure_error_J_m3: float


@dataclass(frozen=True)
class SpatialCoupledStep:
    state: SpatialCoupledState
    ledger: SpatialCoupledLedger
    accepted_dt_s: float
    halvings: int


def _energies_J_m(
    fields: np.ndarray, density_m2: np.ndarray, dx_m: float,
    parameters: SpatialCoupledParameters,
) -> tuple[float, float, float]:
    stored_density = parameters.stored_line_energy_J_m * np.sum(
        density_m2 * interpolation_h(fields), axis=0
    )
    pair_density = parameters.pair_penalty_J_m3 * fields[0] ** 2 * fields[1] ** 2
    gx = (np.roll(fields, -1, axis=2) - fields) / dx_m
    gy = (np.roll(fields, -1, axis=1) - fields) / dx_m
    interface_density = pair_density + 0.5 * parameters.gradient_coefficient_J_m * np.sum(
        gx**2 + gy**2, axis=0
    )
    stored = float(dx_m**2 * np.sum(stored_density))
    interface = float(dx_m**2 * np.sum(interface_density))
    return stored + interface, stored, interface


def _chemical_potential_J_m3(
    fields: np.ndarray, density_m2: np.ndarray, dx_m: float,
    parameters: SpatialCoupledParameters,
) -> np.ndarray:
    other_squared = fields[::-1] ** 2
    local = 2.0 * parameters.pair_penalty_J_m3 * fields * other_squared
    local += parameters.stored_line_energy_J_m * density_m2 * interpolation_h_prime(fields)
    lap = (
        np.roll(fields, -1, axis=1) + np.roll(fields, 1, axis=1)
        + np.roll(fields, -1, axis=2) + np.roll(fields, 1, axis=2) - 4.0 * fields
    ) / dx_m**2
    return local - parameters.gradient_coefficient_J_m * lap


def spatial_coupled_step(
    state: SpatialCoupledState, applied_shear_rate_s_inv: float, dx_m: float,
    proposed_dt_s: float, law: ExpFloorLaw, parameters: SpatialCoupledParameters,
    *, maximum_halvings: int = 40, tolerance_J_m3: float = 1.0e-8,
) -> SpatialCoupledStep:
    if not math.isfinite(applied_shear_rate_s_inv):
        raise ValueError("applied_shear_rate_s_inv must be finite")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    fields = np.asarray(state.eta_fields, dtype=float)
    density = np.asarray(state.forest_density_m2, dtype=float)
    temperature = np.asarray(state.temperature_K, dtype=float)
    weights = interpolation_h(fields)
    if not np.allclose(np.sum(weights, axis=0), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("binary interpolated weights must sum to one")
    rates = np.zeros_like(density)
    if state.stress_Pa != 0.0:
        for label in range(2):
            for index in np.ndindex(temperature.shape):
                local_stress = abs(state.stress_Pa) / law.taylor_ratio(float(density[(label, *index)]))
                rates[(label, *index)] = math.copysign(
                    law.shear_rate_s_inv(local_stress, float(density[(label, *index)]), float(temperature[index])),
                    state.stress_Pa,
                )
    old_total, old_stored, old_interface = _energies_J_m(fields, density, dx_m, parameters)
    area_m2 = fields.shape[1] * fields.shape[2] * dx_m**2
    diffusivity = parameters.thermal_conductivity_W_m_K / parameters.volumetric_heat_capacity_J_m3_K
    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        if diffusivity > 0.0 and diffusivity * dt_s / dx_m**2 > 0.25:
            dt_s *= 0.5
            continue
        grain_increment = rates * dt_s
        local_plastic_increment = np.sum(weights * grain_increment, axis=0)
        mean_plastic_increment = float(np.mean(local_plastic_increment))
        applied_increment = applied_shear_rate_s_inv * dt_s
        new_stress = state.stress_Pa + parameters.shear_modulus_Pa * (
            applied_increment - mean_plastic_increment
        )
        if state.stress_Pa != 0.0 and state.stress_Pa * new_stress < 0.0:
            dt_s *= 0.5
            continue
        mean_stress = 0.5 * (state.stress_Pa + new_stress)
        density_increment = parameters.forest_storage_per_plastic_strain_m2 * np.abs(grain_increment)
        stored_increment_local = parameters.stored_line_energy_J_m * np.sum(weights * density_increment, axis=0)
        mechanical_heat_local = mean_stress * local_plastic_increment - stored_increment_local
        if np.any(mechanical_heat_local < -tolerance_J_m3):
            dt_s *= 0.5
            continue
        mechanical_heat_local = np.maximum(mechanical_heat_local, 0.0)
        new_density = density + density_increment
        source_temperature = temperature + mechanical_heat_local / parameters.volumetric_heat_capacity_J_m3_K
        lap_t = (
            np.roll(source_temperature, -1, axis=0) + np.roll(source_temperature, 1, axis=0)
            + np.roll(source_temperature, -1, axis=1) + np.roll(source_temperature, 1, axis=1)
            - 4.0 * source_temperature
        ) / dx_m**2
        conducted_temperature = source_temperature + dt_s * diffusivity * lap_t
        if np.any(conducted_temperature <= 0.0) or not np.all(np.isfinite(conducted_temperature)):
            dt_s *= 0.5
            continue

        phase_old_total, _, _ = _energies_J_m(fields, new_density, dx_m, parameters)
        chemical = _chemical_potential_J_m3(fields, new_density, dx_m, parameters)
        projected = chemical - np.mean(chemical, axis=0, keepdims=True)
        candidate = fields - dt_s * parameters.phase_mobility_m3_J_s * projected
        if np.any(candidate < -1.0e-14) or np.any(candidate > 1.0 + 1.0e-14) or not np.all(np.isfinite(candidate)):
            dt_s *= 0.5
            continue
        candidate = np.clip(candidate, 0.0, 1.0)
        candidate /= np.sum(candidate, axis=0, keepdims=True)
        new_total, new_stored, new_interface = _energies_J_m(candidate, new_density, dx_m, parameters)
        if new_total > phase_old_total:
            dt_s *= 0.5
            continue
        phase_heat_mean = (phase_old_total - new_total) / area_m2
        dissipation_weight = np.sum(projected**2, axis=0)
        if float(np.mean(dissipation_weight)) > 0.0:
            phase_heat_local = phase_heat_mean * dissipation_weight / float(np.mean(dissipation_weight))
        else:
            phase_heat_local = np.full_like(temperature, phase_heat_mean)
        final_temperature = conducted_temperature + phase_heat_local / parameters.volumetric_heat_capacity_J_m3_K

        external = mean_stress * applied_increment
        elastic = (new_stress**2 - state.stress_Pa**2) / (2.0 * parameters.shear_modulus_Pa)
        stored_change = (new_stored - old_stored) / area_m2
        interface_change = (new_interface - old_interface) / area_m2
        mechanical_heat = float(np.mean(mechanical_heat_local))
        thermal_change = parameters.volumetric_heat_capacity_J_m3_K * float(np.mean(final_temperature - temperature))
        global_closure = external - elastic - stored_change - interface_change - mechanical_heat - phase_heat_mean
        thermal_closure = thermal_change - mechanical_heat - phase_heat_mean
        scale = max(abs(external), abs(elastic), abs(stored_change), abs(interface_change), mechanical_heat, phase_heat_mean, 1.0)
        thermal_floor = 16.0 * np.finfo(float).eps * parameters.volumetric_heat_capacity_J_m3_K * float(np.max(final_temperature))
        if abs(global_closure) > 1.0e-9 * scale or abs(thermal_closure) > max(1.0e-8 * scale, thermal_floor):
            dt_s *= 0.5
            continue
        return SpatialCoupledStep(
            SpatialCoupledState(
                new_stress, state.applied_shear + applied_increment,
                np.asarray(state.plastic_shear) + local_plastic_increment,
                final_temperature, new_density, candidate,
                state.time_s + dt_s, state.accepted_steps + 1,
            ),
            SpatialCoupledLedger(external, elastic, stored_change, interface_change,
                mechanical_heat, phase_heat_mean, thermal_change, global_closure, thermal_closure),
            dt_s, halvings,
        )
    raise RuntimeError("no admissible spatial coupled step found")


def advance_spatial_coupled(
    state: SpatialCoupledState, applied_shear_rate_s_inv: float, dx_m: float,
    proposed_dt_s: float, steps: int, law: ExpFloorLaw,
    parameters: SpatialCoupledParameters,
) -> tuple[SpatialCoupledState, tuple[SpatialCoupledLedger, ...]]:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    current = state
    ledgers = []
    for _ in range(steps):
        accepted = spatial_coupled_step(current, applied_shear_rate_s_inv, dx_m, proposed_dt_s, law, parameters)
        current = accepted.state
        ledgers.append(accepted.ledger)
    return current, tuple(ledgers)


def save_spatial_coupled_checkpoint(path: Path, state: SpatialCoupledState) -> None:
    np.savez(Path(path), stress_Pa=np.asarray(state.stress_Pa), applied_shear=np.asarray(state.applied_shear),
        plastic_shear=state.plastic_shear, temperature_K=state.temperature_K,
        forest_density_m2=state.forest_density_m2, eta_fields=state.eta_fields,
        time_s=np.asarray(state.time_s), accepted_steps=np.asarray(state.accepted_steps, dtype=np.int64))


def load_spatial_coupled_checkpoint(path: Path) -> SpatialCoupledState:
    with np.load(Path(path), allow_pickle=False) as payload:
        return SpatialCoupledState(float(payload["stress_Pa"]), float(payload["applied_shear"]),
            np.array(payload["plastic_shear"], copy=True), np.array(payload["temperature_K"], copy=True),
            np.array(payload["forest_density_m2"], copy=True), np.array(payload["eta_fields"], copy=True),
            float(payload["time_s"]), int(payload["accepted_steps"]))
