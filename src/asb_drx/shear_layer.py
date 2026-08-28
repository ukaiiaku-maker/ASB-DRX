"""Periodic 1-D common-stress thermomechanical shear-layer verification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .analytical import ExpFloorLaw


@dataclass(frozen=True)
class ShearLayerParameters:
    shear_modulus_Pa: float
    volumetric_heat_capacity_J_m3_K: float
    thermal_conductivity_W_m_K: float
    stored_line_energy_J_m: float
    forest_storage_per_plastic_strain_m2: float

    def __post_init__(self) -> None:
        for name in (
            "shear_modulus_Pa",
            "volumetric_heat_capacity_J_m3_K",
            "stored_line_energy_J_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "thermal_conductivity_W_m_K",
            "forest_storage_per_plastic_strain_m2",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class ShearLayerState:
    stress_Pa: float
    applied_shear: float
    plastic_shear: np.ndarray
    temperature_K: np.ndarray
    forest_density_m2: np.ndarray
    time_s: float = 0.0
    accepted_steps: int = 0

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(getattr(self, name), dtype=float) for name in (
            "plastic_shear", "temperature_K", "forest_density_m2"
        ))
        if any(array.ndim != 1 or array.size < 3 for array in arrays):
            raise ValueError("layer arrays must be one-dimensional with at least three cells")
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("layer arrays must have identical shapes")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("layer arrays must be finite")
        if np.any(arrays[1] <= 0.0):
            raise ValueError("temperature_K must be positive")
        if np.any(arrays[2] <= 0.0):
            raise ValueError("forest_density_m2 must be positive")
        for name in ("stress_Pa", "applied_shear", "time_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time_s and accepted_steps must be nonnegative")


@dataclass(frozen=True)
class ShearLayerLedger:
    external_work_J_m3: float
    elastic_energy_change_J_m3: float
    mean_stored_dislocation_J_m3: float
    mean_heat_generated_J_m3: float
    mean_thermal_energy_change_J_m3: float
    mechanical_closure_error_J_m3: float
    thermal_closure_error_J_m3: float


@dataclass(frozen=True)
class ShearLayerStep:
    state: ShearLayerState
    ledger: ShearLayerLedger
    plastic_rate_s_inv: np.ndarray
    accepted_dt_s: float
    halvings: int


def shear_layer_step(
    state: ShearLayerState,
    applied_shear_rate_s_inv: float,
    dx_m: float,
    proposed_dt_s: float,
    law: ExpFloorLaw,
    parameters: ShearLayerParameters,
    *,
    maximum_halvings: int = 40,
    tolerance_J_m3: float = 1.0e-8,
) -> ShearLayerStep:
    if not math.isfinite(applied_shear_rate_s_inv):
        raise ValueError("applied_shear_rate_s_inv must be finite")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")

    temperatures = np.asarray(state.temperature_K, dtype=float)
    densities = np.asarray(state.forest_density_m2, dtype=float)
    plastic_rates = np.zeros_like(temperatures)
    if state.stress_Pa != 0.0:
        for index, (temperature, density) in enumerate(zip(temperatures, densities)):
            q = law.taylor_ratio(float(density))
            local_stress = abs(state.stress_Pa) / q
            plastic_rates[index] = math.copysign(
                law.shear_rate_s_inv(local_stress, float(density), float(temperature)),
                state.stress_Pa,
            )

    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        diffusivity = (
            parameters.thermal_conductivity_W_m_K
            / parameters.volumetric_heat_capacity_J_m3_K
        )
        if diffusivity > 0.0 and dt_s * diffusivity / dx_m**2 > 0.5:
            dt_s *= 0.5
            continue
        applied_increment = applied_shear_rate_s_inv * dt_s
        plastic_increment = plastic_rates * dt_s
        mean_plastic_increment = float(np.mean(plastic_increment))
        new_stress = state.stress_Pa + parameters.shear_modulus_Pa * (
            applied_increment - mean_plastic_increment
        )
        if state.stress_Pa != 0.0 and state.stress_Pa * new_stress < 0.0:
            dt_s *= 0.5
            continue
        mean_stress = 0.5 * (state.stress_Pa + new_stress)
        local_plastic_work = mean_stress * plastic_increment
        forest_increment = (
            parameters.forest_storage_per_plastic_strain_m2 * np.abs(plastic_increment)
        )
        local_stored = parameters.stored_line_energy_J_m * forest_increment
        local_heat = local_plastic_work - local_stored
        if np.any(local_heat < -tolerance_J_m3):
            dt_s *= 0.5
            continue
        local_heat = np.maximum(local_heat, 0.0)

        source_temperature = temperatures + (
            local_heat / parameters.volumetric_heat_capacity_J_m3_K
        )
        laplacian = (
            np.roll(source_temperature, -1)
            - 2.0 * source_temperature
            + np.roll(source_temperature, 1)
        ) / dx_m**2
        new_temperature = source_temperature + dt_s * diffusivity * laplacian
        if np.any(~np.isfinite(new_temperature)) or np.any(new_temperature <= 0.0):
            dt_s *= 0.5
            continue

        external_work = mean_stress * applied_increment
        elastic_change = (
            new_stress**2 - state.stress_Pa**2
        ) / (2.0 * parameters.shear_modulus_Pa)
        mean_stored = float(np.mean(local_stored))
        mean_heat = float(np.mean(local_heat))
        mean_thermal_change = parameters.volumetric_heat_capacity_J_m3_K * float(
            np.mean(new_temperature - temperatures)
        )
        mechanical_closure = external_work - elastic_change - mean_stored - mean_heat
        thermal_closure = mean_thermal_change - mean_heat
        scale = max(abs(external_work), abs(elastic_change), mean_stored, mean_heat, 1.0)
        if abs(mechanical_closure) > 1.0e-10 * scale or abs(thermal_closure) > 1.0e-8 * scale:
            dt_s *= 0.5
            continue

        new_state = ShearLayerState(
            stress_Pa=new_stress,
            applied_shear=state.applied_shear + applied_increment,
            plastic_shear=np.asarray(state.plastic_shear) + plastic_increment,
            temperature_K=new_temperature,
            forest_density_m2=densities + forest_increment,
            time_s=state.time_s + dt_s,
            accepted_steps=state.accepted_steps + 1,
        )
        return ShearLayerStep(
            new_state,
            ShearLayerLedger(
                external_work,
                elastic_change,
                mean_stored,
                mean_heat,
                mean_thermal_change,
                mechanical_closure,
                thermal_closure,
            ),
            plastic_rates,
            dt_s,
            halvings,
        )
    raise RuntimeError("no admissible shear-layer step found")


def advance_shear_layer(
    state: ShearLayerState,
    applied_shear_rate_s_inv: float,
    dx_m: float,
    proposed_dt_s: float,
    steps: int,
    law: ExpFloorLaw,
    parameters: ShearLayerParameters,
) -> tuple[ShearLayerState, tuple[ShearLayerLedger, ...]]:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    current = state
    ledgers = []
    for _ in range(steps):
        accepted = shear_layer_step(
            current,
            applied_shear_rate_s_inv,
            dx_m,
            proposed_dt_s,
            law,
            parameters,
        )
        current = accepted.state
        ledgers.append(accepted.ledger)
    return current, tuple(ledgers)


def save_shear_layer_checkpoint(path: str | Path, state: ShearLayerState) -> None:
    np.savez(
        Path(path),
        schema=np.array("asb-drx-shear-layer-checkpoint/v1"),
        stress_Pa=np.array(state.stress_Pa),
        applied_shear=np.array(state.applied_shear),
        plastic_shear=np.asarray(state.plastic_shear),
        temperature_K=np.asarray(state.temperature_K),
        forest_density_m2=np.asarray(state.forest_density_m2),
        time_s=np.array(state.time_s),
        accepted_steps=np.array(state.accepted_steps, dtype=np.int64),
    )


def load_shear_layer_checkpoint(path: str | Path) -> ShearLayerState:
    with np.load(Path(path), allow_pickle=False) as archive:
        schema = str(archive["schema"])
        if schema != "asb-drx-shear-layer-checkpoint/v1":
            raise ValueError(f"unsupported checkpoint schema: {schema}")
        return ShearLayerState(
            stress_Pa=float(archive["stress_Pa"]),
            applied_shear=float(archive["applied_shear"]),
            plastic_shear=np.array(archive["plastic_shear"], copy=True),
            temperature_K=np.array(archive["temperature_K"], copy=True),
            forest_density_m2=np.array(archive["forest_density_m2"], copy=True),
            time_s=float(archive["time_s"]),
            accepted_steps=int(archive["accepted_steps"]),
        )
