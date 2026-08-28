"""Binary thermomechanical/phase aggregate with one global energy ledger.

This is an operator-split, spatial-phase/homogeneous-mechanics verification
limit.  It is not a production ASB solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .analytical import ExpFloorLaw
from .multi_order import MultiOrderState, interpolation_h, multi_order_free_energy_J_m
from .stored_energy_drx import (
    StoredEnergyDRXParameters,
    StoredEnergyDRXState,
    stored_dislocation_energy_J_m,
    stored_energy_drx_step,
)


@dataclass(frozen=True)
class CoupledParameters:
    shear_modulus_Pa: float
    volumetric_heat_capacity_J_m3_K: float
    stored_line_energy_J_m: float
    forest_storage_per_plastic_strain_m2: float
    pair_penalty_J_m3: float
    gradient_coefficient_J_m: float
    phase_mobility_m3_J_s: float

    def __post_init__(self) -> None:
        for name in (
            "shear_modulus_Pa",
            "volumetric_heat_capacity_J_m3_K",
            "stored_line_energy_J_m",
            "pair_penalty_J_m3",
            "gradient_coefficient_J_m",
            "phase_mobility_m3_J_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.forest_storage_per_plastic_strain_m2)
            or self.forest_storage_per_plastic_strain_m2 < 0.0
        ):
            raise ValueError(
                "forest_storage_per_plastic_strain_m2 must be finite and nonnegative"
            )

    def phase_parameters(
        self, forest_density_m2: np.ndarray | tuple[float, float]
    ) -> StoredEnergyDRXParameters:
        density = tuple(float(value) for value in forest_density_m2)
        if len(density) != 2:
            raise ValueError("the coupled verification limit requires two grains")
        return StoredEnergyDRXParameters(
            self.pair_penalty_J_m3,
            self.gradient_coefficient_J_m,
            self.phase_mobility_m3_J_s,
            self.stored_line_energy_J_m,
            density,
            self.volumetric_heat_capacity_J_m3_K,
        )


@dataclass(frozen=True)
class CoupledState:
    stress_Pa: float
    applied_shear: float
    grain_plastic_shear: np.ndarray
    forest_density_m2: np.ndarray
    eta_fields: np.ndarray
    temperature_K: float
    time_s: float = 0.0
    accepted_steps: int = 0

    def __post_init__(self) -> None:
        plastic = np.asarray(self.grain_plastic_shear, dtype=float)
        density = np.asarray(self.forest_density_m2, dtype=float)
        fields = np.asarray(self.eta_fields, dtype=float)
        if plastic.shape != (2,) or density.shape != (2,):
            raise ValueError("grain plastic shear and density must each have length two")
        if fields.ndim != 3 or fields.shape[0] != 2 or min(fields.shape[1:]) < 8:
            raise ValueError("eta_fields must have shape (2, rows, columns), at least 8x8")
        if not np.all(np.isfinite(plastic)) or not np.all(np.isfinite(density)):
            raise ValueError("grain state arrays must be finite")
        if np.any(density <= 0.0):
            raise ValueError("forest densities must be positive")
        MultiOrderState(fields, self.time_s, self.accepted_steps)
        for name in ("stress_Pa", "applied_shear", "time_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be finite and positive")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time_s and accepted_steps must be nonnegative")


@dataclass(frozen=True)
class CoupledLedger:
    external_work_J_m3: float
    elastic_energy_change_J_m3: float
    stored_energy_change_J_m3: float
    interface_order_energy_change_J_m3: float
    mechanical_heat_J_m3: float
    phase_heat_J_m3: float
    thermal_energy_change_J_m3: float
    global_closure_error_J_m3: float
    thermal_closure_error_J_m3: float


@dataclass(frozen=True)
class CoupledStep:
    state: CoupledState
    ledger: CoupledLedger
    grain_plastic_rate_s_inv: np.ndarray
    phase_fraction: np.ndarray
    accepted_dt_s: float
    halvings: int


def coupled_step(
    state: CoupledState,
    applied_shear_rate_s_inv: float,
    dx_m: float,
    proposed_dt_s: float,
    law: ExpFloorLaw,
    parameters: CoupledParameters,
    *,
    maximum_halvings: int = 40,
    tolerance_J_m3: float = 1.0e-8,
) -> CoupledStep:
    if not math.isfinite(applied_shear_rate_s_inv):
        raise ValueError("applied_shear_rate_s_inv must be finite")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    if maximum_halvings < 0:
        raise ValueError("maximum_halvings must be nonnegative")

    fields = np.asarray(state.eta_fields, dtype=float)
    phase_fraction = np.mean(interpolation_h(fields), axis=(1, 2))
    if not math.isclose(float(np.sum(phase_fraction)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("binary interpolated phase fractions must sum to one")
    density = np.asarray(state.forest_density_m2, dtype=float)
    plastic_rates = np.zeros(2, dtype=float)
    if state.stress_Pa != 0.0:
        for label in range(2):
            local_stress = abs(state.stress_Pa) / law.taylor_ratio(float(density[label]))
            plastic_rates[label] = math.copysign(
                law.shear_rate_s_inv(
                    local_stress, float(density[label]), state.temperature_K
                ),
                state.stress_Pa,
            )

    old_phase_parameters = parameters.phase_parameters(density)
    old_stored_J_m = stored_dislocation_energy_J_m(fields, dx_m, old_phase_parameters)
    old_total_J_m = multi_order_free_energy_J_m(
        fields, dx_m, old_phase_parameters.phase_parameters
    )
    old_interface_J_m = old_total_J_m - old_stored_J_m
    rows, columns = fields.shape[1:]
    domain_area_m2 = rows * columns * dx_m**2

    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        applied_increment = applied_shear_rate_s_inv * dt_s
        grain_plastic_increment = plastic_rates * dt_s
        mean_plastic_increment = float(np.dot(phase_fraction, grain_plastic_increment))
        new_stress = state.stress_Pa + parameters.shear_modulus_Pa * (
            applied_increment - mean_plastic_increment
        )
        if state.stress_Pa != 0.0 and state.stress_Pa * new_stress < 0.0:
            dt_s *= 0.5
            continue
        mean_stress = 0.5 * (state.stress_Pa + new_stress)
        grain_plastic_work = mean_stress * grain_plastic_increment
        density_increment = (
            parameters.forest_storage_per_plastic_strain_m2
            * np.abs(grain_plastic_increment)
        )
        grain_stored_increment = parameters.stored_line_energy_J_m * density_increment
        grain_heat = grain_plastic_work - grain_stored_increment
        if np.any(grain_heat < -tolerance_J_m3):
            dt_s *= 0.5
            continue
        grain_heat = np.maximum(grain_heat, 0.0)
        mechanical_heat_J_m3 = float(np.dot(phase_fraction, grain_heat))
        new_density = density + density_increment
        temperature_after_mechanics = state.temperature_K + (
            mechanical_heat_J_m3 / parameters.volumetric_heat_capacity_J_m3_K
        )

        new_phase_parameters = parameters.phase_parameters(new_density)
        phase_trial = stored_energy_drx_step(
            StoredEnergyDRXState(
                MultiOrderState(fields, state.time_s, state.accepted_steps),
                temperature_after_mechanics,
            ),
            dx_m,
            dt_s,
            new_phase_parameters,
        )
        if phase_trial.accepted_dt_s < dt_s:
            dt_s = phase_trial.accepted_dt_s
            continue

        final_fields = phase_trial.state.phase.eta_fields
        new_stored_J_m = stored_dislocation_energy_J_m(
            final_fields, dx_m, new_phase_parameters
        )
        new_total_J_m = multi_order_free_energy_J_m(
            final_fields, dx_m, new_phase_parameters.phase_parameters
        )
        new_interface_J_m = new_total_J_m - new_stored_J_m
        stored_change_J_m3 = (new_stored_J_m - old_stored_J_m) / domain_area_m2
        interface_change_J_m3 = (
            new_interface_J_m - old_interface_J_m
        ) / domain_area_m2
        phase_heat_J_m3 = phase_trial.ledger.heat_J_m3

        external_work_J_m3 = mean_stress * applied_increment
        elastic_change_J_m3 = (
            new_stress**2 - state.stress_Pa**2
        ) / (2.0 * parameters.shear_modulus_Pa)
        final_temperature = phase_trial.state.temperature_K
        thermal_change_J_m3 = parameters.volumetric_heat_capacity_J_m3_K * (
            final_temperature - state.temperature_K
        )
        global_closure = (
            external_work_J_m3
            - elastic_change_J_m3
            - stored_change_J_m3
            - interface_change_J_m3
            - mechanical_heat_J_m3
            - phase_heat_J_m3
        )
        thermal_closure = (
            thermal_change_J_m3 - mechanical_heat_J_m3 - phase_heat_J_m3
        )
        scale = max(
            abs(external_work_J_m3),
            abs(elastic_change_J_m3),
            abs(stored_change_J_m3),
            abs(interface_change_J_m3),
            mechanical_heat_J_m3,
            phase_heat_J_m3,
            1.0,
        )
        if abs(global_closure) > 1.0e-9 * scale or abs(thermal_closure) > 1.0e-8 * scale:
            dt_s *= 0.5
            continue

        new_state = CoupledState(
            new_stress,
            state.applied_shear + applied_increment,
            np.asarray(state.grain_plastic_shear) + grain_plastic_increment,
            new_density,
            final_fields,
            final_temperature,
            state.time_s + dt_s,
            state.accepted_steps + 1,
        )
        return CoupledStep(
            new_state,
            CoupledLedger(
                external_work_J_m3,
                elastic_change_J_m3,
                stored_change_J_m3,
                interface_change_J_m3,
                mechanical_heat_J_m3,
                phase_heat_J_m3,
                thermal_change_J_m3,
                global_closure,
                thermal_closure,
            ),
            plastic_rates,
            phase_fraction,
            dt_s,
            halvings,
        )
    raise RuntimeError("no admissible coupled thermomechanical/phase step found")


def advance_coupled(
    state: CoupledState,
    applied_shear_rate_s_inv: float,
    dx_m: float,
    proposed_dt_s: float,
    steps: int,
    law: ExpFloorLaw,
    parameters: CoupledParameters,
) -> tuple[CoupledState, tuple[CoupledLedger, ...]]:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    current = state
    ledgers = []
    for _ in range(steps):
        accepted = coupled_step(
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


def save_coupled_checkpoint(path: Path, state: CoupledState) -> None:
    np.savez(
        Path(path),
        stress_Pa=np.asarray(state.stress_Pa),
        applied_shear=np.asarray(state.applied_shear),
        grain_plastic_shear=state.grain_plastic_shear,
        forest_density_m2=state.forest_density_m2,
        eta_fields=state.eta_fields,
        temperature_K=np.asarray(state.temperature_K),
        time_s=np.asarray(state.time_s),
        accepted_steps=np.asarray(state.accepted_steps, dtype=np.int64),
    )


def load_coupled_checkpoint(path: Path) -> CoupledState:
    with np.load(Path(path), allow_pickle=False) as payload:
        return CoupledState(
            float(payload["stress_Pa"]),
            float(payload["applied_shear"]),
            np.array(payload["grain_plastic_shear"], copy=True),
            np.array(payload["forest_density_m2"], copy=True),
            np.array(payload["eta_fields"], copy=True),
            float(payload["temperature_K"]),
            float(payload["time_s"]),
            int(payload["accepted_steps"]),
        )
