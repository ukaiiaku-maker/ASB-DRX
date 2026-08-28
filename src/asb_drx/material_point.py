"""Finite-loading thermomechanical material-point verification model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .analytical import ExpFloorLaw
from .thermodynamics import DislocationReservoirs, close_work_ledger


@dataclass(frozen=True)
class MaterialPointParameters:
    shear_modulus_Pa: float
    volumetric_heat_capacity_J_m3_K: float
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
        value = self.forest_storage_per_plastic_strain_m2
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "forest_storage_per_plastic_strain_m2 must be finite and nonnegative"
            )


@dataclass(frozen=True)
class MaterialPointState:
    stress_Pa: float
    applied_shear: float
    plastic_shear: float
    temperature_K: float
    reservoirs: DislocationReservoirs
    time_s: float = 0.0
    accepted_steps: int = 0

    def __post_init__(self) -> None:
        for name in ("stress_Pa", "applied_shear", "plastic_shear", "time_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be finite and positive")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time_s and accepted_steps must be nonnegative")
        if self.reservoirs.forest_m2 <= 0.0:
            raise ValueError("forest_m2 must be positive for the analytical baseline")


@dataclass(frozen=True)
class MaterialPointLedger:
    external_work_J_m3: float
    elastic_energy_change_J_m3: float
    plastic_work_J_m3: float
    stored_dislocation_J_m3: float
    heat_J_m3: float
    closure_error_J_m3: float


@dataclass(frozen=True)
class MaterialPointStep:
    state: MaterialPointState
    ledger: MaterialPointLedger
    plastic_rate_s_inv: float
    local_activation_stress_Pa: float
    accepted_dt_s: float
    halvings: int


def material_point_step(
    state: MaterialPointState,
    applied_shear_rate_s_inv: float,
    proposed_dt_s: float,
    law: ExpFloorLaw,
    parameters: MaterialPointParameters,
    *,
    maximum_halvings: int = 40,
    closure_tolerance_J_m3: float = 1.0e-8,
) -> MaterialPointStep:
    """Take an energy-admissible finite-loading step.

    The stress update is exact for the increment:
      delta_tau = G (delta_gamma_applied - delta_gamma_plastic).
    Trapezoidal external and elastic work then leave an exact plastic-work
    residual, which is partitioned into stored line energy and heat.
    """

    if not math.isfinite(applied_shear_rate_s_inv):
        raise ValueError("applied_shear_rate_s_inv must be finite")
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    if maximum_halvings < 0:
        raise ValueError("maximum_halvings must be nonnegative")

    q = law.taylor_ratio(state.reservoirs.forest_m2)
    local_stress = abs(state.stress_Pa) / q
    if state.stress_Pa == 0.0:
        plastic_rate = 0.0
    else:
        plastic_rate = math.copysign(
            law.shear_rate_s_inv(local_stress, state.reservoirs.forest_m2, state.temperature_K),
            state.stress_Pa,
        )

    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        applied_increment = applied_shear_rate_s_inv * dt_s
        plastic_increment = plastic_rate * dt_s
        new_stress = state.stress_Pa + parameters.shear_modulus_Pa * (
            applied_increment - plastic_increment
        )
        if state.stress_Pa != 0.0 and state.stress_Pa * new_stress < 0.0:
            dt_s *= 0.5
            continue

        mean_stress = 0.5 * (state.stress_Pa + new_stress)
        external_work = mean_stress * applied_increment
        elastic_change = (
            new_stress**2 - state.stress_Pa**2
        ) / (2.0 * parameters.shear_modulus_Pa)
        plastic_work = external_work - elastic_change
        if plastic_work < -closure_tolerance_J_m3:
            dt_s *= 0.5
            continue

        forest_increment = (
            parameters.forest_storage_per_plastic_strain_m2 * abs(plastic_increment)
        )
        stored_energy = parameters.stored_line_energy_J_m * forest_increment
        try:
            partition = close_work_ledger(
                max(plastic_work, 0.0),
                stored_dislocation_J_m3=stored_energy,
                tolerance_J_m3=closure_tolerance_J_m3,
            )
        except ValueError:
            dt_s *= 0.5
            continue

        reservoirs = DislocationReservoirs(
            mobile_m2=state.reservoirs.mobile_m2,
            forest_m2=state.reservoirs.forest_m2 + forest_increment,
            wall_m2=state.reservoirs.wall_m2,
            gb_m2=state.reservoirs.gb_m2,
        )
        new_temperature = state.temperature_K + (
            partition.heat_J_m3 / parameters.volumetric_heat_capacity_J_m3_K
        )
        new_state = MaterialPointState(
            stress_Pa=new_stress,
            applied_shear=state.applied_shear + applied_increment,
            plastic_shear=state.plastic_shear + plastic_increment,
            temperature_K=new_temperature,
            reservoirs=reservoirs,
            time_s=state.time_s + dt_s,
            accepted_steps=state.accepted_steps + 1,
        )
        closure = external_work - elastic_change - stored_energy - partition.heat_J_m3
        ledger = MaterialPointLedger(
            external_work,
            elastic_change,
            max(plastic_work, 0.0),
            stored_energy,
            partition.heat_J_m3,
            closure,
        )
        return MaterialPointStep(
            new_state, ledger, plastic_rate, local_stress, dt_s, halvings
        )
    raise RuntimeError("no energetically admissible material-point step found")


def advance_material_point(
    state: MaterialPointState,
    applied_shear_rate_s_inv: float,
    proposed_dt_s: float,
    steps: int,
    law: ExpFloorLaw,
    parameters: MaterialPointParameters,
) -> tuple[MaterialPointState, tuple[MaterialPointLedger, ...]]:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    ledgers = []
    current = state
    for _ in range(steps):
        accepted = material_point_step(
            current, applied_shear_rate_s_inv, proposed_dt_s, law, parameters
        )
        current = accepted.state
        ledgers.append(accepted.ledger)
    return current, tuple(ledgers)


def save_material_point_checkpoint(path: str | Path, state: MaterialPointState) -> None:
    np.savez(
        Path(path),
        schema=np.array("asb-drx-material-point-checkpoint/v1"),
        stress_Pa=np.array(state.stress_Pa),
        applied_shear=np.array(state.applied_shear),
        plastic_shear=np.array(state.plastic_shear),
        temperature_K=np.array(state.temperature_K),
        mobile_m2=np.array(state.reservoirs.mobile_m2),
        forest_m2=np.array(state.reservoirs.forest_m2),
        wall_m2=np.array(state.reservoirs.wall_m2),
        gb_m2=np.array(state.reservoirs.gb_m2),
        time_s=np.array(state.time_s),
        accepted_steps=np.array(state.accepted_steps, dtype=np.int64),
    )


def load_material_point_checkpoint(path: str | Path) -> MaterialPointState:
    with np.load(Path(path), allow_pickle=False) as archive:
        schema = str(archive["schema"])
        if schema != "asb-drx-material-point-checkpoint/v1":
            raise ValueError(f"unsupported checkpoint schema: {schema}")
        reservoirs = DislocationReservoirs(
            float(archive["mobile_m2"]),
            float(archive["forest_m2"]),
            float(archive["wall_m2"]),
            float(archive["gb_m2"]),
        )
        return MaterialPointState(
            stress_Pa=float(archive["stress_Pa"]),
            applied_shear=float(archive["applied_shear"]),
            plastic_shear=float(archive["plastic_shear"]),
            temperature_K=float(archive["temperature_K"]),
            reservoirs=reservoirs,
            time_s=float(archive["time_s"]),
            accepted_steps=int(archive["accepted_steps"]),
        )
