"""Dislocation-stored-energy coupling for constrained grain competition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .multi_order import (
    MultiOrderParameters,
    MultiOrderState,
    energy_checked_multi_order_step,
    interpolation_h,
    multi_order_free_energy_J_m,
)


@dataclass(frozen=True)
class StoredEnergyDRXParameters:
    pair_penalty_J_m3: float
    gradient_coefficient_J_m: float
    mobility_m3_J_s: float
    stored_line_energy_J_m: float
    grain_dislocation_density_m2: tuple[float, ...]
    volumetric_heat_capacity_J_m3_K: float

    def __post_init__(self) -> None:
        for name in (
            "pair_penalty_J_m3",
            "gradient_coefficient_J_m",
            "mobility_m3_J_s",
            "stored_line_energy_J_m",
            "volumetric_heat_capacity_J_m3_K",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if len(self.grain_dislocation_density_m2) < 2:
            raise ValueError("at least two grain densities are required")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in self.grain_dislocation_density_m2
        ):
            raise ValueError("grain dislocation densities must be finite and nonnegative")

    @property
    def bulk_energy_J_m3(self) -> tuple[float, ...]:
        return tuple(
            self.stored_line_energy_J_m * density
            for density in self.grain_dislocation_density_m2
        )

    @property
    def phase_parameters(self) -> MultiOrderParameters:
        return MultiOrderParameters(
            self.pair_penalty_J_m3,
            self.gradient_coefficient_J_m,
            self.mobility_m3_J_s,
            self.bulk_energy_J_m3,
        )

    def driving_energy_J_m3(self, parent_label: int, child_label: int) -> float:
        try:
            parent = self.bulk_energy_J_m3[parent_label]
            child = self.bulk_energy_J_m3[child_label]
        except IndexError as error:
            raise ValueError("grain label is outside the density table") from error
        return parent - child


@dataclass(frozen=True)
class StoredEnergyDRXState:
    phase: MultiOrderState
    temperature_K: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be finite and positive")


@dataclass(frozen=True)
class StoredEnergyDRXLedger:
    stored_energy_change_J_m: float
    interfacial_energy_change_J_m: float
    free_energy_change_J_m: float
    heat_J_m: float
    heat_J_m3: float
    closure_error_J_m: float


@dataclass(frozen=True)
class StoredEnergyDRXStep:
    state: StoredEnergyDRXState
    ledger: StoredEnergyDRXLedger
    accepted_dt_s: float
    halvings: int


def stored_dislocation_energy_J_m(
    eta_fields: np.ndarray, dx_m: float, parameters: StoredEnergyDRXParameters
) -> float:
    fields = np.asarray(eta_fields, dtype=float)
    if fields.ndim != 3 or fields.shape[0] != len(parameters.bulk_energy_J_m3):
        raise ValueError("eta_fields must match the grain density table")
    if not np.all(np.isfinite(fields)) or np.any(fields < 0.0) or np.any(fields > 1.0):
        raise ValueError("eta_fields must be finite and in [0, 1]")
    if not np.allclose(np.sum(fields, axis=0), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("eta_fields must sum pointwise to one")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
    density = np.tensordot(
        np.asarray(parameters.bulk_energy_J_m3), interpolation_h(fields), axes=(0, 0)
    )
    return float(dx_m**2 * np.sum(density))


def stored_energy_drx_step(
    state: StoredEnergyDRXState,
    dx_m: float,
    proposed_dt_s: float,
    parameters: StoredEnergyDRXParameters,
) -> StoredEnergyDRXStep:
    phase_parameters = parameters.phase_parameters
    old_total = multi_order_free_energy_J_m(
        state.phase.eta_fields, dx_m, phase_parameters
    )
    old_stored = stored_dislocation_energy_J_m(
        state.phase.eta_fields, dx_m, parameters
    )
    accepted = energy_checked_multi_order_step(
        state.phase.eta_fields, dx_m, proposed_dt_s, phase_parameters
    )
    new_total = accepted.free_energy_J_m
    new_stored = stored_dislocation_energy_J_m(
        accepted.eta_fields, dx_m, parameters
    )
    stored_change = new_stored - old_stored
    interfacial_change = (new_total - new_stored) - (old_total - old_stored)
    free_change = new_total - old_total
    heat_J_m = -free_change
    if heat_J_m < -1.0e-14:
        raise RuntimeError("accepted phase step increased free energy")
    heat_J_m = max(heat_J_m, 0.0)
    rows, columns = accepted.eta_fields.shape[1:]
    domain_area_m2 = rows * columns * dx_m**2
    heat_J_m3 = heat_J_m / domain_area_m2
    closure = -free_change - heat_J_m
    new_phase = MultiOrderState(
        accepted.eta_fields,
        state.phase.time_s + accepted.accepted_dt_s,
        state.phase.accepted_steps + 1,
    )
    new_state = StoredEnergyDRXState(
        new_phase,
        state.temperature_K
        + heat_J_m3 / parameters.volumetric_heat_capacity_J_m3_K,
    )
    return StoredEnergyDRXStep(
        new_state,
        StoredEnergyDRXLedger(
            stored_change,
            interfacial_change,
            free_change,
            heat_J_m,
            heat_J_m3,
            closure,
        ),
        accepted.accepted_dt_s,
        accepted.halvings,
    )


def advance_stored_energy_drx(
    state: StoredEnergyDRXState,
    dx_m: float,
    proposed_dt_s: float,
    steps: int,
    parameters: StoredEnergyDRXParameters,
) -> tuple[StoredEnergyDRXState, tuple[StoredEnergyDRXLedger, ...]]:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    current = state
    ledgers = []
    for _ in range(steps):
        accepted = stored_energy_drx_step(current, dx_m, proposed_dt_s, parameters)
        current = accepted.state
        ledgers.append(accepted.ledger)
    return current, tuple(ledgers)


def save_stored_energy_drx_checkpoint(path: Path, state: StoredEnergyDRXState) -> None:
    np.savez(
        Path(path),
        eta_fields=state.phase.eta_fields,
        temperature_K=np.asarray(state.temperature_K),
        time_s=np.asarray(state.phase.time_s),
        accepted_steps=np.asarray(state.phase.accepted_steps, dtype=np.int64),
    )


def load_stored_energy_drx_checkpoint(path: Path) -> StoredEnergyDRXState:
    with np.load(Path(path), allow_pickle=False) as payload:
        return StoredEnergyDRXState(
            MultiOrderState(
                np.array(payload["eta_fields"], copy=True),
                float(payload["time_s"]),
                int(payload["accepted_steps"]),
            ),
            float(payload["temperature_K"]),
        )
