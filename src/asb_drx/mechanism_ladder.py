"""Matched common-equation controls for thermomechanical mechanism attribution."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .analytical import ExpFloorLaw
from .localization import LocalizationCriteria, LocalizationDecision, classify_localization, localization_history
from .spatial_coupled import (
    SpatialCoupledLedger, SpatialCoupledParameters, SpatialCoupledState,
    SpatialMechanismControls, spatial_coupled_step,
)


@dataclass(frozen=True)
class MechanismCase:
    name: str
    applied_shear_rate_s_inv: float
    controls: SpatialMechanismControls
    unload_initial_stress: bool = False

    def __post_init__(self) -> None:
        if not self.name or not math.isfinite(self.applied_shear_rate_s_inv):
            raise ValueError("mechanism case requires a name and finite rate")
        if not isinstance(self.unload_initial_stress, bool):
            raise TypeError("unload_initial_stress must be boolean")


@dataclass(frozen=True)
class MechanismTrace:
    case: MechanismCase
    states: tuple[SpatialCoupledState, ...]
    ledgers: tuple[SpatialCoupledLedger, ...]
    plastic_rate_s_inv: np.ndarray
    temperature_K: np.ndarray
    stress_Pa: np.ndarray


def standard_mechanism_ladder(intermediate_rate_s_inv: float, high_rate_s_inv: float) -> tuple[MechanismCase, ...]:
    if not 0.0 < intermediate_rate_s_inv < high_rate_s_inv:
        raise ValueError("rates must satisfy 0 < intermediate < high")
    return (
        MechanismCase("unloaded_isothermal_relaxation", 0.0, SpatialMechanismControls(False, True), True),
        MechanismCase("isothermal_deformation_phase_disabled", intermediate_rate_s_inv, SpatialMechanismControls(False, False)),
        MechanismCase("isothermal_drx", intermediate_rate_s_inv, SpatialMechanismControls(False, True)),
        MechanismCase("thermal_high_rate_phase_disabled", high_rate_s_inv, SpatialMechanismControls(True, False)),
        MechanismCase("coupled_intermediate_rate", intermediate_rate_s_inv, SpatialMechanismControls(True, True)),
        MechanismCase("coupled_high_rate", high_rate_s_inv, SpatialMechanismControls(True, True)),
    )


def matched_isothermal_case(case: MechanismCase) -> MechanismCase:
    return MechanismCase(
        f"{case.name}__matched_isothermal",
        case.applied_shear_rate_s_inv,
        SpatialMechanismControls(False, case.controls.evolve_phase),
        case.unload_initial_stress,
    )


def run_mechanism_trace(
    initial: SpatialCoupledState, case: MechanismCase, dx_m: float,
    proposed_dt_s: float, steps: int, law: ExpFloorLaw,
    parameters: SpatialCoupledParameters,
) -> MechanismTrace:
    if steps < 1:
        raise ValueError("steps must be positive")
    states = []
    ledgers = []
    rates = []
    current = initial
    if case.unload_initial_stress:
        current = SpatialCoupledState(
            0.0, initial.applied_shear, initial.plastic_shear,
            initial.temperature_K, initial.forest_density_m2,
            initial.eta_fields, initial.time_s, initial.accepted_steps,
        )
    for _ in range(steps):
        accepted = spatial_coupled_step(
            current, case.applied_shear_rate_s_inv, dx_m, proposed_dt_s,
            law, parameters, controls=case.controls,
        )
        rates.append(np.abs(accepted.state.plastic_shear - current.plastic_shear) / accepted.accepted_dt_s)
        current = accepted.state
        states.append(current)
        ledgers.append(accepted.ledger)
    return MechanismTrace(
        case, tuple(states), tuple(ledgers), np.stack(rates),
        np.stack([item.temperature_K for item in states]),
        np.asarray([item.stress_Pa for item in states]),
    )


def classify_mechanism_trace(
    trace: MechanismTrace, matched_isothermal: MechanismTrace,
    dx_m: float, interface_width_m: float, criteria: LocalizationCriteria,
) -> LocalizationDecision:
    if trace.temperature_K.shape != matched_isothermal.temperature_K.shape:
        raise ValueError("matched control history shape mismatch")
    history = localization_history(
        trace.plastic_rate_s_inv, trace.temperature_K,
        matched_isothermal.temperature_K, trace.stress_Pa, dx_m,
    )
    return classify_localization(history, interface_width_m, criteria)
