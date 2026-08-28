"""Matched mechanism traces for the locally equilibrated coupled kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytical import ExpFloorLaw
from .local_coupled import LocalCoupledState, LocalCoupledStep, local_coupled_step
from .localization import (
    LocalizationCriteria,
    LocalizationDecision,
    classify_localization,
    localization_history,
)
from .mechanism_ladder import MechanismCase, matched_isothermal_case
from .spatial_coupled import SpatialCoupledParameters


@dataclass(frozen=True)
class LocalMechanismTrace:
    case: MechanismCase
    states: tuple[LocalCoupledState, ...]
    steps: tuple[LocalCoupledStep, ...]
    plastic_rate_s_inv: np.ndarray
    temperature_K: np.ndarray
    mean_stress_Pa: np.ndarray
    local_stress_x_Pa: np.ndarray


def run_local_mechanism_trace(
    initial: LocalCoupledState,
    case: MechanismCase,
    dx_m: float,
    proposed_dt_s: float,
    steps: int,
    law: ExpFloorLaw,
    parameters: SpatialCoupledParameters,
) -> LocalMechanismTrace:
    if steps < 1:
        raise ValueError("steps must be positive")
    current = initial
    if case.unload_initial_stress:
        current = LocalCoupledState(
            float(np.mean(initial.plastic_shear)),
            initial.plastic_shear,
            initial.temperature_K,
            initial.forest_density_m2,
            initial.eta_fields,
            initial.time_s,
            initial.accepted_steps,
        )
    states = []
    accepted = []
    rates = []
    for _ in range(steps):
        old_plastic = current.plastic_shear
        step = local_coupled_step(
            current,
            case.applied_shear_rate_s_inv,
            dx_m,
            proposed_dt_s,
            law,
            parameters,
            controls=case.controls,
        )
        current = step.state
        states.append(current)
        accepted.append(step)
        rates.append(np.abs(current.plastic_shear - old_plastic) / step.accepted_dt_s)
    return LocalMechanismTrace(
        case,
        tuple(states),
        tuple(accepted),
        np.stack(rates),
        np.stack([item.temperature_K for item in states]),
        np.asarray([item.equilibrium.mean_stress_Pa for item in accepted]),
        np.stack([item.equilibrium.stress_x_Pa for item in accepted]),
    )


def matched_local_isothermal_trace(
    initial: LocalCoupledState,
    case: MechanismCase,
    dx_m: float,
    proposed_dt_s: float,
    steps: int,
    law: ExpFloorLaw,
    parameters: SpatialCoupledParameters,
) -> LocalMechanismTrace:
    return run_local_mechanism_trace(
        initial,
        matched_isothermal_case(case),
        dx_m,
        proposed_dt_s,
        steps,
        law,
        parameters,
    )


def classify_local_mechanism_trace(
    trace: LocalMechanismTrace,
    matched_isothermal: LocalMechanismTrace,
    dx_m: float,
    interface_width_m: float,
    criteria: LocalizationCriteria,
) -> LocalizationDecision:
    history = localization_history(
        trace.plastic_rate_s_inv,
        trace.temperature_K,
        matched_isothermal.temperature_K,
        trace.mean_stress_Pa,
        dx_m,
    )
    return classify_localization(history, interface_width_m, criteria)
