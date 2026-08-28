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
    statistics: "LocalTraceStatistics"


@dataclass(frozen=True)
class LocalTraceStatistics:
    accepted_steps: int
    minimum_accepted_dt_s: float
    maximum_halvings: int
    maximum_storage_limited_fraction: float
    steps_with_storage_limiting: int
    maximum_absolute_global_closure_error_J_m3: float
    maximum_absolute_thermal_closure_error_J_m3: float
    maximum_flow_iterations: int
    maximum_flow_residual: float


def _trace_statistics(steps: list[LocalCoupledStep]) -> LocalTraceStatistics:
    return LocalTraceStatistics(
        len(steps),
        min(item.accepted_dt_s for item in steps),
        max(item.halvings for item in steps),
        max(item.storage_limited_fraction for item in steps),
        sum(item.storage_limited_fraction > 0.0 for item in steps),
        max(abs(item.ledger.global_closure_error_J_m3) for item in steps),
        max(abs(item.ledger.thermal_closure_error_J_m3) for item in steps),
        max(item.flow_iterations for item in steps),
        max(item.flow_residual for item in steps),
    )


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
        _trace_statistics(accepted),
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


def _build_trace(
    case: MechanismCase,
    states: list[LocalCoupledState],
    steps: list[LocalCoupledStep],
    rates: list[np.ndarray],
    statistics: LocalTraceStatistics | None = None,
) -> LocalMechanismTrace:
    return LocalMechanismTrace(
        case,
        tuple(states),
        tuple(steps),
        np.stack(rates),
        np.stack([item.temperature_K for item in states]),
        np.asarray([item.equilibrium.mean_stress_Pa for item in steps]),
        np.stack([item.equilibrium.stress_x_Pa for item in steps]),
        statistics if statistics is not None else _trace_statistics(steps),
    )


def run_matched_local_strain_pair(
    initial: LocalCoupledState,
    case: MechanismCase,
    dx_m: float,
    maximum_dt_s: float,
    target_applied_shear_increment: float,
    law: ExpFloorLaw,
    parameters: SpatialCoupledParameters,
    *,
    maximum_accepted_steps: int = 100_000,
    retention_strain_increment: float | None = None,
) -> tuple[LocalMechanismTrace, LocalMechanismTrace]:
    """Advance thermal and isothermal cases on one strain/time grid.

    A trial accepted at different timesteps by the two cases is repeated for
    both at the smaller timestep.  This makes every matched temperature and
    stress comparison pointwise in physical time and applied strain.
    """
    if case.applied_shear_rate_s_inv == 0.0:
        raise ValueError("a nonzero applied shear rate is required")
    if target_applied_shear_increment <= 0.0 or maximum_dt_s <= 0.0:
        raise ValueError("target strain and maximum timestep must be positive")
    if maximum_accepted_steps < 1:
        raise ValueError("maximum_accepted_steps must be positive")
    if retention_strain_increment is not None and (
        not np.isfinite(retention_strain_increment)
        or retention_strain_increment <= 0.0
    ):
        raise ValueError("retention_strain_increment must be finite and positive")
    if case.unload_initial_stress:
        raise ValueError("strain-targeted matched traces do not support initial unloading")

    control_case = matched_isothermal_case(case)
    thermal = initial
    control = initial
    thermal_states: list[LocalCoupledState] = []
    control_states: list[LocalCoupledState] = []
    thermal_steps: list[LocalCoupledStep] = []
    control_steps: list[LocalCoupledStep] = []
    thermal_rates: list[np.ndarray] = []
    control_rates: list[np.ndarray] = []
    rate_magnitude = abs(case.applied_shear_rate_s_inv)
    initial_applied = initial.applied_shear
    tolerance = 64.0 * np.finfo(float).eps * target_applied_shear_increment
    next_retained_increment = retention_strain_increment
    total_steps = 0
    minimum_dt = float("inf")
    maximum_halvings = 0
    maximum_storage_limited_fraction = 0.0
    steps_with_storage_limiting = 0
    maximum_global_closure = 0.0
    maximum_thermal_closure = 0.0
    maximum_flow_iterations = 0
    maximum_flow_residual = 0.0

    while True:
        accumulated = abs(thermal.applied_shear - initial_applied)
        remaining = target_applied_shear_increment - accumulated
        if remaining <= tolerance:
            break
        if total_steps >= maximum_accepted_steps:
            raise RuntimeError("matched strain trace exceeded maximum_accepted_steps")
        trial_dt = min(maximum_dt_s, remaining / rate_magnitude)
        while True:
            thermal_trial = local_coupled_step(
                thermal, case.applied_shear_rate_s_inv, dx_m, trial_dt,
                law, parameters, controls=case.controls,
            )
            control_trial = local_coupled_step(
                control, control_case.applied_shear_rate_s_inv, dx_m, trial_dt,
                law, parameters, controls=control_case.controls,
            )
            common_dt = min(
                thermal_trial.accepted_dt_s, control_trial.accepted_dt_s
            )
            if (
                thermal_trial.accepted_dt_s == common_dt
                and control_trial.accepted_dt_s == common_dt
            ):
                break
            trial_dt = common_dt

        old_thermal_plastic = thermal.plastic_shear
        old_control_plastic = control.plastic_shear
        thermal = thermal_trial.state
        control = control_trial.state
        total_steps += 1
        minimum_dt = min(minimum_dt, common_dt)
        maximum_halvings = max(
            maximum_halvings, thermal_trial.halvings, control_trial.halvings
        )
        maximum_storage_limited_fraction = max(
            maximum_storage_limited_fraction,
            thermal_trial.storage_limited_fraction,
            control_trial.storage_limited_fraction,
        )
        steps_with_storage_limiting += (
            thermal_trial.storage_limited_fraction > 0.0
            or control_trial.storage_limited_fraction > 0.0
        )
        maximum_global_closure = max(
            maximum_global_closure,
            abs(thermal_trial.ledger.global_closure_error_J_m3),
            abs(control_trial.ledger.global_closure_error_J_m3),
        )
        maximum_thermal_closure = max(
            maximum_thermal_closure,
            abs(thermal_trial.ledger.thermal_closure_error_J_m3),
            abs(control_trial.ledger.thermal_closure_error_J_m3),
        )
        maximum_flow_iterations = max(
            maximum_flow_iterations,
            thermal_trial.flow_iterations,
            control_trial.flow_iterations,
        )
        maximum_flow_residual = max(
            maximum_flow_residual,
            thermal_trial.flow_residual,
            control_trial.flow_residual,
        )
        accumulated = abs(thermal.applied_shear - initial_applied)
        retain = retention_strain_increment is None or (
            next_retained_increment is not None
            and accumulated + tolerance >= next_retained_increment
        )
        if retain:
            thermal_states.append(thermal)
            control_states.append(control)
            thermal_steps.append(thermal_trial)
            control_steps.append(control_trial)
            thermal_rates.append(
                np.abs(thermal.plastic_shear - old_thermal_plastic) / common_dt
            )
            control_rates.append(
                np.abs(control.plastic_shear - old_control_plastic) / common_dt
            )
            if next_retained_increment is not None:
                while next_retained_increment <= accumulated + tolerance:
                    next_retained_increment += retention_strain_increment

    if not thermal_states or thermal_states[-1] is not thermal:
        thermal_states.append(thermal)
        control_states.append(control)
        thermal_steps.append(thermal_trial)
        control_steps.append(control_trial)
        thermal_rates.append(
            np.abs(thermal.plastic_shear - old_thermal_plastic) / common_dt
        )
        control_rates.append(
            np.abs(control.plastic_shear - old_control_plastic) / common_dt
        )

    statistics = LocalTraceStatistics(
        total_steps,
        minimum_dt,
        maximum_halvings,
        maximum_storage_limited_fraction,
        steps_with_storage_limiting,
        maximum_global_closure,
        maximum_thermal_closure,
        maximum_flow_iterations,
        maximum_flow_residual,
    )

    return (
        _build_trace(case, thermal_states, thermal_steps, thermal_rates, statistics),
        _build_trace(
            control_case, control_states, control_steps, control_rates, statistics
        ),
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
