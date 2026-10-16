"""Energy-qualified implicit relaxation of the stiff wall-order field.

The ordinary mechanics map advances transport, reactions, slip, and heat.
When a finite-amplitude ordered-wall nucleus is present, its explicit
Allen--Cahn equation can become much faster than those channels.  This module
provides an operator split for that limit: a bounded minimization locates the
nearest downhill state of the *same* wall functional, while the existing
Arrhenius EXP-floor order clock controls how much of that displacement is
allowed during one physical interval.  The optimizer is not a kinetic clock
and cannot publish an uphill state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.optimize import minimize

from .common_tensorial_wall import (
    _divergence, _spectral_gradient, exp_floor_rate,
    wall_free_energy_derivatives, wall_total_free_energy_density_J_m3,
)
from .complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
    evaluate_complete_multigrain_local_fields,
)


@dataclass(frozen=True)
class ImplicitWallOrderRelaxationParameters:
    maximum_fraction_per_interval: float = .15
    owner_support_threshold: float = 1.0e-8
    optimizer_max_iterations: int = 300
    optimizer_gradient_tolerance: float = 1.0e-9
    maximum_backtracks: int = 16

    def __post_init__(self):
        if not 0.0 < self.maximum_fraction_per_interval <= 1.0:
            raise ValueError("wall-order fraction must lie in (0,1]")
        if not 0.0 <= self.owner_support_threshold < 1.0:
            raise ValueError("owner support threshold must lie in [0,1)")
        if self.optimizer_max_iterations < 1 or self.maximum_backtracks < 0:
            raise ValueError("wall-order iteration limits are invalid")


@dataclass(frozen=True)
class ImplicitWallOrderRelaxationDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    owner_fraction_by_id: dict[int, float]
    owner_rate_s_by_id: dict[int, float]
    owner_optimizer_success_by_id: dict[int, bool]
    owner_target_energy_change_J_by_id: dict[int, float]
    backtracks: int
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _owner_target(owner, support, wall_parameters, topologies, systems,
                  parameters):
    """Return the local-basin q minimizer and its supported energy change."""
    q0 = np.asarray(owner.wall_order, dtype=float)
    weight = np.asarray(support, dtype=float)
    active = weight > parameters.owner_support_threshold
    if not np.any(active):
        return q0.copy(), True, 0.0
    active_indices = np.flatnonzero(active.ravel())
    cell_volume = float(wall_parameters.spacing_m)**2
    scale = max(float(wall_parameters.wall_order_barrier_J_m3), 1.0)
    normalization = max(float(np.sum(weight)), 1.0)

    def state_from_vector(vector):
        q = q0.copy().ravel()
        q[active_indices] = vector
        return replace(owner, wall_order=q.reshape(q0.shape))

    reference_density = wall_total_free_energy_density_J_m3(
        owner, wall_parameters, topologies, systems)
    reference = float(np.sum(weight*reference_density, dtype=np.longdouble)
                      *cell_volume)

    def objective(vector):
        trial = state_from_vector(vector)
        density = wall_total_free_energy_density_J_m3(
            trial, wall_parameters, topologies, systems)
        derivatives = wall_free_energy_derivatives(
            trial, wall_parameters, topologies, systems)
        qx, qy = _spectral_gradient(
            trial.wall_order, wall_parameters.spacing_m)
        weighted_flux = np.stack((weight*qx, weight*qy), axis=-1)
        gradient = (weight*derivatives["wall_order_derivative_J_m3"]
                    -wall_parameters.wall_order_gradient_J_m
                    *_divergence(weighted_flux, wall_parameters.spacing_m))
        energy = float(np.sum(weight*density, dtype=np.longdouble)*cell_volume)
        value = (energy-reference)/(scale*normalization*cell_volume)
        jacobian = (gradient.ravel()[active_indices]
                    /(scale*normalization))
        return value, jacobian

    result = minimize(
        objective, q0.ravel()[active_indices], method="L-BFGS-B", jac=True,
        bounds=[(0.0, 1.0)]*len(active_indices),
        options={"ftol": 1.0e-14,
                 "gtol": parameters.optimizer_gradient_tolerance,
                 "maxiter": parameters.optimizer_max_iterations,
                 "maxls": 40})
    target = state_from_vector(result.x).wall_order
    target_density = wall_total_free_energy_density_J_m3(
        replace(owner, wall_order=target), wall_parameters, topologies, systems)
    target_change = float(np.sum(
        weight*(target_density-reference_density), dtype=np.longdouble)
                          *cell_volume)
    # L-BFGS-B can report a line-search warning at a bound after it has found
    # a valid lower state.  Energy monotonicity, not its status string, is the
    # physical acceptance condition; retain status as an audit field.
    return np.asarray(target), bool(result.success), target_change


def evaluate_implicit_wall_order_relaxation(
        state, *, interval_s, mean_strain, spacing_m,
        represented_thickness_m, topologies, systems, wall_parameters,
        energy_kwargs=None,
        parameters=ImplicitWallOrderRelaxationParameters(),
        maximum_relative_first_law_residual=.05):
    """Advance wall order with an Arrhenius-paced implicit energy step."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite wall-order interval required")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    local = evaluate_complete_multigrain_local_fields(
        state, spacing_m=spacing, wall_parameters=wall_parameters,
        mean_strain=np.asarray(mean_strain), topologies=topologies,
        systems=systems)
    stress_scale = np.max(np.abs(local.stress_Pa), axis=(-2, -1))
    fractions = {}
    rates = {}
    successes = {}
    target_changes = {}
    targets = []
    for grain_id, support, owner in zip(
            state.grain_ids, state.supports, state.owners):
        target, success, target_change_per_thickness = _owner_target(
            owner, support, wall_parameters, topologies, systems, parameters)
        # _owner_target integrates area; restore represented thickness.
        target_change = target_change_per_thickness*thickness
        temperature = float(np.sum(
            np.asarray(support)*np.asarray(owner.temperature_K),
            dtype=np.longdouble)/max(float(np.sum(support)), 1e-300))
        owner_stress = float(np.max(stress_scale[np.asarray(support)
                                                 >parameters.owner_support_threshold],
                                    initial=0.0))
        rate = float(exp_floor_rate(
            owner_stress, temperature, wall_parameters.order_barrier_eV,
            wall_parameters))
        fraction = min(-math.expm1(-rate*dt),
                       parameters.maximum_fraction_per_interval)
        if target_change >= 0.0 or np.allclose(target, owner.wall_order,
                                               rtol=0.0, atol=1e-14):
            fraction = 0.0
        fractions[int(grain_id)] = fraction
        rates[int(grain_id)] = rate
        successes[int(grain_id)] = success
        target_changes[int(grain_id)] = target_change
        targets.append(target)

    def cold_at(scale):
        owners = []
        for grain_id, owner, target in zip(
                state.grain_ids, state.owners, targets):
            fraction = scale*fractions[int(grain_id)]
            q = np.asarray(owner.wall_order)+fraction*(
                target-np.asarray(owner.wall_order))
            owners.append(replace(owner, wall_order=np.clip(q, 0.0, 1.0)))
        candidate = replace(state, owners=tuple(owners))
        candidate.validate()
        return candidate

    active = any(value > 0.0 for value in fractions.values())
    resolution = 4096*np.finfo(float).eps*max(abs(before.helmholtz_J), 1e-300)
    scale = 1.0
    backtracks = 0
    cold = state
    cold_energy = before
    if active:
        for backtracks in range(parameters.maximum_backtracks+1):
            cold = cold_at(scale)
            cold_energy = evaluate_complete_multigrain_energy(cold, **common)
            if cold_energy.helmholtz_J-before.helmholtz_J <= resolution:
                break
            scale *= .5
        else:  # pragma: no cover - loop always exhausts through its final body
            backtracks = parameters.maximum_backtracks+1
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    downhill = delta <= resolution
    heat = max(-delta, 0.0) if active and downhill else 0.0
    heat_cells = np.zeros(state.supports.shape[1:])
    activity = np.zeros_like(heat_cells)
    if heat > 0.0:
        for support, old, new in zip(state.supports, state.owners, cold.owners):
            activity += np.asarray(support)*np.abs(
                np.asarray(new.wall_order)-np.asarray(old.wall_order))
        if float(np.sum(activity)) > 0.0:
            heat_cells = heat*activity/float(np.sum(activity, dtype=np.longdouble))
    if heat > 0.0 and float(np.sum(heat_cells)) > 0.0:
        cell_volume = spacing**2*thickness
        rise = heat_cells/(wall_parameters.volumetric_heat_capacity_J_m3_K
                           *cell_volume)
        committed = replace(cold, owners=tuple(
            replace(owner, temperature_K=np.asarray(owner.temperature_K)+rise)
            for owner in cold.owners))
        committed.validate()
        hot = evaluate_complete_multigrain_energy(committed, **common)
        residual = hot.internal_J-before.internal_J
        relative = abs(residual)/max(abs(delta), heat, resolution, 1e-300)
    else:
        committed = state
        residual = relative = 0.0
    accepted = bool(active and downhill
                    and relative <= maximum_relative_first_law_residual)
    if not active:
        classification = "INACTIVE_NO_DOWNHILL_WALL_ORDER_STATE"
    elif not downhill:
        classification = "REJECTED_WALL_ORDER_UPHILL"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_WALL_ORDER_FIRST_LAW"
    else:
        classification = "ACCEPTED_IMPLICIT_ARRHENIUS_WALL_ORDER_RELAXATION"
    effective_fractions = {
        key: (scale*value if accepted else 0.0)
        for key, value in fractions.items()}
    return ImplicitWallOrderRelaxationDecision(
        accepted, classification, cold, committed if accepted else state,
        effective_fractions, rates, successes, target_changes, backtracks,
        delta, heat if accepted else 0.0, residual if accepted else 0.0,
        relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
