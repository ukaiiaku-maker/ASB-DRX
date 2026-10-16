"""Energy-qualified climb recovery of redundant ordered-wall line pairs.

Opposite signs in one Burgers family do not contribute to its signed wall
content.  At elevated temperature they may climb and annihilate while the
surviving signed excess remains available for a subsequent conservative
Frank--Bilby reorganization.  This transaction removes equal support-weighted
positive and negative wall density only; it cannot rotate a lattice, allocate
a phase, or change any Burgers resultant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .common_tensorial_wall import wall_free_energy_derivatives
from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)


@dataclass(frozen=True)
class WallNeutralPairRecoveryParameters:
    """One shared Arrhenius EXP-floor law for dynamic and static recovery."""

    activation: StructuralBoundaryParameters = StructuralBoundaryParameters(
        activation_enthalpy_eV=1.20, barrier_floor_eV=.05,
        attempt_frequency_Hz=1.0e7, activation_segment_m=2.48e-10)
    minimum_wall_order: float = .20
    owner_support_threshold: float = .75
    maximum_backtracks: int = 20

    def __post_init__(self):
        if not 0.0 <= float(self.minimum_wall_order) <= 1.0:
            raise ValueError("minimum wall order must lie in [0,1]")
        if not 0.0 <= float(self.owner_support_threshold) < 1.0:
            raise ValueError("owner support threshold must lie in [0,1)")
        if int(self.maximum_backtracks) < 0:
            raise ValueError("wall recovery backtracks cannot be negative")


@dataclass(frozen=True)
class WallNeutralPairRecoveryDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    annihilated_pair_cell_sum_m2: float
    signed_content_residual_m2: float
    line_balance_residual_m2: float
    maximum_rate_s: float
    backtracks: int
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _inventories(state):
    signed = []
    line = np.longdouble(0.0)
    for support, owner in zip(state.supports, state.owners):
        weight = np.asarray(support, dtype=float)[..., None]
        signed.append(np.sum(
            weight*(np.asarray(owner.wall_plus_m2)
                    -np.asarray(owner.wall_minus_m2)),
            axis=(0, 1), dtype=np.longdouble))
        line += np.sum(
            weight*(np.asarray(owner.wall_plus_m2)
                    +np.asarray(owner.wall_minus_m2)), dtype=np.longdouble)
    return np.asarray(signed, dtype=np.longdouble), line


def evaluate_wall_neutral_pair_recovery(
        state, *, interval_s, mean_strain, spacing_m,
        represented_thickness_m, systems, wall_parameters,
        energy_kwargs=None,
        parameters=WallNeutralPairRecoveryParameters(),
        maximum_relative_first_law_residual=.05):
    """Remove redundant wall pairs and deposit their Helmholtz release as heat."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite wall-recovery interval required")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    initial_signed, initial_line = _inventories(state)
    proposed_owners = []
    proposed_pair = 0.0
    maximum_rate = 0.0
    activity = np.zeros(state.supports.shape[1:], dtype=float)
    topologies = options.get("topologies", ())
    for support, owner in zip(state.supports, state.owners):
        chemical = wall_free_energy_derivatives(
            owner, wall_parameters, topologies, systems)
        affinity = np.maximum(
            np.asarray(chemical["wall_plus_mu_J_m"])
            +np.asarray(chemical["wall_minus_mu_J_m"]), 0.0)
        rate = exp_floor_event_rate_Hz(
            affinity, np.asarray(owner.temperature_K)[..., None],
            parameters.activation)
        active = ((np.asarray(support) > parameters.owner_support_threshold)
                  &(np.asarray(owner.wall_order)
                    >= parameters.minimum_wall_order))
        rate = np.where(active[..., None], rate, 0.0)
        fraction = -np.expm1(-rate*dt)
        neutral = np.minimum(
            np.asarray(owner.wall_plus_m2),
            np.asarray(owner.wall_minus_m2))*fraction
        candidate = replace(
            owner,
            wall_plus_m2=np.asarray(owner.wall_plus_m2)-neutral,
            wall_minus_m2=np.asarray(owner.wall_minus_m2)-neutral)
        candidate.validate(systems, topologies)
        proposed_owners.append(candidate)
        weighted = np.asarray(support)[..., None]*neutral
        proposed_pair += float(np.sum(weighted, dtype=np.longdouble))
        activity += np.sum(weighted, axis=2)
        maximum_rate = max(maximum_rate, float(np.max(rate, initial=0.0)))
    proposed = replace(state, owners=tuple(proposed_owners))
    proposed.validate()

    resolution = 4096*np.finfo(float).eps*max(
        abs(before.helmholtz_J), 1e-300)
    scale = 1.0
    backtracks = 0
    cold = state
    cold_energy = before
    if proposed_pair > 0.0:
        for backtracks in range(parameters.maximum_backtracks+1):
            owners = tuple(replace(
                old,
                wall_plus_m2=np.asarray(old.wall_plus_m2)+scale*(
                    np.asarray(trial.wall_plus_m2)-np.asarray(old.wall_plus_m2)),
                wall_minus_m2=np.asarray(old.wall_minus_m2)+scale*(
                    np.asarray(trial.wall_minus_m2)-np.asarray(old.wall_minus_m2)))
                for old, trial in zip(state.owners, proposed.owners))
            cold = replace(state, owners=owners)
            cold.validate()
            cold_energy = evaluate_complete_multigrain_energy(cold, **common)
            if cold_energy.helmholtz_J-before.helmholtz_J <= resolution:
                break
            scale *= .5
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    downhill = delta <= resolution
    annihilated_pair = proposed_pair*scale if proposed_pair > 0.0 else 0.0
    heat = max(-delta, 0.0) if annihilated_pair > 0.0 and downhill else 0.0
    cell_volume = spacing**2*thickness
    heat_cells = np.zeros_like(activity)
    if heat > 0.0 and float(np.sum(activity)) > 0.0:
        heat_cells = heat*activity/float(np.sum(activity, dtype=np.longdouble))
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

    final_signed, final_line = _inventories(cold)
    signed_residual = float(np.max(np.abs(final_signed-initial_signed)))
    line_residual = float(final_line-initial_line+2.0*annihilated_pair)
    inventory_scale = max(float(initial_line), 1.0)
    balanced = (signed_residual <= 1e-12*inventory_scale
                and abs(line_residual) <= 1e-12*inventory_scale)
    changed = annihilated_pair > 0.0 and any(
        not np.array_equal(old.wall_plus_m2, new.wall_plus_m2)
        or not np.array_equal(old.wall_minus_m2, new.wall_minus_m2)
        for old, new in zip(state.owners, cold.owners))
    accepted = bool(changed and downhill and balanced
                    and relative <= maximum_relative_first_law_residual)
    if accepted:
        classification = "ACCEPTED_WALL_NEUTRAL_PAIR_RECOVERY"
    elif proposed_pair <= 0.0:
        classification = "INACTIVE_NO_ACCESSIBLE_NEUTRAL_WALL_PAIR"
    elif not changed:
        classification = "INACTIVE_BELOW_NUMERICAL_WALL_RECOVERY_RESOLUTION"
    elif not downhill:
        classification = "INACTIVE_NO_FINITE_DOWNHILL_WALL_RECOVERY_STEP"
    elif not balanced:
        classification = "REJECTED_WALL_RECOVERY_BALANCE"
    else:
        classification = "REJECTED_WALL_RECOVERY_FIRST_LAW"
    return WallNeutralPairRecoveryDecision(
        accepted, classification, proposed, committed if accepted else state,
        annihilated_pair if accepted else 0.0, signed_residual, line_residual,
        maximum_rate, backtracks, delta, heat if accepted else 0.0,
        residual if accepted else 0.0, relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
