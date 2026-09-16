"""Sparse parent/child defect state and conservative moving-front processing.

Only promoted parent/child pairs allocate phase-resolved fields.  Initial
matrix labels continue to use the common global material class, avoiding one
full field per possible order-parameter slot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math

import numpy as np

try:
    from .arrhenius_kinetics import (
        ActivatedProcess, KB_J_K, exp_floor_enthalpy_j)
except ImportError:  # pragma: no cover - direct production-script execution
    from arrhenius_kinetics import (
        ActivatedProcess, KB_J_K, exp_floor_enthalpy_j)


@dataclass(frozen=True)
class DefectState:
    rp: np.ndarray
    rm: np.ndarray
    forest: np.ndarray
    wall: np.ndarray


@dataclass(frozen=True)
class FrontLedger:
    parent_line_processed_m: float = 0.0
    child_line_transmitted_m: float = 0.0
    boundary_line_stored_m: float = 0.0
    neutral_pair_annihilated_m: float = 0.0
    sink_line_m: float = 0.0
    line_closure_m: float = 0.0
    signed_burgers_change_m2: float = 0.0
    line_energy_released_J: float = 0.0
    heat_released_J: float = 0.0
    swept_volume_m3: float = 0.0
    requested_swept_volume_m3: float = 0.0
    capacity_limited_volume_m3: float = 0.0
    boundary_line_recovered_m: float = 0.0
    boundary_signed_released_m: float = 0.0
    boundary_neutral_recovered_m: float = 0.0


@dataclass(frozen=True)
class FrontTransfer:
    """Per-unit-volume conservative parent-to-child reaction."""

    child: DefectState
    boundary_excess_line_density_m2: np.ndarray
    boundary_excess_signed_density_m2: np.ndarray
    annihilated_line_density_m2: np.ndarray
    sink_line_density_m2: np.ndarray
    sink_signed_density_m2: np.ndarray
    line_closure_density_m2: np.ndarray
    signed_closure_density_m2: np.ndarray


@dataclass(frozen=True)
class SparseFrontState:
    parent: DefectState
    child: DefectState
    recovered_wake: DefectState
    chi: np.ndarray
    processed_max: np.ndarray
    cleanup_max: np.ndarray
    boundary_line_density_m2: np.ndarray
    boundary_signed_density_m2: np.ndarray
    ledger: FrontLedger
    parent_label: int
    child_label: int

    @property
    def current_child_fraction(self):
        """Current child support (canonical schema-v6 name; legacy ``chi``)."""
        return self.chi

    @property
    def maximum_swept_fraction(self):
        """Maximum historically swept support (legacy ``processed_max``)."""
        return self.processed_max

    @property
    def recovered_wake_fraction(self):
        """Processed support no longer occupied by the current child."""
        return self.processed_max-self.chi

    def material_support_weights(self):
        """Return parent, child, and wake weights after checking the simplex."""
        tolerance = 32.0*np.finfo(float).eps
        parent = 1.0-self.processed_max
        child = self.chi
        wake = self.processed_max-self.chi
        if (not all(np.all(np.isfinite(x)) for x in (parent, child, wake))
                or min(float(np.min(x)) for x in (parent, child, wake)) < -tolerance
                or float(np.max(np.abs(parent+child+wake-1.0))) > tolerance):
            raise ValueError("front material-support weights are inadmissible")
        return tuple(np.maximum(x, 0.0) for x in (parent, child, wake))


class FrontAdmissibilityError(RuntimeError):
    """A requested sweep lies outside the conservative transfer cone."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def canonicalize_normal_sweep(increment, *, roundoff_factor=4096.0):
    """Remove only floating-point contour noise from a normal-sweep field."""
    value = np.asarray(increment, dtype=float)
    factor = float(roundoff_factor)
    if not np.all(np.isfinite(value)) or not math.isfinite(factor) or factor < 0.0:
        raise ValueError("normal sweep and roundoff factor must be finite")
    tolerance = factor*np.finfo(float).eps
    return np.where(np.abs(value) <= tolerance, 0.0, value)


def complete_front_state_is_exactly_equal(state):
    """True only for the label-symmetric, reservoir-free complete state."""
    reservoirs = ("rp", "rm", "forest", "wall")
    return bool(
        all(np.array_equal(getattr(state.parent, name),
                           getattr(state.child, name))
            and np.array_equal(getattr(state.parent, name),
                               getattr(state.recovered_wake, name))
            for name in reservoirs)
        and not np.any(state.boundary_line_density_m2)
        and not np.any(state.boundary_signed_density_m2))


def _validate_defect(state):
    arrays = tuple(np.asarray(x, dtype=float) for x in (
        state.rp, state.rm, state.forest, state.wall))
    if arrays[0].ndim != 3 or arrays[1].shape != arrays[0].shape \
            or arrays[2].shape != arrays[0].shape \
            or arrays[3].shape != arrays[0].shape[:2]:
        raise ValueError("defect reservoirs must share a grid and slip count")
    if any(not np.all(np.isfinite(x)) or np.any(x < 0.0) for x in arrays):
        raise ValueError("defect reservoirs must be finite and nonnegative")
    return arrays


def total_line_density(state):
    rp, rm, forest, wall = _validate_defect(state)
    return np.sum(rp+rm+forest, axis=2)+wall


def signed_density(state):
    rp, rm, _, _ = _validate_defect(state)
    return np.sum(rp-rm, axis=2)


def front_feasibility_fields(state):
    """Return the pointwise no-signed-source front admissibility fields.

    The existing HAGB structure is intrinsic interface state.  Only the
    parent/child *excess* lattice-line mismatch enters this calculation.
    """
    parent_total = total_line_density(state.parent)
    child_total = total_line_density(state.child)
    signed_difference = ((np.asarray(state.parent.rp)-np.asarray(state.parent.rm))
                         -(np.asarray(state.child.rp)-np.asarray(state.child.rm)))
    signed_minimum = np.sum(np.abs(signed_difference), axis=2)
    removable = parent_total-child_total
    return {
        "parent_total_line_density_m2": parent_total,
        "child_total_line_density_m2": child_total,
        "removable_line_density_m2": removable,
        "signed_difference_by_family_m2": signed_difference,
        "signed_minimum_line_density_m2": signed_minimum,
        "feasibility_margin_m2": removable-signed_minimum,
    }


def conservative_front_transfer(parent, *, transmission_fraction=0.0,
                                boundary_storage_fraction=0.0,
                                neutral_sink_fraction=0.0,
                                signed_sink_fraction=0.0):
    """Partition incoming signed line through bounded physical channels.

    ``transmission_fraction`` may be scalar, per-family, or grid/family data.
    The boundary fields are *excess* content only; intrinsic HAGB structure is
    represented by the phase-field interfacial energy and never charged here.
    """
    rp, rm, forest, wall = _validate_defect(parent)
    transmission = np.asarray(transmission_fraction, dtype=float)
    try:
        transmission = np.broadcast_to(transmission, rp.shape)
    except ValueError as exc:
        raise ValueError("transmission fraction is not slip-grid broadcastable") from exc
    fractions = (float(boundary_storage_fraction),
                 float(neutral_sink_fraction), float(signed_sink_fraction))
    if (not np.all(np.isfinite(transmission)) or np.any(transmission < 0.0)
            or np.any(transmission > 1.0)
            or any(not math.isfinite(x) or x < 0.0 or x > 1.0
                   for x in fractions)
            or fractions[0]+fractions[1] > 1.0):
        raise ValueError("front transfer fractions must be finite and bounded")
    fb, fs, f_signed = fractions
    wall_transmission = np.mean(transmission, axis=2)
    child = DefectState(
        transmission*rp, transmission*rm, transmission*forest,
        wall_transmission*wall)
    blocked_plus = (1.0-transmission)*rp
    blocked_minus = (1.0-transmission)*rm
    blocked_signed = blocked_plus-blocked_minus
    sink_signed = f_signed*blocked_signed
    boundary_signed = blocked_signed-sink_signed
    mandatory_boundary = np.sum(np.abs(boundary_signed), axis=2)
    signed_sink_line = np.sum(np.abs(sink_signed), axis=2)
    neutral = (
        np.sum(blocked_plus+blocked_minus-np.abs(blocked_signed), axis=2)
        +np.sum((1.0-transmission)*forest, axis=2)
        +(1.0-wall_transmission)*wall)
    boundary_line = mandatory_boundary+fb*neutral
    sink_line = signed_sink_line+fs*neutral
    annihilated = (1.0-fb-fs)*neutral
    parent_total = total_line_density(parent)
    child_total = total_line_density(child)
    line_closure = parent_total-(
        child_total+boundary_line+sink_line+annihilated)
    signed_closure = (
        (rp-rm)-(child.rp-child.rm)-boundary_signed-sink_signed)
    return FrontTransfer(
        child, boundary_line, boundary_signed, annihilated, sink_line,
        sink_signed, line_closure, signed_closure)


def _cellwise_failure_record(state, newly, geometric, feasibility):
    scale = max(float(np.max(feasibility["parent_total_line_density_m2"])),
                float(np.max(feasibility["child_total_line_density_m2"])), 1.0)
    tolerance = 64.0*np.finfo(float).eps*scale
    failing = ((newly > 0.0)
               & (feasibility["feasibility_margin_m2"] < -tolerance))
    cells = []
    for i, j in np.argwhere(failing):
        families = []
        for family in range(state.parent.rp.shape[2]):
            parent_signed = float(state.parent.rp[i, j, family]
                                  -state.parent.rm[i, j, family])
            child_signed = float(state.child.rp[i, j, family]
                                 -state.child.rm[i, j, family])
            families.append({
                "family": int(family),
                "parent_rho_plus_m2": float(state.parent.rp[i, j, family]),
                "parent_rho_minus_m2": float(state.parent.rm[i, j, family]),
                "parent_forest_m2": float(state.parent.forest[i, j, family]),
                "child_rho_plus_m2": float(state.child.rp[i, j, family]),
                "child_rho_minus_m2": float(state.child.rm[i, j, family]),
                "child_forest_m2": float(state.child.forest[i, j, family]),
                "parent_signed_m2": parent_signed,
                "child_signed_m2": child_signed,
                "signed_mismatch_m2": parent_signed-child_signed,
            })
        cells.append({
            "grid_index": [int(i), int(j)],
            "processed_max": float(state.processed_max[i, j]),
            "cleanup_max": float(state.cleanup_max[i, j]),
            "requested_newly_swept_fraction": float(newly[i, j]),
            "geometric_newly_swept_fraction": (
                None if geometric is None else float(geometric[i, j])),
            "parent_wall_m2": float(state.parent.wall[i, j]),
            "child_wall_m2": float(state.child.wall[i, j]),
            "parent_total_line_density_m2": float(
                feasibility["parent_total_line_density_m2"][i, j]),
            "child_total_line_density_m2": float(
                feasibility["child_total_line_density_m2"][i, j]),
            "removable_line_density_m2": float(
                feasibility["removable_line_density_m2"][i, j]),
            "signed_minimum_line_density_m2": float(
                feasibility["signed_minimum_line_density_m2"][i, j]),
            "feasibility_margin_m2": float(
                feasibility["feasibility_margin_m2"][i, j]),
            "boundary_excess_line_density_m2": float(
                state.boundary_line_density_m2[i, j]),
            "boundary_excess_signed_density_by_family_m2": [
                float(x) for x in state.boundary_signed_density_m2[i, j]],
            "families": families,
        })
    return {
        "schema": "full-v34-moving-front-admissibility-failure/v1",
        "classification": "MOVING_FRONT_PHASE_STATE_ADMISSIBILITY_FAILED",
        "intrinsic_hagb_content_in_bulk_ledger": False,
        "signed_sink_enabled": False,
        "domain_minimum_feasibility_margin_m2": float(np.min(
            feasibility["feasibility_margin_m2"])),
        "active_sweep_minimum_feasibility_margin_m2": float(np.min(
            feasibility["feasibility_margin_m2"][newly > 0.0])),
        "density_tolerance_m2": tolerance,
        "failing_cell_count": int(np.count_nonzero(failing)),
        "failing_cells": cells,
    }


def recovered_child_state(parent, target_total_density_m2):
    """Return a low-line child while preserving signed content pointwise."""
    rp, rm, forest, wall = _validate_defect(parent)
    target = np.asarray(target_total_density_m2, dtype=float)
    if target.ndim == 0:
        target = np.full(parent.wall.shape, float(target))
    if target.shape != parent.wall.shape or not np.all(np.isfinite(target)) \
            or np.any(target < 0.0):
        raise ValueError("target density must be finite and nonnegative")
    signed = rp-rm
    cp = np.maximum(signed, 0.0)
    cm = np.maximum(-signed, 0.0)
    signed_total = np.sum(np.abs(signed), axis=2)
    neutral = np.maximum(target-signed_total, 0.0)
    cf = np.broadcast_to(neutral[:, :, None]/rp.shape[2], forest.shape).copy()
    return DefectState(cp, cm, cf, np.zeros_like(wall))


def initialize_sparse_front(parent, child_fraction, target_total_density_m2,
                            parent_label, child_label, reaction_fraction=1.0):
    """Create sparse state at an unswept phase fraction.

    Nonzero initial support must subsequently be passed to ``advance_front``;
    initialization itself neither changes material content nor prepays cleanup.
    """
    _validate_defect(parent)
    chi = np.asarray(child_fraction, dtype=float)
    if chi.shape != parent.wall.shape or not np.all(np.isfinite(chi)) \
            or np.any(chi < 0.0) or np.any(chi > 1.0):
        raise ValueError("child fraction must be finite, bounded, and grid matched")
    reacted = float(reaction_fraction)
    if not math.isfinite(reacted) or reacted < 0.0 or reacted > 1.0:
        raise ValueError("reaction fraction must be finite and bounded")
    recovered = recovered_child_state(parent, target_total_density_m2)
    # Never interpolate rp and rm independently: subtracting the resulting
    # O(1e14) populations can perturb a much smaller signed difference by ulps.
    # Carry signed mobile content explicitly and put all neutral residual line
    # into the unsigned forest reservoir. This is exact for every reaction
    # fraction, including the partially recovered state.
    signed = np.asarray(parent.rp)-np.asarray(parent.rm)
    child_rp = np.maximum(signed, 0.0)
    child_rm = np.maximum(-signed, 0.0)
    target_total = ((1.0-reacted)*total_line_density(parent)
                    +reacted*total_line_density(recovered))
    neutral = np.maximum(
        target_total-np.sum(np.abs(signed), axis=2), 0.0)
    child_forest = np.broadcast_to(
        neutral[:, :, None]/parent.rp.shape[2], parent.forest.shape).copy()
    child = DefectState(
        child_rp, child_rm, child_forest, np.zeros_like(parent.wall))
    zero = np.zeros_like(chi)
    return SparseFrontState(
        DefectState(*(np.asarray(x).copy() for x in (
            parent.rp, parent.rm, parent.forest, parent.wall))), child, child,
        zero, zero.copy(), zero.copy(), np.zeros_like(chi), np.zeros_like(parent.rp),
        FrontLedger(),
        int(parent_label), int(child_label))


def initialize_existing_boundary_front(parent, child_fraction,
                                       target_total_density_m2,
                                       parent_label, child_label):
    """Map an already existing two-grain boundary without changing defects.

    The full-field state is the only observed material state at initialization.
    Copying it to every intensive slot is an exact support-weighted
    representation because absent slots have zero extensive weight.  The
    recovered density is a later kinetic attractor, not an independently
    prescribed latent child target.  Newly swept child state is constructed
    conservatively from the current parent by :func:`advance_front`.
    """
    _validate_defect(parent)
    target = float(target_total_density_m2)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("recovered target must be finite and nonnegative")
    chi = np.asarray(child_fraction, dtype=float)
    if (chi.shape != parent.wall.shape or not np.all(np.isfinite(chi))
            or np.any(chi < 0.0) or np.any(chi > 1.0)):
        raise ValueError("child fraction must be finite, bounded, and grid matched")
    copied = lambda: DefectState(*(np.asarray(x).copy() for x in (
        parent.rp, parent.rm, parent.forest, parent.wall)))
    zero = np.zeros_like(chi)
    state = SparseFrontState(
        copied(), copied(), copied(), chi.copy(), chi.copy(), chi.copy(),
        zero, np.zeros_like(parent.rp), FrontLedger(),
        int(parent_label), int(child_label))
    mixture = reconstruct_mixture(state)
    for actual, expected in zip(
            (mixture.rp, mixture.rm, mixture.forest, mixture.wall),
            (parent.rp, parent.rm, parent.forest, parent.wall)):
        scale = max(float(np.max(np.abs(expected))), 1.0)
        if float(np.max(np.abs(actual-expected))) > 4.0*np.finfo(float).eps*scale:
            raise RuntimeError("existing-boundary representation map changed defects")
    return state


def initialize_existing_subgrain_front(parent, child_fraction,
                                       parent_label, child_label):
    """Map an already physical subgrain with zero sweep and zero heat.

    Current and historically processed support both equal the observed
    subgrain support. Every intensive defect field is copied from the common
    physical state, so this operation changes representation only. A later
    positive normal increment is the first operation allowed to process virgin
    parent material.
    """

    observed_total = total_line_density(parent)
    return initialize_existing_boundary_front(
        parent, child_fraction, float(np.min(observed_total)),
        parent_label, child_label)


def reconstruct_mixture(state, chi=None):
    """Reconstruct unswept parent, active child, and recovered wake.

    The weights are ``1-processed_max``, ``chi``, and
    ``processed_max-chi``. Therefore retreat replaces child by recovered wake,
    never by virgin parent, and cannot recreate annihilated content.
    """
    fraction = state.chi if chi is None else np.asarray(chi, dtype=float)
    if fraction.shape != state.chi.shape or np.any(fraction < 0.0) \
            or np.any(fraction > 1.0) or not np.all(np.isfinite(fraction)):
        raise ValueError("mixture fraction is invalid")
    if np.any(fraction > state.processed_max+16.0*np.finfo(float).eps):
        raise ValueError("mixture fraction exceeds processed front history")
    # A neutral representation handoff gives every material slot the same
    # intensive state.  Preserve that state bit for bit instead of evaluating
    # a partition-of-unity sum, whose multiplication and addition can inject a
    # one-ulp defect change even though the physical map is the identity.
    reservoirs = ("rp", "rm", "forest", "wall")
    if all(np.array_equal(getattr(state.parent, name),
                          getattr(state.child, name))
           and np.array_equal(getattr(state.parent, name),
                              getattr(state.recovered_wake, name))
           for name in reservoirs):
        return DefectState(*(np.array(getattr(state.parent, name), copy=True)
                             for name in reservoirs))
    virgin, _, _ = state.material_support_weights()
    wake = state.processed_max-fraction
    c3 = fraction[:, :, None]
    v3 = virgin[:, :, None]
    w3 = wake[:, :, None]
    return DefectState(
        v3*state.parent.rp+c3*state.child.rp+w3*state.recovered_wake.rp,
        v3*state.parent.rm+c3*state.child.rm+w3*state.recovered_wake.rm,
        v3*state.parent.forest+c3*state.child.forest+w3*state.recovered_wake.forest,
        virgin*state.parent.wall+fraction*state.child.wall
        +wake*state.recovered_wake.wall)


def phase_total_line_densities(state):
    """Return nonchild and child line densities for common thermodynamics."""
    parent = total_line_density(state.parent)
    child = total_line_density(state.child)
    wake = total_line_density(state.recovered_wake)
    nonchild_weight = 1.0-state.chi
    numerator = ((1.0-state.processed_max)*parent
                 +(state.processed_max-state.chi)*wake)
    nonchild = np.divide(
        numerator, nonchild_weight, out=parent.copy(),
        where=nonchild_weight > 64.0*np.finfo(float).eps)
    return nonchild, child


def apply_common_constitutive_increment(state, updated_mixture):
    """Apply a constitutive update only to material with physical support.

    Positive mixture increments are shared by supported parent, child and wake
    states. Negative increments scale their existing extensive content. A
    zero-weight latent slot is set to zero and cannot acquire history. This is
    the sparse equivalent of evolving ``Q_p=(1-m)rho_p``, ``Q_c=chi rho_c``
    and ``Q_w=(m-chi)rho_w``.
    """
    old = reconstruct_mixture(state)
    _validate_defect(updated_mixture)
    weights2 = (1.0-state.processed_max, state.chi,
                state.processed_max-state.chi)
    owner_values = [[], [], []]
    for parent, child, recovered, old_mix, new_mix in zip(
            (state.parent.rp, state.parent.rm, state.parent.forest,
             state.parent.wall),
            (state.child.rp, state.child.rm, state.child.forest,
             state.child.wall),
            (state.recovered_wake.rp, state.recovered_wake.rm,
             state.recovered_wake.forest, state.recovered_wake.wall),
            (old.rp, old.rm, old.forest, old.wall),
            (updated_mixture.rp, updated_mixture.rm,
             updated_mixture.forest, updated_mixture.wall)):
        old_mix = np.asarray(old_mix)
        target = np.asarray(new_mix)
        ratio = np.divide(target, old_mix, out=np.zeros_like(target),
                          where=old_mix > 0.0)
        addition = np.maximum(target-old_mix, 0.0)
        for owner_index, (value, weight2) in enumerate(zip(
                (parent, child, recovered), weights2)):
            value = np.asarray(value)
            weight = weight2[:, :, None] if value.ndim == 3 else weight2
            supported = weight > 64.0*np.finfo(float).eps
            result = np.where(target < old_mix, value*ratio, value+addition)
            owner_values[owner_index].append(
                np.where(supported, np.maximum(result, 0.0), 0.0))
    candidate = replace(
        state,
        parent=DefectState(*owner_values[0]),
        child=DefectState(*owner_values[1]),
        recovered_wake=DefectState(*owner_values[2]))
    check = reconstruct_mixture(candidate)
    error = max(float(np.max(np.abs(a-b))) for a, b in zip(
        (check.rp, check.rm, check.forest, check.wall),
        (updated_mixture.rp, updated_mixture.rm,
         updated_mixture.forest, updated_mixture.wall)))
    scale = max(*(float(np.max(np.abs(x))) for x in (
        updated_mixture.rp, updated_mixture.rm,
        updated_mixture.forest, updated_mixture.wall)), 1.0)
    if error > 256.0*np.finfo(float).eps*scale:
        raise RuntimeError("common constitutive mixture reconstruction failed")
    return candidate


def activated_front_fraction(process, stress_pa, temperature_K, dt_s, *,
                             h0_J, critical_stress_pa, exp_a,
                             exp_n, exp_floor):
    """Finite-step reacted fraction for one EXP-floor front channel.

    Activation entropy enters exactly once through
    ``Delta G = Delta H - T Delta S``. Negative barriers use the process's
    declared drag branch instead of an exponential extrapolation.
    """
    if not isinstance(process, ActivatedProcess):
        raise TypeError("process must be an ActivatedProcess")
    temperature = np.asarray(temperature_K, dtype=float)
    if (not math.isfinite(float(dt_s)) or dt_s < 0.0
            or not np.all(np.isfinite(temperature))
            or np.any(temperature <= 0.0)):
        raise ValueError("front activation requires dt>=0 and T>0")
    enthalpy = np.asarray(exp_floor_enthalpy_j(
        stress_pa, h0_J, critical_stress_pa, exp_a, exp_n, exp_floor),
        dtype=float)
    enthalpy, temperature = np.broadcast_arrays(enthalpy, temperature)
    barrier = enthalpy-KB_J_K*temperature*process.entropy_over_kB
    thermal_rate = (process.attempt_frequency_s
                    * np.exp(np.clip(-barrier/(KB_J_K*temperature), -745.0, 700.0)))
    if process.negative_barrier_mode == "reject" and np.any(barrier < 0.0):
        raise ValueError(f"{process.name}: negative free barrier outside validity envelope")
    drag = min(process.attempt_frequency_s, process.drag_rate_s)
    rate = np.where(barrier < 0.0, drag, thermal_rate)
    return -np.expm1(-rate*float(dt_s))


def _advance_front_v13_infeasible_reference(
        state, new_child_fraction, *, cell_area_m2,
        represented_thickness_m, line_energy_J_m,
        boundary_storage_fraction=0.0, sink_fraction=0.0,
        newly_swept_fraction=None):
    """Advance or retreat a resolved front with no repeated cleanup.

    Physical processing occurs only for ``new_child_fraction`` beyond the
    maximum fraction previously swept at each point. Retreat changes the
    mixture but cannot reverse annihilation; re-advance through an already
    processed interval performs no second reaction.
    """
    chi1 = np.asarray(new_child_fraction, dtype=float)
    if chi1.shape != state.chi.shape or not np.all(np.isfinite(chi1)) \
            or np.any(chi1 < 0.0) or np.any(chi1 > 1.0):
        raise ValueError("new child fraction must be finite and bounded")
    for value, name in ((cell_area_m2, "cell area"),
                        (represented_thickness_m, "thickness"),
                        (line_energy_J_m, "line energy")):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    fb = float(boundary_storage_fraction); fs = float(sink_fraction)
    if fb < 0.0 or fs < 0.0 or fb+fs > 1.0:
        raise ValueError("front partition fractions must be nonnegative and sum to <=1")
    newly = np.maximum(chi1-state.cleanup_max, 0.0)
    if newly_swept_fraction is not None:
        geometric = np.asarray(newly_swept_fraction, dtype=float)
        if (geometric.shape != chi1.shape or not np.all(np.isfinite(geometric))
                or np.any(geometric < 0.0) or np.any(geometric > 1.0)):
            raise ValueError("newly swept fraction must be finite and bounded")
        # The phase-fraction maximum prevents repeated cleanup; the geometric
        # cap prevents profile sharpening/broadening from masquerading as
        # normal contour sweep.
        newly = np.minimum(newly, geometric)
    volume = float(cell_area_m2)*float(represented_thickness_m)
    feasibility = front_feasibility_fields(state)
    parent_line = feasibility["parent_total_line_density_m2"]
    child_line = feasibility["child_total_line_density_m2"]
    # Continued common deformation can re-harden a latent child state above
    # the material it is about to sweep.  A boundary may transmit the parent
    # state but cannot create the excess line.  Replace only those newly swept
    # latent cells by exact transmission before forming the partition ledger.
    transmit = (newly > 0.0) & (child_line > parent_line)
    if np.any(transmit):
        def transmitted(child_value, parent_value):
            mask = transmit[:, :, None] if child_value.ndim == 3 else transmit
            return np.where(mask, parent_value, child_value)
        child = DefectState(*(transmitted(c, p) for c, p in zip(
            (state.child.rp, state.child.rm, state.child.forest, state.child.wall),
            (state.parent.rp, state.parent.rm, state.parent.forest, state.parent.wall))))
        wake = DefectState(*(transmitted(c, p) for c, p in zip(
            (state.recovered_wake.rp, state.recovered_wake.rm,
             state.recovered_wake.forest, state.recovered_wake.wall),
            (state.parent.rp, state.parent.rm, state.parent.forest, state.parent.wall))))
        state = replace(state, child=child, recovered_wake=wake)
        feasibility = front_feasibility_fields(state)
        child_line = feasibility["child_total_line_density_m2"]
    line_difference = feasibility["removable_line_density_m2"]
    density_tolerance = 64.0*np.finfo(float).eps*max(
        float(np.max(parent_line)), float(np.max(child_line)), 1.0)
    if np.any(newly > 0.0) and np.min(line_difference[newly > 0.0]) < -density_tolerance:
        raise RuntimeError("moving front child contains more line than parent")
    removable = np.maximum(line_difference, 0.0)
    removed_m = float(np.sum(newly*removable, dtype=np.longdouble)*volume)
    parent_m = float(np.sum(newly*parent_line, dtype=np.longdouble)*volume)
    # Define transmitted content by the local partition identity.  Independently
    # summing two O(parent) fields and subtracting their much smaller difference
    # loses the front increment to cancellation on physical grids.
    child_m = parent_m-removed_m
    signed_residual_density = feasibility["signed_minimum_line_density_m2"]
    signed_required_density = newly*signed_residual_density
    signed_required_m = float(np.sum(
        signed_required_density, dtype=np.longdouble)*volume)
    if signed_required_m > removed_m+1024.0*math.ulp(max(removed_m, 1e-300)):
        record = _cellwise_failure_record(
            state, newly,
            None if newly_swept_fraction is None else geometric,
            feasibility)
        record.update({
            "integrated_removable_line_m": removed_m,
            "integrated_signed_minimum_line_m": signed_required_m,
        })
        raise FrontAdmissibilityError(
            "signed boundary residual exceeds removable line content", record)
    neutral_removed_m = max(removed_m-signed_required_m, 0.0)
    boundary_m = signed_required_m+fb*neutral_removed_m
    sink_m = fs*neutral_removed_m
    annihilated_m = removed_m-boundary_m-sink_m
    closure = parent_m-(child_m+boundary_m+annihilated_m+sink_m)
    boundary_density = state.boundary_line_density_m2.copy()
    boundary_signed = state.boundary_signed_density_m2.copy()
    if boundary_m > 0.0:
        neutral_removable = np.maximum(
            removable-signed_residual_density, 0.0)
        boundary_density += newly*(
            signed_residual_density+fb*neutral_removable)
        signed_increment = newly[:, :, None]*(
            (state.parent.rp-state.parent.rm)
            -(state.child.rp-state.child.rm))
        boundary_signed += signed_increment
    else:
        signed_increment = np.zeros_like(boundary_signed)
    mixture0 = reconstruct_mixture(state, state.chi)
    candidate = replace(
        state, chi=chi1.copy(),
        # Advance the processed marker only by content that the geometric
        # contour audit accepts as newly swept.  A diffuse-profile change that
        # is rejected here must remain eligible for later, genuine contour
        # passage.
        processed_max=np.maximum(state.processed_max, chi1),
        cleanup_max=np.minimum(1.0, state.cleanup_max+newly),
        boundary_line_density_m2=boundary_density,
        boundary_signed_density_m2=boundary_signed)
    mixture1 = reconstruct_mixture(candidate, chi1)
    # Balance the represented phase states, not a subtraction of two blended
    # O(1e17) mixture fields. The latter can differ by an ulp even when every
    # signed population is identical. State-space equality is the exact
    # Burgers-content invariant used by the ledger.
    # ``signed_increment`` is assigned directly to the boundary reservoir, so
    # the incremental Burgers balance is exact by construction. Subtracting it
    # back out of a much larger cumulative field only measures roundoff.
    signed_change = 0.0
    scale = max(abs(parent_m), abs(child_m), abs(removed_m), 1e-300)
    tolerance = 8192.0*math.ulp(scale)
    if abs(closure) > tolerance or signed_change > 0.0:
        raise RuntimeError(
            f"moving-front balance failed: line={closure:.17g} m, "
            f"signed={signed_change:.17g} m^-2")
    energy = annihilated_m*float(line_energy_J_m)
    old = state.ledger
    ledger = FrontLedger(
        old.parent_line_processed_m+parent_m,
        old.child_line_transmitted_m+child_m,
        old.boundary_line_stored_m+boundary_m,
        old.neutral_pair_annihilated_m+annihilated_m,
        old.sink_line_m+sink_m,
        old.line_closure_m+closure,
        max(old.signed_burgers_change_m2, signed_change),
        old.line_energy_released_J+energy,
        old.heat_released_J+energy,
        old.swept_volume_m3+float(np.sum(newly))*volume)
    return replace(candidate, ledger=ledger), mixture1


def _blend_intensive(old_value, old_weight, incoming_value, increment):
    weight = old_weight[:, :, None] if old_value.ndim == 3 else old_weight
    amount = increment[:, :, None] if old_value.ndim == 3 else increment
    new_weight = weight+amount
    return np.divide(
        weight*old_value+amount*incoming_value, new_weight,
        out=np.asarray(old_value).copy(), where=new_weight > 0.0)


def _blend_state(old, old_weight, incoming, increment):
    return DefectState(*(_blend_intensive(a, old_weight, b, increment)
                         for a, b in zip(
                             (old.rp, old.rm, old.forest, old.wall),
                             (incoming.rp, incoming.rm,
                              incoming.forest, incoming.wall))))


def advance_front(state, new_child_fraction, *, cell_area_m2,
                  represented_thickness_m, line_energy_J_m,
                  boundary_storage_fraction=0.0, sink_fraction=0.0,
                  newly_swept_fraction=None, transmission_fraction=1.0,
                  signed_sink_fraction=0.0,
                  boundary_capacity_density_m2=None):
    """Advance the physical front with a feasible per-sign transfer map.

    When ``newly_swept_fraction`` is supplied it is the signed normal-contour
    conversion. The phase-field target is then diagnostic only: diffuse width
    relaxation cannot process material. Positive virgin sweep constructs child
    and future-wake state from the current parent through
    :func:`conservative_front_transfer`. Retreat and re-advance only exchange
    already processed child/wake material and never repeat cleanup.
    """
    requested_phase = np.asarray(new_child_fraction, dtype=float)
    if (requested_phase.shape != state.chi.shape
            or not np.all(np.isfinite(requested_phase))
            or np.any(requested_phase < 0.0)
            or np.any(requested_phase > 1.0)):
        raise ValueError("new child fraction must be finite and bounded")
    for value, name in ((cell_area_m2, "cell area"),
                        (represented_thickness_m, "thickness"),
                        (line_energy_J_m, "line energy")):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if newly_swept_fraction is None:
        normal_change = requested_phase-state.chi
    else:
        normal_change = np.asarray(newly_swept_fraction, dtype=float)
        if (normal_change.shape != state.chi.shape
                or not np.all(np.isfinite(normal_change))
                or np.any(normal_change < -1.0)
                or np.any(normal_change > 1.0)):
            raise ValueError("normal sweep fraction must be finite and in [-1,1]")

    # Retreat transfers active child into recovered wake without line removal.
    retreat = np.minimum(np.maximum(-normal_change, 0.0), state.chi)
    wake_weight0 = state.processed_max-state.chi
    wake = _blend_state(
        state.recovered_wake, wake_weight0, state.child, retreat)
    child_weight = state.chi-retreat
    child = state.child

    # Re-advance through processed wake without repeating the front reaction.
    positive = np.maximum(normal_change, 0.0)
    revisit_available = state.processed_max-child_weight
    revisit = np.minimum(positive, revisit_available)
    child = _blend_state(child, child_weight, wake, revisit)
    child_weight = child_weight+revisit
    virgin_requested = np.minimum(
        positive-revisit, np.maximum(1.0-state.processed_max, 0.0))

    transfer = conservative_front_transfer(
        state.parent, transmission_fraction=transmission_fraction,
        boundary_storage_fraction=boundary_storage_fraction,
        neutral_sink_fraction=sink_fraction,
        signed_sink_fraction=signed_sink_fraction)
    closure_scale = max(float(np.max(total_line_density(state.parent))), 1.0)
    closure_tolerance = 512.0*np.finfo(float).eps*closure_scale
    if (float(np.max(np.abs(transfer.line_closure_density_m2)))
            > closure_tolerance
            or float(np.max(np.abs(transfer.signed_closure_density_m2)))
            > closure_tolerance):
        raise RuntimeError("constrained front transfer failed algebraic closure")

    if boundary_capacity_density_m2 is None:
        alpha = np.ones_like(state.chi)
    else:
        capacity = np.asarray(boundary_capacity_density_m2, dtype=float)
        if capacity.ndim == 0:
            capacity = np.full(state.chi.shape, float(capacity))
        if (capacity.shape != state.chi.shape or not np.all(np.isfinite(capacity))
                or np.any(capacity < 0.0)):
            raise ValueError("boundary capacity must be finite and nonnegative")
        available = np.maximum(capacity-state.boundary_line_density_m2, 0.0)
        demand = virgin_requested*transfer.boundary_excess_line_density_m2
        alpha = np.divide(available, demand, out=np.ones_like(demand),
                          where=demand > 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)
    accepted = virgin_requested*alpha
    child = _blend_state(child, child_weight, transfer.child, accepted)
    child_weight = child_weight+accepted
    processed = state.processed_max+accepted

    boundary_density = (state.boundary_line_density_m2
                        +accepted*transfer.boundary_excess_line_density_m2)
    boundary_signed = (state.boundary_signed_density_m2
                       +accepted[:, :, None]
                       *transfer.boundary_excess_signed_density_m2)
    candidate = replace(
        state, child=child, recovered_wake=wake, chi=child_weight,
        processed_max=processed, cleanup_max=processed.copy(),
        boundary_line_density_m2=boundary_density,
        boundary_signed_density_m2=boundary_signed)
    mixture = reconstruct_mixture(candidate)

    volume = float(cell_area_m2)*float(represented_thickness_m)
    parent_m = float(np.sum(
        accepted*total_line_density(state.parent), dtype=np.longdouble)*volume)
    child_m = float(np.sum(
        accepted*total_line_density(transfer.child), dtype=np.longdouble)*volume)
    boundary_m = float(np.sum(
        accepted*transfer.boundary_excess_line_density_m2,
        dtype=np.longdouble)*volume)
    annihilated_m = float(np.sum(
        accepted*transfer.annihilated_line_density_m2,
        dtype=np.longdouble)*volume)
    sink_m = float(np.sum(
        accepted*transfer.sink_line_density_m2,
        dtype=np.longdouble)*volume)
    closure = parent_m-(child_m+boundary_m+annihilated_m+sink_m)
    signed_change = float(np.max(np.abs(
        accepted[:, :, None]*transfer.signed_closure_density_m2)))
    scale = max(abs(parent_m), abs(child_m), abs(boundary_m), 1e-300)
    if abs(closure) > 8192.0*math.ulp(scale) or signed_change > closure_tolerance:
        raise RuntimeError(
            f"moving-front balance failed: line={closure:.17g} m, "
            f"signed={signed_change:.17g} m^-2")
    requested_volume = float(np.sum(virgin_requested))*volume
    swept_volume = float(np.sum(accepted))*volume
    energy = annihilated_m*float(line_energy_J_m)
    old = state.ledger
    ledger = FrontLedger(
        old.parent_line_processed_m+parent_m,
        old.child_line_transmitted_m+child_m,
        old.boundary_line_stored_m+boundary_m,
        old.neutral_pair_annihilated_m+annihilated_m,
        old.sink_line_m+sink_m,
        old.line_closure_m+closure,
        max(old.signed_burgers_change_m2, signed_change),
        old.line_energy_released_J+energy,
        old.heat_released_J+energy,
        old.swept_volume_m3+swept_volume,
        old.requested_swept_volume_m3+requested_volume,
        old.capacity_limited_volume_m3+(requested_volume-swept_volume),
        old.boundary_line_recovered_m,
        old.boundary_signed_released_m,
        old.boundary_neutral_recovered_m)
    return replace(candidate, ledger=ledger), mixture


def recover_boundary_reservoir(state, recovery_fraction, *, cell_area_m2,
                               represented_thickness_m, line_energy_J_m):
    """Recover boundary excess while conserving signed Burgers content.

    Signed excess is released into the processed material populations.  The
    active child and recovered wake receive the same intensive increment, so
    their support-weighted extensive increment is exactly the released signed
    reservoir even after complete front retreat.  Neutral excess annihilates
    and its exact line energy is deposited as heat.  Intrinsic HAGB content is
    absent from this excess-only reservoir.
    """
    fraction = float(recovery_fraction)
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("boundary recovery fraction must lie in [0,1]")
    for value, name in ((cell_area_m2, "cell area"),
                        (represented_thickness_m, "thickness"),
                        (line_energy_J_m, "line energy")):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if fraction == 0.0 or not np.any(state.boundary_line_density_m2):
        return state, reconstruct_mixture(state)
    released_signed = fraction*state.boundary_signed_density_m2
    released_signed_line = np.sum(np.abs(released_signed), axis=2)
    released_total = fraction*state.boundary_line_density_m2
    released_neutral = np.maximum(released_total-released_signed_line, 0.0)
    processed_support = state.processed_max
    unsupported = ((released_signed_line > 0.0)
                   & (processed_support <= 64.0*np.finfo(float).eps))
    if np.any(unsupported):
        raise RuntimeError(
            "signed boundary content has no processed-material support "
            "for release")
    support = processed_support[:, :, None]
    plus = np.maximum(released_signed, 0.0)
    minus = np.maximum(-released_signed, 0.0)
    plus_increment = np.divide(
        plus, support, out=np.zeros_like(plus), where=support > 0.0)
    minus_increment = np.divide(
        minus, support, out=np.zeros_like(minus), where=support > 0.0)
    child_support = state.chi[:, :, None] > 0.0
    wake_support = (state.processed_max-state.chi)[:, :, None] > 0.0
    child = DefectState(
        state.child.rp+np.where(child_support, plus_increment, 0.0),
        state.child.rm+np.where(child_support, minus_increment, 0.0),
        state.child.forest.copy(), state.child.wall.copy())
    wake = DefectState(
        state.recovered_wake.rp+np.where(wake_support, plus_increment, 0.0),
        state.recovered_wake.rm+np.where(wake_support, minus_increment, 0.0),
        state.recovered_wake.forest.copy(), state.recovered_wake.wall.copy())
    volume = float(cell_area_m2)*float(represented_thickness_m)
    total_m = float(np.sum(released_total, dtype=np.longdouble)*volume)
    signed_m = float(np.sum(released_signed_line, dtype=np.longdouble)*volume)
    neutral_m = float(np.sum(released_neutral, dtype=np.longdouble)*volume)
    closure = total_m-signed_m-neutral_m
    if abs(closure) > 8192.0*math.ulp(max(total_m, 1e-300)):
        raise RuntimeError("boundary recovery line balance failed")
    energy = neutral_m*float(line_energy_J_m)
    old = state.ledger
    ledger = replace(
        old,
        neutral_pair_annihilated_m=old.neutral_pair_annihilated_m+neutral_m,
        line_closure_m=old.line_closure_m+closure,
        line_energy_released_J=old.line_energy_released_J+energy,
        heat_released_J=old.heat_released_J+energy,
        boundary_line_recovered_m=old.boundary_line_recovered_m+total_m,
        boundary_signed_released_m=old.boundary_signed_released_m+signed_m,
        boundary_neutral_recovered_m=old.boundary_neutral_recovered_m+neutral_m)
    candidate = replace(
        state, child=child, recovered_wake=wake,
        boundary_line_density_m2=(1.0-fraction)
        *state.boundary_line_density_m2,
        boundary_signed_density_m2=(1.0-fraction)
        *state.boundary_signed_density_m2,
        ledger=ledger)
    return candidate, reconstruct_mixture(candidate)


def state_metadata_json(state):
    return json.dumps({
        "schema": "full-v34-sparse-front/v6",
        "parent_label": state.parent_label,
        "child_label": state.child_label,
        "boundary_content_semantics": "excess_only_intrinsic_HAGB_excluded",
        "state_ownership": "support_weighted",
        "fraction_semantics": {
            "current_child_fraction": "current child support; legacy alias chi",
            "maximum_swept_fraction": (
                "maximum historically swept support; legacy alias processed_max"),
            "recovered_wake_fraction": (
                "maximum_swept_fraction-current_child_fraction; derived, not stored"),
            "weights": [
                "1-maximum_swept_fraction", "current_child_fraction",
                "maximum_swept_fraction-current_child_fraction"],
        },
        "schema_migration": (
            "v3-v5 chi -> current_child_fraction; processed_max -> "
            "maximum_swept_fraction; cleanup_max retained as reaction-history alias"),
        "ledger": state.ledger.__dict__,
    }, sort_keys=True, separators=(",", ":"))


def state_arrays(state):
    return {
        "parent_rp": state.parent.rp, "parent_rm": state.parent.rm,
        "parent_forest": state.parent.forest, "parent_wall": state.parent.wall,
        "child_rp": state.child.rp, "child_rm": state.child.rm,
        "child_forest": state.child.forest, "child_wall": state.child.wall,
        "wake_rp": state.recovered_wake.rp,
        "wake_rm": state.recovered_wake.rm,
        "wake_forest": state.recovered_wake.forest,
        "wake_wall": state.recovered_wake.wall,
        # Canonical names are written together with byte-identical legacy
        # aliases so older full-v34 drivers can restart v6 checkpoints.
        "current_child_fraction": state.chi,
        "maximum_swept_fraction": state.processed_max,
        "chi": state.chi, "processed_max": state.processed_max,
        "cleanup_max": state.cleanup_max,
        "boundary_line_density_m2": state.boundary_line_density_m2,
        "boundary_signed_density_m2": state.boundary_signed_density_m2,
    }


def state_from_checkpoint(metadata_json, arrays):
    raw = json.loads(str(metadata_json))
    if raw.get("schema") not in ("full-v34-sparse-front/v3",
                                  "full-v34-sparse-front/v4",
                                  "full-v34-sparse-front/v5",
                                  "full-v34-sparse-front/v6"):
        raise ValueError("unsupported sparse-front schema")
    parent = DefectState(*(np.asarray(arrays[k]).copy() for k in (
        "parent_rp", "parent_rm", "parent_forest", "parent_wall")))
    child = DefectState(*(np.asarray(arrays[k]).copy() for k in (
        "child_rp", "child_rm", "child_forest", "child_wall")))
    wake = DefectState(*(np.asarray(arrays[k]).copy() for k in (
        "wake_rp", "wake_rm", "wake_forest", "wake_wall")))
    current_key = ("current_child_fraction"
                   if "current_child_fraction" in arrays else "chi")
    swept_key = ("maximum_swept_fraction"
                 if "maximum_swept_fraction" in arrays else "processed_max")
    current = np.asarray(arrays[current_key]).copy()
    processed = np.asarray(arrays[swept_key]).copy()
    if "chi" in arrays and not np.array_equal(current, np.asarray(arrays["chi"])):
        raise ValueError("conflicting current-child fraction aliases")
    if ("processed_max" in arrays
            and not np.array_equal(processed, np.asarray(arrays["processed_max"]))):
        raise ValueError("conflicting maximum-swept fraction aliases")
    cleanup = (np.asarray(arrays["cleanup_max"]).copy()
               if "cleanup_max" in arrays else processed.copy())
    state = SparseFrontState(
        parent, child, wake, current,
        processed, cleanup,
        np.asarray(arrays["boundary_line_density_m2"]).copy(),
        np.asarray(arrays["boundary_signed_density_m2"]).copy(),
        FrontLedger(**raw["ledger"]), int(raw["parent_label"]),
        int(raw["child_label"]))
    _validate_defect(parent); _validate_defect(child); _validate_defect(wake)
    state.material_support_weights()
    return state
