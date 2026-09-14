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


class FrontAdmissibilityError(RuntimeError):
    """A requested sweep lies outside the conservative transfer cone."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


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

    The existing child side owns the current full-field state exactly.  Only
    unswept parent cells carry a latent recovered child state for subsequent
    advance.  This makes initialization an exact representation map while
    allowing newly swept material to use the normal v9 front ledger.
    """
    local_target = np.minimum(
        float(target_total_density_m2), total_line_density(parent))
    base = initialize_sparse_front(
        parent, child_fraction, local_target,
        parent_label, child_label, reaction_fraction=1.0)
    chi = np.asarray(child_fraction, dtype=float)
    def mapped(current, recovered):
        q = chi[:, :, None] if current.ndim == 3 else chi
        # On the established child side the current field is already the child
        # state.  On the parent side retain a latent recovered state even when
        # diffuse eta tails make chi numerically nonzero.
        latent = np.asarray(recovered)
        # Exact nonnegative deconvolution requires q*C <= current.  Diffuse
        # tails can overlap a reservoir that is identically zero, so reduce
        # only that latent component rather than introducing negative parent
        # content.
        cap = np.divide(np.asarray(current), q,
                        out=np.full_like(np.asarray(current), np.inf),
                        where=q > 0.0)
        latent = np.minimum(latent, cap)
        return np.where(q >= 0.5, current, latent)
    child_values = tuple(mapped(np.asarray(current), np.asarray(recovered))
                         for current, recovered in zip(
                             (parent.rp, parent.rm, parent.forest, parent.wall),
                             (base.child.rp, base.child.rm,
                              base.child.forest, base.child.wall)))
    child = DefectState(*child_values)
    parent_values = []
    for current, child_value in zip(
            (parent.rp, parent.rm, parent.forest, parent.wall), child_values):
        q = chi[:, :, None] if current.ndim == 3 else chi
        parent_value = np.divide(
            np.asarray(current)-q*child_value, 1.0-q,
            out=np.asarray(current).copy(), where=(1.0-q) > 64*np.finfo(float).eps)
        if np.min(parent_value) < -64*np.finfo(float).eps*max(float(np.max(current)), 1.0):
            raise RuntimeError("existing-boundary map requires negative parent content")
        parent_values.append(np.maximum(parent_value, 0.0))
    mapped_parent = DefectState(*parent_values)
    state = replace(
        base, parent=mapped_parent, child=child, recovered_wake=child,
        chi=chi.copy(), processed_max=chi.copy(), cleanup_max=chi.copy())
    mixture = reconstruct_mixture(state)
    for actual, expected in zip(
            (mixture.rp, mixture.rm, mixture.forest, mixture.wall),
            (parent.rp, parent.rm, parent.forest, parent.wall)):
        scale = max(float(np.max(np.abs(expected))), 1.0)
        if float(np.max(np.abs(actual-expected))) > 4.0*np.finfo(float).eps*scale:
            raise RuntimeError("existing-boundary representation map changed defects")
    return state


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
    virgin = 1.0-state.processed_max
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
    """Apply the full-field constitutive increment to both material states.

    If ``delta = updated_mixture - reconstruct_mixture(state)``, adding delta
    to both states reconstructs the updated mixture exactly for every phase
    fraction. A step that would make either state negative is rejected rather
    than clipped, because clipping would silently violate the balance.
    """
    old = reconstruct_mixture(state)
    _validate_defect(updated_mixture)
    values = []
    wake_values = []
    for reservoir_index, (parent, child, recovered, old_mix, new_mix) in enumerate(zip(
            (state.parent.rp, state.parent.rm, state.parent.forest,
             state.parent.wall),
            (state.child.rp, state.child.rm, state.child.forest,
             state.child.wall),
            (state.recovered_wake.rp, state.recovered_wake.rm,
             state.recovered_wake.forest, state.recovered_wake.wall),
            (old.rp, old.rm, old.forest, old.wall),
            (updated_mixture.rp, updated_mixture.rm,
             updated_mixture.forest, updated_mixture.wall))):
        delta = np.asarray(new_mix)-np.asarray(old_mix)
        p1 = np.asarray(parent)+delta
        c1 = np.asarray(child)+delta
        w1 = np.asarray(recovered)+delta
        scale = max(float(np.max(np.abs(parent))),
                    float(np.max(np.abs(child))), 1.0)
        tol = 64.0*np.finfo(float).eps*scale
        if np.min(p1) < -tol or np.min(c1) < -tol or np.min(w1) < -tol:
            # Project provisional populations onto the nonnegative simplex
            # while preserving each physical mixture reservoir exactly. Any
            # resulting signed parent/child mismatch is carried by the explicit
            # signed boundary reservoir when that material is swept.
            p1, c1, w1 = (np.maximum(x, 0.0) for x in (p1, c1, w1))
            virgin = 1.0-state.processed_max
            active = state.chi
            wake_weight = state.processed_max-state.chi
            if p1.ndim == 3:
                virgin = virgin[:, :, None]
                active = active[:, :, None]
                wake_weight = wake_weight[:, :, None]
            projected = virgin*p1+active*c1+wake_weight*w1
            target = np.asarray(new_mix)
            factor = np.divide(
                target, projected, out=np.ones_like(target),
                where=projected > 0.0)
            p1, c1, w1 = p1*factor, c1*factor, w1*factor
            missing = (projected <= 0.0) & (target > 0.0)
            if np.any(missing):
                weights = np.stack(np.broadcast_arrays(
                    virgin, active, wake_weight), axis=0)
                owner = np.argmax(weights, axis=0)
                for owner_index, (value, weight) in enumerate(
                        ((p1, virgin), (c1, active), (w1, wake_weight))):
                    mask = missing & (owner == owner_index) & (weight > 0.0)
                    value[mask] = target[mask]/weight[mask]
        values.append((np.maximum(p1, 0.0), np.maximum(c1, 0.0)))
        wake_values.append(np.maximum(w1, 0.0))
    candidate = replace(
        state,
        parent=DefectState(*(pair[0] for pair in values)),
        child=DefectState(*(pair[1] for pair in values)),
        recovered_wake=DefectState(*wake_values))
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


def advance_front(state, new_child_fraction, *, cell_area_m2,
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


def state_metadata_json(state):
    return json.dumps({
        "schema": "full-v34-sparse-front/v4",
        "parent_label": state.parent_label,
        "child_label": state.child_label,
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
        "chi": state.chi, "processed_max": state.processed_max,
        "cleanup_max": state.cleanup_max,
        "boundary_line_density_m2": state.boundary_line_density_m2,
        "boundary_signed_density_m2": state.boundary_signed_density_m2,
    }


def state_from_checkpoint(metadata_json, arrays):
    raw = json.loads(str(metadata_json))
    if raw.get("schema") not in ("full-v34-sparse-front/v3",
                                  "full-v34-sparse-front/v4"):
        raise ValueError("unsupported sparse-front schema")
    parent = DefectState(*(np.asarray(arrays[k]).copy() for k in (
        "parent_rp", "parent_rm", "parent_forest", "parent_wall")))
    child = DefectState(*(np.asarray(arrays[k]).copy() for k in (
        "child_rp", "child_rm", "child_forest", "child_wall")))
    wake = DefectState(*(np.asarray(arrays[k]).copy() for k in (
        "wake_rp", "wake_rm", "wake_forest", "wake_wall")))
    processed = np.asarray(arrays["processed_max"]).copy()
    cleanup = (np.asarray(arrays["cleanup_max"]).copy()
               if "cleanup_max" in arrays else processed.copy())
    state = SparseFrontState(
        parent, child, wake, np.asarray(arrays["chi"]).copy(),
        processed, cleanup,
        np.asarray(arrays["boundary_line_density_m2"]).copy(),
        np.asarray(arrays["boundary_signed_density_m2"]).copy(),
        FrontLedger(**raw["ledger"]), int(raw["parent_label"]),
        int(raw["child_label"]))
    _validate_defect(parent); _validate_defect(child); _validate_defect(wake)
    return state
