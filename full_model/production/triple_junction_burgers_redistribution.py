"""Local conservative redistribution of explicit interface Burgers content.

This is a bounded three-edge discriminator for a diffuse triple junction.  At
each cell it moves the three interface Burgers tensors toward their common
mean, multiplied by the symmetric triple-support weight.  The sum over the
three reservoirs is therefore unchanged pointwise.  No lattice Nye, plastic
distortion, support, orientation, or label is modified.

The optional carrier relaxation sets scalar interface line to the exact local
``|B_row|/b`` lower bound.  It is an upper-bound energetic construction, not a
production kinetic law: comparison against the fraction-zero construction
separates the benefit of Burgers redistribution from ordinary vector-neutral
line removal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class TripleJunctionBurgersProjection:
    candidate_state: object
    reaction_fraction: float
    maximum_junction_weight: float
    maximum_realized_coordinate: float
    capacity_limited_cell_count: int
    maximum_pointwise_vector_balance_m1: float
    integrated_vector_balance_m: np.ndarray
    removed_line_cell_sum_m2: float
    carrier_capacity_before_cell_sum_m2: float
    carrier_capacity_after_cell_sum_m2: float
    maximum_carrier_deficit_m2: float
    feasible: bool


def project_triple_junction_burgers(
        state, *, burgers_m, reaction_fraction,
        relax_scalar_line_to_capacity=True):
    """Return a permutation-symmetric, pointwise Burgers-conserving candidate.

    The operator is intentionally restricted to a three-grain/three-interface
    junction.  General network transport requires explicit edge connectivity
    and is outside this atomic discriminator.
    """
    state.validate()
    b = float(burgers_m)
    fraction = float(reaction_fraction)
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError("Burgers magnitude must be positive finite")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("reaction fraction must lie in [0,1]")
    if len(state.grain_ids) != 3 or len(state.interfaces) != 3:
        raise ValueError("the atomic triple-junction probe requires 3 grains/edges")
    pairs = {frozenset((item.grain_a_id, item.grain_b_id))
             for item in state.interfaces}
    expected = {frozenset(pair) for pair in (
        (state.grain_ids[0], state.grain_ids[1]),
        (state.grain_ids[0], state.grain_ids[2]),
        (state.grain_ids[1], state.grain_ids[2]))}
    if pairs != expected:
        raise ValueError("interfaces must form one complete three-edge junction")

    if any(item.boundary_burgers_m1 is None for item in state.interfaces):
        raise ValueError("every interface requires an explicit Burgers tensor")
    burgers = np.stack([
        np.asarray(item.boundary_burgers_m1, dtype=float)
        for item in state.interfaces])
    triple = np.prod(np.maximum(np.asarray(state.supports), 0.0), axis=0)
    peak = float(np.max(triple, initial=0.0))
    weight = (triple/peak if peak > 0.0 else np.zeros_like(triple))
    common = np.mean(burgers, axis=0, keepdims=True)
    direction = common-burgers
    line_capacity = np.stack([
        np.sum(np.asarray(item.boundary_plus_m2)
               +np.asarray(item.boundary_minus_m2), axis=-1)
        for item in state.interfaces])
    requested = fraction*weight
    low = np.zeros_like(requested)
    high = requested.copy()
    # One coordinate per spatial cell is shared by all three edges and rows;
    # hence backtracking cannot break the pointwise vector balance.
    for _ in range(60):
        middle = .5*(low+high)
        trial = burgers+middle[None, ..., None, None]*direction
        required = np.linalg.norm(trial, axis=-1)/b
        feasible = np.all(required <= line_capacity, axis=(0, 3))
        low = np.where(feasible, middle, low)
        high = np.where(feasible, high, middle)
    coordinate = low[None, ..., None, None]
    next_burgers = burgers+coordinate*direction
    capacity_limited = int(np.count_nonzero(
        low < requested-4096*np.finfo(float).eps*np.maximum(requested, 1.0)))
    pointwise_residual = np.sum(next_burgers-burgers, axis=0)
    maximum_balance = float(np.max(np.abs(pointwise_residual), initial=0.0))
    integrated_balance = np.sum(
        pointwise_residual, axis=(0, 1), dtype=np.longdouble).astype(float)

    next_interfaces = []
    removed = before_capacity = after_capacity = maximum_deficit = 0.0
    for component, old_burgers, new_burgers in zip(
            state.interfaces, burgers, next_burgers):
        plus = np.asarray(component.boundary_plus_m2, dtype=float)
        minus = np.asarray(component.boundary_minus_m2, dtype=float)
        line = np.sum(plus+minus, axis=-1)
        old_required = np.linalg.norm(old_burgers, axis=-1)/b
        new_required = np.linalg.norm(new_burgers, axis=-1)/b
        before_capacity += float(np.sum(old_required, dtype=np.longdouble))
        after_capacity += float(np.sum(new_required, dtype=np.longdouble))
        deficit = np.maximum(new_required-line, 0.0)
        maximum_deficit = max(
            maximum_deficit, float(np.max(deficit, initial=0.0)))
        if relax_scalar_line_to_capacity:
            retained = np.divide(
                new_required, line, out=np.ones_like(line), where=line > 0.0)
            retained = np.minimum(retained, 1.0)
            next_plus = plus*retained[..., None]
            next_minus = minus*retained[..., None]
        else:
            next_plus = plus.copy()
            next_minus = minus.copy()
        removed += float(np.sum(
            plus+minus-next_plus-next_minus, dtype=np.longdouble))
        next_interfaces.append(replace(
            component, boundary_plus_m2=next_plus,
            boundary_minus_m2=next_minus,
            boundary_burgers_m1=new_burgers))
    candidate = replace(state, interfaces=tuple(next_interfaces))
    candidate.validate()
    scale = max(float(np.max(np.abs(burgers), initial=0.0)), 1.0)
    balance_tolerance = 4096.0*np.finfo(float).eps*scale
    return TripleJunctionBurgersProjection(
        candidate, fraction, float(np.max(weight, initial=0.0)),
        float(np.max(low, initial=0.0)), capacity_limited,
        maximum_balance, integrated_balance, removed, before_capacity,
        after_capacity, maximum_deficit,
        bool(maximum_balance <= balance_tolerance
             and maximum_deficit <= balance_tolerance))
