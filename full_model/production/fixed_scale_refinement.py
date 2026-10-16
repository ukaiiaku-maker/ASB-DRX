"""Conservative fixed-domain refinement of the common multi-grain state.

This utility is for spatial qualification, not state evolution.  Intensive
owner line fields are refined through their physical support-weighted
inventories, so changing grid representation does not create material line.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
from scipy import ndimage

from .common_tensorial_wall import CommonWallState
from .multigrain_common_state import LINE_FIELDS


def _periodic_refine(value, factor):
    array = np.asarray(value)
    if array.shape[0] == 0 or array.shape[1] == 0:
        return np.zeros(
            (array.shape[0]*factor, array.shape[1]*factor)+array.shape[2:],
            dtype=float)
    zoom = (float(factor), float(factor))+(1.0,)*(array.ndim-2)
    return ndimage.zoom(
        array, zoom, order=1, mode="grid-wrap", prefilter=False,
        grid_mode=True)


def refine_multigrain_common_state(state, factor=2):
    """Return an integer fixed-domain refinement and an inventory audit."""
    state.validate()
    factor = int(factor)
    if factor < 2:
        raise ValueError("refinement factor must be an integer >= 2")
    support = np.asarray(state.supports, dtype=float)
    refined_support = np.stack(
        [_periodic_refine(item, factor) for item in support], axis=0)
    refined_support = np.maximum(refined_support, 0.0)
    refined_support /= np.maximum(
        np.sum(refined_support, axis=0, keepdims=True), 1e-300)
    owners = []
    inventory_before = {}
    inventory_after = {}
    for owner_index, owner in enumerate(state.owners):
        old_weight = support[owner_index]
        new_weight = refined_support[owner_index]
        arrays = {}
        for item in fields(CommonWallState):
            value = np.asarray(getattr(owner, item.name), dtype=float)
            if value.shape[2:3] == (0,):
                arrays[item.name] = np.zeros(
                    (value.shape[0]*factor, value.shape[1]*factor)
                    +value.shape[2:])
            elif item.name in LINE_FIELDS:
                trailing = (None,)*(value.ndim-2)
                extensive = old_weight[(...,)+trailing]*value
                refined_extensive = _periodic_refine(extensive, factor)
                direct = _periodic_refine(value, factor)
                denominator = new_weight[(...,)+trailing]
                arrays[item.name] = np.divide(
                    refined_extensive, denominator, out=direct,
                    where=denominator > 1e-14)
                inventory_before[f"owner{owner_index}:{item.name}"] = float(
                    np.sum(extensive, dtype=np.longdouble))
                inventory_after[f"owner{owner_index}:{item.name}"] = float(
                    np.sum(denominator*arrays[item.name],
                           dtype=np.longdouble)/factor**2)
            else:
                arrays[item.name] = _periodic_refine(value, factor)
        owners.append(CommonWallState(**arrays))
    interfaces = []
    for interface in state.interfaces:
        replacements = {}
        for name in (
                "first_passage_fraction", "revisit_fraction",
                "boundary_plus_m2", "boundary_minus_m2",
                "boundary_junction_m2", "boundary_burgers_m1"):
            value = getattr(interface, name)
            if value is not None:
                replacements[name] = _periodic_refine(value, factor)
        interfaces.append(replace(interface, **replacements))
    refined = replace(
        state, supports=refined_support, owners=tuple(owners),
        interfaces=tuple(interfaces))
    refined.validate()
    errors = {
        key: abs(inventory_after[key]-value)/max(abs(value), 1.0)
        for key, value in inventory_before.items()}
    audit = {
        "factor": factor,
        "maximum_support_closure": float(np.max(np.abs(
            np.sum(refined_support, axis=0)-1.0))),
        "maximum_relative_line_inventory_error": max(
            errors.values(), default=0.0),
        "relative_line_inventory_error": errors,
    }
    return refined, audit
