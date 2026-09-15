"""Qualified subgrain-to-phase handoff using the common moving-front ledger."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .dislocation_free_energy import phase_owned_free_energy_J_m3
from .moving_front import DefectState, initialize_existing_subgrain_front
from .stored_energy_coupling import common_variational_stored_energy


@dataclass(frozen=True)
class IntragranularPromotion:
    eta: np.ndarray
    orientations_rad: np.ndarray
    parent_label: int
    child_label: int
    inherited_orientation_rad: float
    common_energy_before_J: float
    common_energy_after_J: float
    phase_simplex_residual: float
    front_state: object


def smooth_support_from_component(component_mask, interface_cells=2.0):
    component = np.asarray(component_mask, dtype=bool)
    if component.ndim != 2 or not np.any(component) or interface_cells <= 0.0:
        raise ValueError("a resolved component and positive interface width are required")
    inside = ndimage.distance_transform_edt(component)
    outside = ndimage.distance_transform_edt(~component)
    return np.clip(0.5+(inside-outside)/(2.0*interface_cells), 0.0, 1.0)


def promote_qualified_subgrain(state, recognition, intragranular_parameters,
                               free_energy_parameters, *, represented_thickness_m=None):
    """Allocate child support only after independent physical recognition."""
    if not recognition.get("qualified", False):
        raise ValueError("phase support requires a qualified compatible subgrain")
    if not recognition.get("lower_density_interior", False):
        raise ValueError("qualified geometry lacks a lower-energy recovered interior")
    component = np.asarray(recognition["component_mask"], dtype=bool)
    interior = np.asarray(recognition["interior_mask"], dtype=bool)
    support = smooth_support_from_component(component)
    eta = np.stack((1.0-support, support), axis=2)
    inherited = float(np.mean(state.orientation_rad[interior]))
    orientations = np.array([float(recognition["exterior_orientation_rad"]), inherited])

    parent = DefectState(
        state.mobile_plus_m2.transpose(1, 2, 0),
        state.mobile_minus_m2.transpose(1, 2, 0),
        state.forest_m2.transpose(1, 2, 0),
        np.sum(state.wall_plus_m2+state.wall_minus_m2, axis=0),
    )
    total = (np.sum(state.mobile_plus_m2+state.mobile_minus_m2+state.forest_m2
                    +state.wall_plus_m2+state.wall_minus_m2, axis=0))
    front = initialize_existing_subgrain_front(parent, support, 0, 1)
    spacing = intragranular_parameters.domain_m/support.shape[0]
    thickness = (2.0*intragranular_parameters.burgers_m
                 if represented_thickness_m is None else float(represented_thickness_m))
    parent_wall = np.sum(state.wall_plus_m2+state.wall_minus_m2, axis=0)
    parent_energy = phase_owned_free_energy_J_m3(
        total, parent_wall, free_energy_parameters)
    child_total = np.sum(front.child.rp+front.child.rm+front.child.forest, axis=2)+front.child.wall
    child_energy = phase_owned_free_energy_J_m3(
        child_total, front.child.wall, free_energy_parameters)
    phase_energy = np.stack((parent_energy, child_energy), axis=2)
    mixture, _ = common_variational_stored_energy(eta, phase_energy)
    volume = spacing**2*thickness
    before = float(np.sum(parent_energy)*volume)
    after = float(np.sum(mixture)*volume)
    # Phase allocation is a representation change, not a cleanup event.
    tolerance = 256.0*math.ulp(max(abs(before), abs(after), 1e-300))
    if abs(after-before) > tolerance:
        raise ValueError("intragranular representation handoff changed common defect energy")
    return IntragranularPromotion(
        eta, orientations, 0, 1, inherited, before, after,
        float(np.max(np.abs(np.sum(eta, axis=2)-1.0))), front,
    )


def signed_growth_drive_J_m3(parent_density_m2, child_density_m2, free_energy_parameters):
    """Positive means child advance; swapping states reverses direction exactly."""
    parent = np.asarray(parent_density_m2, dtype=float)
    child = np.asarray(child_density_m2, dtype=float)
    # The scalar direction-control helper compares equally organized states;
    # swapping them must reverse the thermodynamic sign exactly.
    return (phase_owned_free_energy_J_m3(parent, parent, free_energy_parameters)
            -phase_owned_free_energy_J_m3(child, child, free_energy_parameters))
