"""Authoritative dislocation-density inventory for the V23 full model.

Every population is a line length per material volume [m/m^3 = m^-2].
No plane-thickness factor occurs in these state variables.  A 2-D integral
``sum(rho)*dx*dy`` is therefore line length per out-of-plane thickness [m/m].
Junction extent is counted with its declared product-line multiplicity exactly
once.  The legacy scalar fields are derived views and never additional line.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


SIGNED_RESERVOIRS = (
    "mobile_plus_m2", "mobile_minus_m2",
    "forest_plus_m2", "forest_minus_m2",
    "wall_tangle_plus_m2", "wall_tangle_minus_m2",
    "wall_ordered_plus_m2", "wall_ordered_minus_m2",
)


@dataclass(frozen=True)
class DensityInventory:
    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    forest_plus_m2: np.ndarray
    forest_minus_m2: np.ndarray
    wall_tangle_plus_m2: np.ndarray
    wall_tangle_minus_m2: np.ndarray
    wall_ordered_plus_m2: np.ndarray
    wall_ordered_minus_m2: np.ndarray
    junction_m2: np.ndarray

    def validate(self, family_count: int, topology_count: int):
        shape = np.asarray(self.mobile_plus_m2).shape
        if len(shape) != 3 or shape[-1] != family_count:
            raise ValueError("signed densities require grid x Burgers-family layout")
        for name in SIGNED_RESERVOIRS:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"inadmissible density reservoir: {name}")
        junction = np.asarray(self.junction_m2, dtype=float)
        if junction.shape != shape[:2] + (topology_count,):
            raise ValueError("junction density requires grid x topology layout")
        if np.any(~np.isfinite(junction)) or np.any(junction < 0):
            raise ValueError("junction extent must be finite and nonnegative")
        return shape


def junction_multiplicities(topologies):
    return np.asarray(
        [topology.product_line_multiplicity for topology in topologies],
        dtype=float,
    )


def derived_density_fields(inventory: DensityInventory, topologies=()):
    """Return the only authorized scalar projections of the extensive state."""
    mobile = np.sum(inventory.mobile_plus_m2 + inventory.mobile_minus_m2, axis=2)
    forest = np.sum(inventory.forest_plus_m2 + inventory.forest_minus_m2, axis=2)
    tangle = np.sum(
        inventory.wall_tangle_plus_m2 + inventory.wall_tangle_minus_m2, axis=2)
    ordered = np.sum(
        inventory.wall_ordered_plus_m2 + inventory.wall_ordered_minus_m2, axis=2)
    multiplicity = junction_multiplicities(topologies)
    junction_line = np.sum(inventory.junction_m2 * multiplicity, axis=2)
    return {
        "rho_mobile_m2": mobile,
        "rho_forest_m2": forest,
        "rho_wall_tangle_m2": tangle,
        "rho_wall_ordered_m2": ordered,
        "rho_wall_m2": tangle + ordered,
        "rho_junction_line_m2": junction_line,
        "rho_total_m2": mobile + forest + tangle + ordered + junction_line,
        "q_wall_diagnostic": ordered / np.maximum(tangle + ordered, 1e-300),
    }


def line_energy_density_J_m3(inventory, topologies, line_energy_J_m):
    """Reconstruct E_line*rho_total with no sign/family/junction double count."""
    return float(line_energy_J_m) * derived_density_fields(
        inventory, topologies)["rho_total_m2"]


def taylor_obstacle_by_family_m2(inventory, topologies=(), *, wall_weight=1.0,
                                  junction_weight=1.0):
    obstacle = (
        inventory.forest_plus_m2 + inventory.forest_minus_m2
        + float(wall_weight) * (
            inventory.wall_tangle_plus_m2 + inventory.wall_tangle_minus_m2
            + inventory.wall_ordered_plus_m2 + inventory.wall_ordered_minus_m2)
    ).copy()
    for index, topology in enumerate(topologies):
        line = (float(junction_weight) * topology.product_line_multiplicity
                * inventory.junction_m2[..., index])
        obstacle[..., topology.parent_a] += 0.5 * line
        obstacle[..., topology.parent_b] += 0.5 * line
    return obstacle


def taylor_resistance_Pa(inventory, topologies, shear_modulus_Pa, burgers_m,
                         alpha=0.3, wall_weight=1.0, junction_weight=1.0):
    obstacle = taylor_obstacle_by_family_m2(
        inventory, topologies, wall_weight=wall_weight,
        junction_weight=junction_weight)
    return (float(alpha) * float(shear_modulus_Pa) * float(burgers_m)
            * np.sqrt(np.maximum(obstacle, 0.0)))


def from_v22_common_state(state, *, ordered_fraction=None):
    """Declared one-time restart conversion from V22.

    If ``ordered_fraction`` is omitted, the checkpoint's old scalar q is used
    solely as a partition coordinate.  It creates no line and is discarded
    after conversion.  New V23 initial states should pass zero explicitly.
    """
    wall_plus = np.asarray(state.wall_plus_m2, dtype=float)
    wall_minus = np.asarray(state.wall_minus_m2, dtype=float)
    if ordered_fraction is None:
        fraction = np.asarray(state.wall_order, dtype=float)[..., None]
    else:
        fraction = np.asarray(ordered_fraction, dtype=float)
        if fraction.ndim == 0:
            fraction = np.full(wall_plus.shape[:2] + (1,), float(fraction))
        elif fraction.ndim == 2:
            fraction = fraction[..., None]
    if np.any(fraction < 0) or np.any(fraction > 1):
        raise ValueError("ordered restart fraction must lie in [0,1]")
    return DensityInventory(
        np.asarray(state.mobile_plus_m2).copy(),
        np.asarray(state.mobile_minus_m2).copy(),
        np.asarray(state.forest_plus_m2).copy(),
        np.asarray(state.forest_minus_m2).copy(),
        wall_plus * (1.0 - fraction), wall_minus * (1.0 - fraction),
        wall_plus * fraction, wall_minus * fraction,
        np.asarray(state.junction_m2).copy(),
    )


def legacy_scalar_views(inventory, topologies=()):
    """Compatibility views used by the old driver; these are not state."""
    fields = derived_density_fields(inventory, topologies)
    return {
        "rho_forest": np.sum(
            inventory.forest_plus_m2 + inventory.forest_minus_m2, axis=2),
        "rho_wall": fields["rho_wall_m2"],
        "rho": fields["rho_total_m2"],
        "q_wall_v19": fields["q_wall_diagnostic"],
    }


def checkpoint_arrays(inventory: DensityInventory, prefix="v23_density__"):
    """Lossless checkpoint payload; scalar compatibility views are excluded."""
    return {prefix + name: np.asarray(getattr(inventory, name))
            for name in SIGNED_RESERVOIRS + ("junction_m2",)}


def from_checkpoint_arrays(mapping, family_count, topology_count,
                           prefix="v23_density__"):
    required = SIGNED_RESERVOIRS + ("junction_m2",)
    missing = [name for name in required if prefix + name not in mapping]
    if missing:
        raise ValueError("incomplete V23 extensive restart: " + ", ".join(missing))
    result = DensityInventory(**{
        name: np.asarray(mapping[prefix + name]).copy() for name in required})
    result.validate(family_count, topology_count)
    return result
