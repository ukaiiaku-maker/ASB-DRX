"""Coupled V23 accepted map: signed transport/mechanics plus extensive ordering.

This module is the integration seam between the qualified V21 transport and
kinematic residual and the V23 extensive wall state.  Scalar wall order is
disabled in the transport substep.  New captured wall line enters the tangle
reservoir; removal acts proportionally on the existing tangle/ordered split.
The subsequent detailed-balanced exchange changes only that split.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

try:
    from .common_tensorial_wall import (
        CommonWallDriving, CommonWallParameters, CommonWallState,
        accepted_euler_step, resolved_driving_components,
    )
    from .density_state_map import DensityInventory, derived_density_fields
    from .extensive_wall import (
        ExtensiveWallParameters, accepted_ordering_step,
        orientation_gradient_frank_bilby_target_m1,
    )
    from .tensorial_nye import plastic_distortion_from_slip, spectral_derivatives
except ImportError:
    from common_tensorial_wall import (
        CommonWallDriving, CommonWallParameters, CommonWallState,
        accepted_euler_step, resolved_driving_components,
    )
    from density_state_map import DensityInventory, derived_density_fields
    from extensive_wall import (
        ExtensiveWallParameters, accepted_ordering_step,
        orientation_gradient_frank_bilby_target_m1,
    )
    from tensorial_nye import plastic_distortion_from_slip, spectral_derivatives


@dataclass(frozen=True)
class V23CoupledState:
    common: CommonWallState
    density: DensityInventory

    def validate(self, systems, topologies):
        self.common.validate(systems, topologies)
        self.density.validate(len(systems), len(topologies))
        pairs = (
            (self.common.mobile_plus_m2, self.density.mobile_plus_m2),
            (self.common.mobile_minus_m2, self.density.mobile_minus_m2),
            (self.common.forest_plus_m2, self.density.forest_plus_m2),
            (self.common.forest_minus_m2, self.density.forest_minus_m2),
            (self.common.junction_m2, self.density.junction_m2),
            (self.common.wall_plus_m2,
             self.density.wall_tangle_plus_m2+self.density.wall_ordered_plus_m2),
            (self.common.wall_minus_m2,
             self.density.wall_tangle_minus_m2+self.density.wall_ordered_minus_m2),
        )
        for left, right in pairs:
            if not np.array_equal(left, right):
                raise ValueError("common and extensive density states are not synchronized")
        expected_q = derived_density_fields(
            self.density, topologies)["q_wall_diagnostic"]
        if not np.array_equal(self.common.wall_order, expected_q):
                raise ValueError("common wall_order must be the derived V23 diagnostic")


def initialize_coupled_state(shape, systems, topologies, *,
                             mobile_plus_m2=8e13, mobile_minus_m2=7e13,
                             forest_plus_m2=5e13, forest_minus_m2=4e13,
                             wall_tangle_plus_m2=2e13,
                             wall_tangle_minus_m2=1.5e13,
                             junction_m2=1e12, temperature_K=1100.0):
    nx, ny = map(int, shape); families = len(systems)
    family_shape = (nx, ny, families)
    def full(value):
        return np.full(family_shape, float(value))
    density = DensityInventory(
        full(mobile_plus_m2), full(mobile_minus_m2),
        full(forest_plus_m2), full(forest_minus_m2),
        full(wall_tangle_plus_m2), full(wall_tangle_minus_m2),
        np.zeros(family_shape), np.zeros(family_shape),
        np.full((nx, ny, len(topologies)), float(junction_m2)))
    slip = np.zeros(family_shape)
    orientation = np.zeros((nx, ny))
    common = CommonWallState(
        density.mobile_plus_m2, density.mobile_minus_m2,
        density.forest_plus_m2, density.forest_minus_m2,
        density.wall_tangle_plus_m2, density.wall_tangle_minus_m2,
        density.junction_m2, np.zeros((nx, ny)), np.zeros((nx, ny)),
        slip, plastic_distortion_from_slip(slip, systems, orientation),
        np.zeros(family_shape+(3,)), np.zeros(family_shape+(3, 3)),
        orientation, np.full((nx, ny), float(temperature_K)))
    return make_coupled_state(common, density, topologies)


def make_coupled_state(common, density, topologies=()):
    fields = derived_density_fields(density, topologies)
    synced = replace(
        common,
        mobile_plus_m2=density.mobile_plus_m2,
        mobile_minus_m2=density.mobile_minus_m2,
        forest_plus_m2=density.forest_plus_m2,
        forest_minus_m2=density.forest_minus_m2,
        wall_plus_m2=density.wall_tangle_plus_m2+density.wall_ordered_plus_m2,
        wall_minus_m2=density.wall_tangle_minus_m2+density.wall_ordered_minus_m2,
        junction_m2=density.junction_m2,
        wall_order=fields["q_wall_diagnostic"])
    return V23CoupledState(synced, density)


def _remap_wall_split(old_tangle, old_ordered, new_total):
    old_total = old_tangle + old_ordered
    growth = new_total >= old_total
    tangle = np.where(growth, old_tangle + (new_total-old_total),
                      old_tangle-(old_total-new_total))
    ordered = old_ordered.copy()
    if np.min(tangle) < -1e-6:
        raise FloatingPointError("transport removed more than the V23 tangle donor")
    tangle = np.maximum(tangle, 0.0)
    return tangle, ordered


def accepted_coupled_step(state, driving, systems, topologies,
                          common_parameters, extensive_parameters, dt_s):
    """Advance the exact declared split map and return both substep ledgers."""
    state.validate(systems, topologies)
    # Remove every scalar-q energetic/kinetic effect. A tiny positive partition
    # coefficient is used because the frozen V21 parameter validator excludes
    # zero; it is many orders below floating-point relevance at campaign rho.
    transport_parameters = replace(
        common_parameters, wall_order_enabled=False,
        wall_order_amplitude_J_m3=0.0,
        wall_absent_penalty_J_m3=1e-300,
        wall_partition_J_m=1e-300,
        extensive_wall_partition_enabled=True,
        transport_scheme="upwind",
        mobile_correlation_diffusivity_m2_s=0.0)
    transported, transport_residual, transport_scale = accepted_euler_step(
        state.common,
        driving, systems, topologies, transport_parameters, dt_s)
    density_updates = {
        "mobile_plus_m2": transported.mobile_plus_m2,
        "mobile_minus_m2": transported.mobile_minus_m2,
        "forest_plus_m2": transported.forest_plus_m2,
        "forest_minus_m2": transported.forest_minus_m2,
        "junction_m2": transported.junction_m2,
    }
    for sign in ("plus", "minus"):
        tangle, ordered = _remap_wall_split(
            getattr(state.density, f"wall_tangle_{sign}_m2"),
            getattr(state.density, f"wall_ordered_{sign}_m2"),
            getattr(transported, f"wall_{sign}_m2"))
        density_updates[f"wall_tangle_{sign}_m2"] = tangle
        density_updates[f"wall_ordered_{sign}_m2"] = ordered
    transported_density = replace(state.density, **density_updates)
    curl_beta_nye = np.sum(transported.family_nye_m1, axis=2)
    target_nye = orientation_gradient_frank_bilby_target_m1(
        transported.orientation_rad, extensive_parameters.spacing_m)
    drive = resolved_driving_components(
        transported, driving, systems, topologies, transport_parameters)
    ordered_density, ordering_ledger, ordering_scale = accepted_ordering_step(
        transported_density, systems, topologies, transported.orientation_rad,
        target_nye, drive["effective_stress_Pa"], transported.temperature_K,
        extensive_parameters, dt_s*transport_scale)
    result = make_coupled_state(transported, ordered_density, topologies)
    result.validate(systems, topologies)
    return result, {
        "transport_residual": transport_residual,
        "ordering": ordering_ledger,
        "target_nye_m1": target_nye,
        "curl_beta_nye_m1": curl_beta_nye,
        "driving": drive,
    }, {"transport_scale": transport_scale,
        "ordering_scale": ordering_scale,
        "accepted_dt_s": dt_s*transport_scale}


def mechanical_organization_diagnostics(before, after, ledger, spacing_m):
    """Required heterogeneous-minus-homogeneous route observables."""
    slip_increment = after.common.slip-before.common.slip
    gx, gy = spectral_derivatives(slip_increment, spacing_m)
    slip_gradient = np.sqrt(np.sum(gx*gx+gy*gy, axis=2))
    alpha = ledger["curl_beta_nye_m1"]
    return {
        "rss_Pa": ledger["driving"]["raw_stress_Pa"],
        "effective_rss_Pa": ledger["driving"]["effective_stress_Pa"],
        "slip_gradient_m1": slip_gradient,
        "curl_beta_p_m1": alpha,
        "nye_norm_m1": np.linalg.norm(alpha, axis=(-2, -1)),
        "orientation_rad": after.common.orientation_rad,
        "orientation_increment_rad": (
            after.common.orientation_rad-before.common.orientation_rad),
        "frank_bilby_target_m1": ledger["target_nye_m1"],
    }
