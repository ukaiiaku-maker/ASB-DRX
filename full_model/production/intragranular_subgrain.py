"""One-grain/no-preexisting-boundary intragranular subgrain qualification.

The loading fixture imposes a smooth stress concentration, not an orientation
or grain. Differential slip creates the continuous orientation field. Signed
wall populations relax toward the resulting incompatibility by conservative
mobile-to-wall transfer. Recognition is a read-only operation and this module
contains no phase-label allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from .arrhenius_kinetics import (
    ActivatedProcess, EV_J, activated_rate_s, exp_floor_enthalpy_j,
)


FAMILY_NORMALS = np.array([
    [1.0, 0.0], [2.0**-0.5, 2.0**-0.5],
    [0.0, 1.0], [-2.0**-0.5, 2.0**-0.5],
])


@dataclass(frozen=True)
class IntragranularParameters:
    domain_m: float = 10e-6
    burgers_m: float = 2.48e-10
    temperature_K: float = 1300.0
    applied_stress_Pa: float = 650e6
    stress_concentration_radius_m: float = 1.5e-6
    stress_transition_width_m: float = 0.35e-6
    differential_spin_rate_s: float = 160.0
    initial_mobile_each_sign_m2: float = 8e14
    initial_forest_each_family_m2: float = 3e14
    capture_attempt_s: float = 3e5
    capture_h0_eV: float = 0.55
    capture_entropy_kB: float = 0.0
    ordering_attempt_s: float = 3e6
    ordering_h0_eV: float = 0.80
    ordering_entropy_kB: float = 0.0
    recovery_attempt_s: float = 1e6
    recovery_h0_eV: float = 0.90
    recovery_entropy_kB: float = 0.0
    critical_stress_Pa: float = 1.50e9
    exp_a: float = 2.20562
    exp_n: float = 2.52073
    exp_floor: float = 0.03
    minimum_wall_density_m2: float = 8e13
    recognition_misorientation_deg: float = 2.0
    recognition_order: float = 0.45

    def __post_init__(self):
        for name, value in vars(self).items():
            if name.endswith(("_m", "_m2", "_Pa", "_K", "_s", "_s2")):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
        if self.exp_n < 1.0 or not 0.0 <= self.exp_floor <= 1.0:
            raise ValueError("invalid production EXP-floor parameters")


@dataclass(frozen=True)
class IntragranularState:
    differential_slip: np.ndarray
    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    forest_m2: np.ndarray
    wall_plus_m2: np.ndarray
    wall_minus_m2: np.ndarray
    wall_order: np.ndarray
    time_s: float = 0.0
    step: int = 0
    recovered_line_m_inv: float = 0.0
    ordering_heat_J_m3: float = 0.0

    def __post_init__(self):
        slip = np.asarray(self.differential_slip, dtype=float)
        arrays = [np.asarray(x, dtype=float) for x in (
            self.mobile_plus_m2, self.mobile_minus_m2, self.forest_m2,
            self.wall_plus_m2, self.wall_minus_m2,
        )]
        q = np.asarray(self.wall_order, dtype=float)
        if slip.ndim != 2 or any(x.shape != (4,)+slip.shape for x in arrays):
            raise ValueError("four-family populations must match the 2-D slip field")
        if q.shape != slip.shape or np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError("wall order must match the grid and lie in [0,1]")
        if any(np.any(~np.isfinite(x)) or np.any(x < 0.0) for x in arrays):
            raise ValueError("populations must be finite and nonnegative")
        if np.any(~np.isfinite(slip)) or self.time_s < 0.0 or self.step < 0:
            raise ValueError("invalid kinematic state")
        object.__setattr__(self, "differential_slip", slip.copy())
        for name, value in zip(("mobile_plus_m2", "mobile_minus_m2", "forest_m2",
                                "wall_plus_m2", "wall_minus_m2"), arrays):
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "wall_order", q.copy())

    @property
    def orientation_rad(self):
        return 0.5*self.differential_slip

    @property
    def physical_grain_count(self):
        return 1


def initialize_one_grain(grid_points, parameters):
    if grid_points < 32:
        raise ValueError("one-grain fixture requires at least 32 points")
    shape = (grid_points, grid_points)
    family_shape = (4,)+shape
    return IntragranularState(
        np.zeros(shape),
        np.full(family_shape, parameters.initial_mobile_each_sign_m2),
        np.full(family_shape, parameters.initial_mobile_each_sign_m2),
        np.full(family_shape, parameters.initial_forest_each_family_m2),
        np.zeros(family_shape), np.zeros(family_shape), np.zeros(shape),
    )


def _coordinates(n, domain):
    axis = (np.arange(n)+0.5)*domain/n-0.5*domain
    return np.meshgrid(axis, axis, indexing="ij")


def loading_profile(state, parameters):
    x, y = _coordinates(state.differential_slip.shape[0], parameters.domain_m)
    radius = np.hypot(x, y)
    return 0.5*(1.0-np.tanh(
        (radius-parameters.stress_concentration_radius_m)
        /parameters.stress_transition_width_m))


def _gradient(field, spacing):
    return ((np.roll(field, -1, 0)-np.roll(field, 1, 0))/(2*spacing),
            (np.roll(field, -1, 1)-np.roll(field, 1, 1))/(2*spacing))


def incompatibility_by_family_m2(orientation_rad, parameters):
    """Least-norm family inventory for grad[2 sin(theta/2)]/b."""
    n = orientation_rad.shape[0]
    spacing = parameters.domain_m/n
    chord = 2.0*np.sin(0.5*orientation_rad)
    gx, gy = _gradient(chord, spacing)
    vector = np.stack((gx, gy), axis=0)/parameters.burgers_m
    pseudo = np.linalg.pinv(FAMILY_NORMALS.T)
    return np.einsum("ai,ixy->axy", pseudo, vector)


def reconstructed_incompatibility_vector_m2(state):
    signed = state.wall_plus_m2-state.wall_minus_m2
    return np.einsum("ai,axy->ixy", FAMILY_NORMALS, signed)


def _process_rate(attempt, entropy, h0_eV, stress, p, name):
    process = ActivatedProcess(name, attempt, entropy, attempt)
    enthalpy = exp_floor_enthalpy_j(
        stress, h0_eV*EV_J, p.critical_stress_Pa,
        p.exp_a, p.exp_n, p.exp_floor,
    )
    return activated_rate_s(process, enthalpy, p.temperature_K)


def intragranular_step(state, dt_s, parameters):
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt must be finite and positive")
    profile = loading_profile(state, parameters)
    slip = state.differential_slip + 2.0*parameters.differential_spin_rate_s*profile*dt_s
    theta = 0.5*slip
    target_signed = incompatibility_by_family_m2(theta, parameters)
    target_plus = np.maximum(target_signed, 0.0)
    target_minus = np.maximum(-target_signed, 0.0)

    capture_rate = _process_rate(
        parameters.capture_attempt_s, parameters.capture_entropy_kB,
        parameters.capture_h0_eV, parameters.applied_stress_Pa,
        parameters, "wall-capture",
    )
    fraction = -math.expm1(-capture_rate*dt_s)
    mp, mm = state.mobile_plus_m2.copy(), state.mobile_minus_m2.copy()
    wp, wm = state.wall_plus_m2.copy(), state.wall_minus_m2.copy()
    for mobile, wall, target in ((mp, wp, target_plus), (mm, wm, target_minus)):
        delta = fraction*(target-wall)
        delta = np.maximum(delta, -wall)
        delta = np.minimum(delta, mobile)
        wall += delta
        mobile -= delta

    wall_total = np.sum(wp+wm, axis=0)
    signed_magnitude = np.linalg.norm(
        np.einsum("ai,axy->ixy", FAMILY_NORMALS, wp-wm), axis=0)
    polarization = signed_magnitude/np.maximum(wall_total, 1.0)
    density_activation = wall_total/(wall_total+parameters.minimum_wall_density_m2)
    q_equilibrium = np.clip(polarization*density_activation, 0.0, 1.0)
    order_rate = _process_rate(
        parameters.ordering_attempt_s, parameters.ordering_entropy_kB,
        parameters.ordering_h0_eV, 0.0, parameters, "lagb-ordering",
    )
    order_fraction = -math.expm1(-order_rate*dt_s)
    q = state.wall_order+order_fraction*(q_equilibrium-state.wall_order)

    # Recovery is enabled by an enclosing ordered wall and acts only on neutral
    # content; it therefore lowers interior energy without changing Burgers content.
    shell = (profile > 0.08) & (profile < 0.92)
    closure_signal = float(np.mean(q[shell])) if np.any(shell) else 0.0
    recovery_rate = _process_rate(
        parameters.recovery_attempt_s, parameters.recovery_entropy_kB,
        parameters.recovery_h0_eV, 0.0, parameters, "subgrain-recovery",
    )*closure_signal
    recovery_fraction = -math.expm1(-recovery_rate*dt_s)
    interior = profile > 0.92
    removed = np.zeros_like(mp)
    neutral = np.minimum(mp, mm)*recovery_fraction
    removed[:, interior] = neutral[:, interior]
    mp -= removed; mm -= removed
    forest = state.forest_m2.copy()
    forest_removed = np.zeros_like(forest)
    forest_removed[:, interior] = recovery_fraction*forest[:, interior]
    forest -= forest_removed
    spacing = parameters.domain_m/theta.shape[0]
    recovered = float(np.sum(2.0*removed+forest_removed)*spacing**2/parameters.domain_m)
    # Ordering heat is a declared nonnegative kinetic diagnostic. Detailed
    # common-functional energy accounting is performed at promotion/front stage.
    heat_increment = float(np.mean(np.maximum(q-state.wall_order, 0.0))
                           *parameters.minimum_wall_density_m2
                           *0.5*45e9*parameters.burgers_m**2)
    return IntragranularState(
        slip, mp, mm, forest, wp, wm, q,
        state.time_s+dt_s, state.step+1,
        state.recovered_line_m_inv+recovered,
        state.ordering_heat_J_m3+heat_increment,
    )


def advance_intragranular(state, duration_s, dt_s, parameters):
    steps = int(round(duration_s/dt_s))
    if steps <= 0 or not math.isclose(steps*dt_s, duration_s, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("duration must be a positive integer multiple of dt")
    current = state
    for _ in range(steps):
        current = intragranular_step(current, dt_s, parameters)
    return current


def recognize_subgrain(state, parameters):
    """Measure an orientation island and independent signed-content closure."""
    theta = state.orientation_rad
    exterior_reference = float(np.median(theta))
    threshold = math.radians(parameters.recognition_misorientation_deg)
    contrast = np.abs(theta-exterior_reference)
    # Segment the resolved plateau core, not the entire diffuse transition.
    # The relative level is a measurement convention and does not enter the
    # evolution equations or assign an orientation.
    candidate = contrast >= max(threshold, 0.75*float(np.max(contrast)))
    labels, count = ndimage.label(candidate)
    if count == 0:
        return {"qualified": False, "reason": "no_resolved_orientation_plateau"}
    sizes = ndimage.sum(candidate, labels, range(1, count+1))
    component = labels == 1+int(np.argmax(sizes))
    interior = ndimage.binary_erosion(component, iterations=2)
    shell = ndimage.binary_dilation(component, iterations=2) & ~interior
    if not np.any(interior) or not np.any(shell):
        return {"qualified": False, "reason": "unresolved_interior_or_boundary"}
    spacing = parameters.domain_m/theta.shape[0]
    gx, gy = _gradient(theta, spacing)
    grad = np.hypot(gx, gy)
    theta_in = float(np.mean(theta[interior]))
    theta_out = float(np.mean(theta[~component]))
    mis = abs(theta_in-theta_out)
    order_support = float(np.mean(state.wall_order[shell]))
    closure = float(np.mean(state.wall_order[shell] >= parameters.recognition_order))

    # Independent Frank--Bilby comparison along radial bins through the actual
    # evolved wall inventory. Orientation and wall fields are measured separately.
    coords = np.argwhere(interior)
    center_index = np.mean(coords, axis=0)
    ii, jj = np.indices(theta.shape)
    dx = (ii-center_index[0])*spacing; dy = (jj-center_index[1])*spacing
    radius = np.hypot(dx, dy)
    normal_x = np.divide(dx, radius, out=np.zeros_like(dx), where=radius > 0)
    normal_y = np.divide(dy, radius, out=np.zeros_like(dy), where=radius > 0)
    incompat = reconstructed_incompatibility_vector_m2(state)
    radial = incompat[0]*normal_x+incompat[1]*normal_y
    bins = np.arange(0.0, 0.5*parameters.domain_m+spacing, spacing)
    indices = np.clip(np.digitize(radius.ravel(), bins)-1, 0, len(bins)-2)
    sums = np.bincount(indices, weights=radial.ravel(), minlength=len(bins)-1)
    nums = np.bincount(indices, minlength=len(bins)-1)
    radial_mean = np.divide(sums, nums, out=np.zeros_like(sums), where=nums > 0)
    wall_chord = abs(parameters.burgers_m*np.sum(radial_mean)*spacing)
    orientation_chord = abs(2.0*math.sin(0.5*(theta_out-theta_in)))
    fb_residual = wall_chord-orientation_chord
    fb_relative = abs(fb_residual)/max(orientation_chord, 1e-30)
    total_density = np.sum(
        state.mobile_plus_m2+state.mobile_minus_m2+state.forest_m2
        +state.wall_plus_m2+state.wall_minus_m2, axis=0)
    result = {
        "qualified": bool(mis >= threshold and closure >= 0.70
                          and fb_relative <= 0.20
                          and float(np.mean(grad[interior])) < float(np.mean(grad[shell]))),
        "area_m2": float(np.sum(component)*spacing**2),
        "equivalent_radius_m": float(math.sqrt(np.sum(component)*spacing**2/math.pi)),
        "interior_orientation_rad": theta_in,
        "exterior_orientation_rad": theta_out,
        "misorientation_rad": mis,
        "misorientation_deg": math.degrees(mis),
        "interior_orientation_gradient_rms_m1": float(np.sqrt(np.mean(grad[interior]**2))),
        "boundary_orientation_gradient_rms_m1": float(np.sqrt(np.mean(grad[shell]**2))),
        "boundary_order_mean": order_support,
        "boundary_order_closure_fraction": closure,
        "frank_bilby_orientation_chord": orientation_chord,
        "frank_bilby_wall_chord": wall_chord,
        "frank_bilby_residual": fb_residual,
        "frank_bilby_relative_residual": fb_relative,
        "interior_total_density_m2": float(np.mean(total_density[interior])),
        "exterior_total_density_m2": float(np.mean(total_density[~component])),
        "lower_density_interior": bool(np.mean(total_density[interior]) < np.mean(total_density[~component])),
        "component_mask": component,
        "interior_mask": interior,
        "shell_mask": shell,
    }
    return result


def save_checkpoint(path, state):
    np.savez(Path(path), schema="asb-drx/v18-intragranular/v1",
             differential_slip=state.differential_slip,
             mobile_plus_m2=state.mobile_plus_m2, mobile_minus_m2=state.mobile_minus_m2,
             forest_m2=state.forest_m2, wall_plus_m2=state.wall_plus_m2,
             wall_minus_m2=state.wall_minus_m2, wall_order=state.wall_order,
             time_s=state.time_s, step=state.step,
             recovered_line_m_inv=state.recovered_line_m_inv,
             ordering_heat_J_m3=state.ordering_heat_J_m3)


def load_checkpoint(path):
    with np.load(Path(path), allow_pickle=False) as data:
        if str(data["schema"]) != "asb-drx/v18-intragranular/v1":
            raise ValueError("unsupported intragranular checkpoint")
        return IntragranularState(
            data["differential_slip"], data["mobile_plus_m2"], data["mobile_minus_m2"],
            data["forest_m2"], data["wall_plus_m2"], data["wall_minus_m2"],
            data["wall_order"], float(data["time_s"]), int(data["step"]),
            float(data["recovered_line_m_inv"]), float(data["ordering_heat_J_m3"]))
