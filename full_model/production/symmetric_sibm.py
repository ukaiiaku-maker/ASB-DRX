"""Label-symmetric common-state SIBM isolation model for V25.

This module constructs a clean zero-load bicrystal and evaluates every phase
through the same functional.  Phase labels index physical states; they never
select different equations.  The isolation switches are also used by campaign
drivers to prove which channels are frozen at each sequential activation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .stored_energy_coupling import common_variational_stored_energy


@dataclass(frozen=True)
class DriverActivation:
    phase: bool = True
    conservative_transfer: bool = False
    boundary_storage: bool = False
    recovery_and_heat: bool = False
    constitutive_and_rehardening: bool = False

    @property
    def all_defects_frozen(self):
        return not any((self.conservative_transfer, self.boundary_storage,
                        self.recovery_and_heat,
                        self.constitutive_and_rehardening))


def sequential_activations():
    return (
        DriverActivation(),
        DriverActivation(conservative_transfer=True),
        DriverActivation(conservative_transfer=True, boundary_storage=True),
        DriverActivation(conservative_transfer=True, boundary_storage=True,
                         recovery_and_heat=True),
        DriverActivation(conservative_transfer=True, boundary_storage=True,
                         recovery_and_heat=True,
                         constitutive_and_rehardening=True),
    )


def clean_periodic_bicrystal(n=256, length_m=10e-6, interface_width_m=.3e-6):
    """Return a planar periodic two-grain state with no mechanical history."""
    if n < 16 or length_m <= 0.0 or interface_width_m <= 0.0:
        raise ValueError("clean bicrystal requires a resolved positive domain")
    x = np.arange(n)*length_m/n
    # A child slab bounded by a planar periodic interface pair.
    left, right = .25*length_m, .75*length_m
    child = .5*(np.tanh((x-left)/interface_width_m)
                -np.tanh((x-right)/interface_width_m))
    child = np.clip(child, 0.0, 1.0)
    eta = np.stack((1.0-child, child), axis=-1)
    return {
        "x_m": x, "eta": eta, "total_strain": np.zeros((n, 2, 2)),
        "plastic_strain": np.zeros((n, 2, 2)), "stress_Pa": np.zeros((n, 2, 2)),
        "sweep_history": np.zeros(n), "boundary_reservoir_m2": np.zeros(n),
        "orientations_rad": np.asarray((0.0, np.deg2rad(10.0))),
        "interface_width_m": float(interface_width_m),
    }


def common_complete_phase_energies(defect_J_m3, elastic_J_m3=None,
                                   gnd_J_m3=None, compatibility_J_m3=None,
                                   boundary_J_m3=None):
    """Combine physical-state energies with no parent/child ownership."""
    result = np.asarray(defect_J_m3, dtype=float).copy()
    if result.shape[-1] != 2 or np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError("two finite nonnegative phase-state energies required")
    for term in (elastic_J_m3, gnd_J_m3, compatibility_J_m3, boundary_J_m3):
        if term is not None:
            value = np.asarray(term, dtype=float)
            if value.shape != result.shape or np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError("every phase-owned energy must share the common layout")
            result += value
    return result


def complete_energy(eta, phase_energy_J_m3, spacing_m, kappa_J_m,
                    barrier_J_m3):
    fields = np.asarray(eta, dtype=float)
    phase = np.asarray(phase_energy_J_m3, dtype=float)
    if phase.shape == (2,):
        phase = np.broadcast_to(phase, fields.shape)
    bulk_density, _ = common_variational_stored_energy(fields, phase)
    grad = (np.roll(fields, -1, axis=0)-np.roll(fields, 1, axis=0))/(2*spacing_m)
    interface_density = (.5*kappa_J_m*np.sum(grad*grad, axis=-1)
                         +barrier_J_m3*np.prod(fields*fields, axis=-1))
    return {
        "phase_J_m2": float(np.sum(bulk_density)*spacing_m),
        "interface_J_m2": float(np.sum(interface_density)*spacing_m),
        "total_J_m2": float(np.sum(bulk_density+interface_density)*spacing_m),
    }


def allen_cahn_step(eta, phase_energy_J_m3, spacing_m, kappa_J_m,
                    barrier_J_m3, mobility_m3_J_s, dt_s):
    fields = np.asarray(eta, dtype=float)
    phase = np.asarray(phase_energy_J_m3, dtype=float)
    if phase.shape == (2,):
        phase = np.broadcast_to(phase, fields.shape)
    _, bulk_derivative = common_variational_stored_energy(fields, phase)
    lap = (np.roll(fields, -1, axis=0)-2*fields
           +np.roll(fields, 1, axis=0))/spacing_m**2
    other = fields[:, ::-1]
    derivative = (-kappa_J_m*lap
                  +2*barrier_J_m3*fields*other*other
                  +bulk_derivative)
    # Project onto eta_0+eta_1=1; this operation is exactly label symmetric.
    derivative -= np.mean(derivative, axis=-1, keepdims=True)
    updated = fields-dt_s*mobility_m3_J_s*derivative
    updated = np.clip(updated, 0.0, 1.0)
    updated /= np.sum(updated, axis=-1, keepdims=True)
    return updated, derivative


def flat_front_variational_audit(eta, phase_terms_J_m3, spacing_m,
                                 kappa_J_m, barrier_J_m3,
                                 perturbation_m=1e-10):
    """Decompose the child-expansion derivative of a planar interface pair.

    The mode translates the left and right fronts outwards with unit normal
    speed.  Its components have units 1/m, so each reported derivative is a
    pressure (J/m3) after integrating the one-dimensional energy per area.
    """
    fields = np.asarray(eta, dtype=float)
    if fields.ndim != 2 or fields.shape[1] != 2:
        raise ValueError("planar audit requires (n,2) phase support")
    dx = float(spacing_m)
    grad_child = (np.roll(fields[:, 1], -1)-np.roll(fields[:, 1], 1))/(2*dx)
    direction = np.stack((-np.abs(grad_child), np.abs(grad_child)), axis=-1)
    decomposition = {}
    summed_energy = np.zeros_like(fields)
    for name in ("phase", "defect", "elastic", "gnd", "compatibility", "boundary"):
        energy = np.asarray(phase_terms_J_m3.get(name, np.zeros(2)), dtype=float)
        if energy.shape == (2,):
            energy = np.broadcast_to(energy, fields.shape)
        _, derivative = common_variational_stored_energy(fields, energy)
        decomposition[name+"_Pa"] = float(np.sum(derivative*direction)*dx)
        summed_energy += energy
    grad = (np.roll(fields, -1, axis=0)-np.roll(fields, 1, axis=0))/(2*dx)
    grad_direction = (np.roll(direction, -1, axis=0)
                      -np.roll(direction, 1, axis=0))/(2*dx)
    barrier_derivative = (2*fields[:, 0]*fields[:, 1]**2*direction[:, 0]
                          +2*fields[:, 1]*fields[:, 0]**2*direction[:, 1])
    decomposition["interface_Pa"] = float(np.sum(
        kappa_J_m*grad*grad_direction)*dx
        +np.sum(barrier_J_m3*barrier_derivative)*dx)
    analytical = float(sum(decomposition.values()))
    h = float(perturbation_m)
    plus = fields+h*direction; minus = fields-h*direction
    # The small centered perturbation preserves the simplex exactly.
    fp = complete_energy(plus, summed_energy, dx, kappa_J_m,
                         barrier_J_m3)["total_J_m2"]
    fm = complete_energy(minus, summed_energy, dx, kappa_J_m,
                         barrier_J_m3)["total_J_m2"]
    centered = (fp-fm)/(2*h)
    return {**decomposition, "analytical_total_Pa": analytical,
            "centered_finite_difference_Pa": float(centered),
            "relative_derivative_mismatch": abs(analytical-centered)/max(
                abs(analytical), abs(centered), 1.0)}


def phase_only_directional_audit(parent_energy_J_m3, child_energy_J_m3, *,
                                 mobility_m3_J_s=1e-9, steps=40):
    state = clean_periodic_bicrystal()
    eta = state["eta"].copy(); dx = state["x_m"][1]-state["x_m"][0]
    width = state["interface_width_m"]
    kappa = 5e-7; barrier = kappa/width**2
    energy = np.asarray((parent_energy_J_m3, child_energy_J_m3), dtype=float)
    initial_fraction = float(np.mean(eta[:, 1]))
    signs = []
    # Bound the explicit relaxation by the largest local energy scale.
    dt = .02/(mobility_m3_J_s*max(barrier, np.max(energy), 1.0))
    first_derivative = None
    for _ in range(int(steps)):
        eta, derivative = allen_cahn_step(
            eta, energy, dx, kappa, barrier, mobility_m3_J_s, dt)
        if first_derivative is None:
            first_derivative = derivative.copy()
        delta = float(np.mean(eta[:, 1])-initial_fraction)
        if abs(delta) > 1e-15:
            signs.append(int(np.sign(delta)))
    final_fraction = float(np.mean(eta[:, 1]))
    # Two interfaces change child length, hence half the volume-derived motion.
    displacement = .5*(final_fraction-initial_fraction)*(
        state["x_m"].size*dx)
    expected = np.sign(parent_energy_J_m3-child_energy_J_m3)
    return {
        "initial_child_fraction": initial_fraction,
        "final_child_fraction": final_fraction,
        "displacement_m": displacement,
        "displacement_interface_widths": displacement/width,
        "expected_sign": int(expected),
        "observed_sign": int(np.sign(displacement)),
        "stable_velocity_sign": bool(not signs or all(x == signs[0] for x in signs)),
        "first_phase_only_rate_sign": int(np.sign(-np.mean(
            first_derivative[:, 1][eta[:, 1] > .05]))),
        "all_defects_frozen": DriverActivation().all_defects_frozen,
        "zero_load": True,
    }
