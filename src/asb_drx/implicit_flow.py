"""Backward-Euler local EXP-floor flow coupled through antiplane equilibrium."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .analytical import ExpFloorLaw, KB_J_PER_K
from .antiplane import AntiplaneEquilibrium, solve_periodic_antiplane


@dataclass(frozen=True)
class ImplicitFlowIncrement:
    grain_increment: np.ndarray
    plastic_increment: np.ndarray
    equilibrium: AntiplaneEquilibrium
    newton_iterations: int
    maximum_residual: float


def _vectorized_signed_rates_and_tangent(
    stress_Pa: np.ndarray,
    density_m2: np.ndarray,
    temperature_K: np.ndarray,
    weights: np.ndarray,
    law: ExpFloorLaw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grain rates, interpolated rate, and d(rate)/d(stress)."""
    stress = np.asarray(stress_Pa, dtype=float)
    density = np.asarray(density_m2, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    phase_weights = np.asarray(weights, dtype=float)
    if density.shape != (2, *stress.shape) or phase_weights.shape != density.shape:
        raise ValueError("density and weights must have shape (2, *stress.shape)")
    if temperature.shape != stress.shape:
        raise ValueError("temperature shape mismatch")

    reduced_temperature = (
        temperature - law.reference_temperature_K
    ) / law.reference_temperature_K
    barrier_scale = law.barrier_ref_J * np.exp(
        -law.barrier_temperature_coefficient * reduced_temperature
    ) - KB_J_PER_K * law.barrier_entropy_kB * (
        temperature - law.reference_temperature_K
    )
    stress_scale = law.stress_ref_Pa * np.exp(
        -law.stress_temperature_coefficient * reduced_temperature
    )
    if np.any(barrier_scale <= 0.0):
        raise ValueError("barrier scale is nonpositive")

    q = law.taylor_geometry_factor * law.burgers_m * np.sqrt(density)
    obstacle = np.abs(stress)[None, :, :] / q
    reduced_stress = obstacle / stress_scale[None, :, :]
    exponential = np.exp(-law.shape_a * reduced_stress**law.shape_n)
    barrier = barrier_scale[None, :, :] * (
        law.floor_fraction + (1.0 - law.floor_fraction) * exponential
    )
    forward_magnitude = (
        law.rate_prefactor_s_inv
        * q**law.density_exponent_p
        * np.exp(-barrier / (KB_J_PER_K * temperature[None, :, :]))
    )
    reverse_magnitude = (
        law.rate_prefactor_s_inv
        * q**law.density_exponent_p
        * np.exp(-barrier_scale[None, :, :] / (KB_J_PER_K * temperature[None, :, :]))
    )
    magnitude = np.maximum(forward_magnitude - reverse_magnitude, 0.0)
    grain_rate = np.sign(stress)[None, :, :] * magnitude

    activation_volume = (
        barrier_scale[None, :, :]
        * (1.0 - law.floor_fraction)
        * law.shape_a
        * law.shape_n
        * reduced_stress ** (law.shape_n - 1.0)
        * exponential
        / stress_scale[None, :, :]
    )
    grain_tangent = (
        forward_magnitude
        * activation_volume
        / (KB_J_PER_K * temperature[None, :, :] * q)
    )
    grain_tangent[:, stress == 0.0] = 0.0
    return (
        grain_rate,
        np.sum(phase_weights * grain_rate, axis=0),
        np.sum(phase_weights * grain_tangent, axis=0),
    )


def backward_euler_antiplane_flow(
    applied_shear: float,
    plastic_shear: np.ndarray,
    applied_shear_increment: float,
    density_m2: np.ndarray,
    temperature_K: np.ndarray,
    weights: np.ndarray,
    dt_s: float,
    dx_m: float,
    shear_modulus_Pa: float,
    law: ExpFloorLaw,
    *,
    maximum_newton_iterations: int = 20,
    maximum_gmres_iterations: int = 80,
    relative_tolerance: float = 1.0e-10,
) -> ImplicitFlowIncrement:
    """Solve ``delta_gamma = dt * rate(stress_new)`` by damped Newton--GMRES."""
    plastic = np.asarray(plastic_shear, dtype=float)
    if plastic.ndim != 2 or not np.all(np.isfinite(plastic)):
        raise ValueError("plastic_shear must be a finite two-dimensional field")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if maximum_newton_iterations < 1 or maximum_gmres_iterations < 1:
        raise ValueError("iteration limits must be positive")
    if not math.isfinite(relative_tolerance) or not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be finite and in (0,1)")

    new_applied = applied_shear + applied_shear_increment
    trial_without_flow = solve_periodic_antiplane(
        new_applied, plastic, shear_modulus_Pa, dx_m
    )
    _, predictor_rate, predictor_tangent = _vectorized_signed_rates_and_tangent(
        trial_without_flow.stress_x_Pa, density_m2, temperature_K, weights, law
    )
    increment = dt_s * predictor_rate / (
        1.0 + dt_s * shear_modulus_Pa * predictor_tangent
    )

    def evaluate(candidate: np.ndarray):
        equilibrium = solve_periodic_antiplane(
            new_applied, plastic + candidate, shear_modulus_Pa, dx_m
        )
        grain_rate, rate, tangent = _vectorized_signed_rates_and_tangent(
            equilibrium.stress_x_Pa, density_m2, temperature_K, weights, law
        )
        residual = candidate - dt_s * rate
        return residual, equilibrium, grain_rate, tangent

    residual, equilibrium, grain_rate, tangent = evaluate(increment)
    scale = max(
        float(np.max(np.abs(increment))),
        dt_s * float(np.max(np.abs(predictor_rate))),
        abs(applied_shear_increment),
        np.finfo(float).tiny,
    )
    tolerance = relative_tolerance * scale + 32.0 * np.finfo(float).eps * scale
    maximum_residual = float(np.max(np.abs(residual)))

    for iteration in range(maximum_newton_iterations + 1):
        if maximum_residual <= tolerance:
            grain_increment = dt_s * grain_rate
            plastic_increment = np.sum(weights * grain_increment, axis=0)
            final_equilibrium = solve_periodic_antiplane(
                new_applied, plastic + plastic_increment, shear_modulus_Pa, dx_m
            )
            return ImplicitFlowIncrement(
                grain_increment,
                plastic_increment,
                final_equilibrium,
                iteration,
                maximum_residual,
            )
        if iteration == maximum_newton_iterations:
            break

        shape = plastic.shape
        size = plastic.size

        def jacobian_vector(flat_vector: np.ndarray) -> np.ndarray:
            vector = flat_vector.reshape(shape)
            stress_change = solve_periodic_antiplane(
                0.0, vector, shear_modulus_Pa, dx_m
            ).stress_x_Pa
            return (vector - dt_s * tangent * stress_change).ravel()

        operator = LinearOperator((size, size), matvec=jacobian_vector, dtype=float)
        update, info = gmres(
            operator,
            -residual.ravel(),
            rtol=min(1.0e-6, math.sqrt(relative_tolerance)),
            atol=0.0,
            restart=min(40, size),
            maxiter=maximum_gmres_iterations,
        )
        if info != 0 or not np.all(np.isfinite(update)):
            raise RuntimeError(f"implicit flow GMRES failed with info={info}")
        update = update.reshape(shape)

        old_norm = maximum_residual
        accepted = False
        damping = 1.0
        for _ in range(14):
            trial_increment = increment + damping * update
            trial = evaluate(trial_increment)
            trial_norm = float(np.max(np.abs(trial[0])))
            if trial_norm < old_norm:
                increment = trial_increment
                residual, equilibrium, grain_rate, tangent = trial
                maximum_residual = trial_norm
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            raise RuntimeError("implicit flow Newton line search failed")

    raise RuntimeError(
        "implicit flow Newton iteration limit exceeded; "
        f"maximum_residual={maximum_residual:.16g}, tolerance={tolerance:.16g}"
    )
