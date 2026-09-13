"""Frozen-state nonlinear evolution for the delayed junction-memory closure."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

from .arrhenius_v3 import ArrheniusMechanism
from .vector_topology_cdd_v3 import JunctionReaction, VectorTopologyNetwork


@dataclass(frozen=True)
class FrozenMemoryParameters:
    network: VectorTopologyNetwork
    glide: tuple[ArrheniusMechanism, ...]
    event_lengths_m: np.ndarray
    resolved_stress_Pa: np.ndarray
    forest_coefficients_Pa_m2: np.ndarray
    mobile_diffusivity_m2_s: np.ndarray
    memory_relaxation_s: np.ndarray
    temperature_K: float
    domain_m: float

    def __post_init__(self) -> None:
        reactions = len(self.network.reactions)
        arrays = {
            "event_lengths_m": (self.event_lengths_m, (4,)),
            "resolved_stress_Pa": (self.resolved_stress_Pa, (4,)),
            "forest_coefficients_Pa_m2": (
                self.forest_coefficients_Pa_m2, (4, reactions)
            ),
            "mobile_diffusivity_m2_s": (self.mobile_diffusivity_m2_s, (8,)),
            "memory_relaxation_s": (self.memory_relaxation_s, (reactions,)),
        }
        if len(self.glide) != 4:
            raise ValueError("exactly four glide mechanisms are required")
        for name, (value, shape) in arrays.items():
            array = np.asarray(value, dtype=float)
            if array.shape != shape or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must have shape {shape} and be finite")
            object.__setattr__(self, name, array.copy())
        if np.any(self.event_lengths_m <= 0.0):
            raise ValueError("event lengths must be positive")
        if np.any(self.forest_coefficients_Pa_m2 < 0.0):
            raise ValueError("forest coefficients must be nonnegative")
        if np.any(self.mobile_diffusivity_m2_s <= 0.0):
            raise ValueError("mobile diffusivities must be positive")
        if np.any(self.memory_relaxation_s <= 0.0):
            raise ValueError("memory relaxation times must be positive")
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0.0:
            raise ValueError("temperature must be positive")
        if not math.isfinite(self.domain_m) or self.domain_m <= 0.0:
            raise ValueError("domain must be positive")


@dataclass(frozen=True)
class FrozenMemoryState:
    mobile_m2: np.ndarray
    junction_m2: np.ndarray
    memory_m2: np.ndarray
    time_s: float = 0.0

    def __post_init__(self) -> None:
        mobile = np.asarray(self.mobile_m2, dtype=float)
        junction = np.asarray(self.junction_m2, dtype=float)
        memory = np.asarray(self.memory_m2, dtype=float)
        if mobile.ndim != 2 or mobile.shape[0] != 8:
            raise ValueError("mobile state must have shape (8,cells)")
        if junction.ndim != 2 or junction.shape[1] != mobile.shape[1]:
            raise ValueError("junction state must have shape (reactions,cells)")
        if memory.shape != junction.shape:
            raise ValueError("memory must match junction state")
        if any(np.any(~np.isfinite(x)) for x in (mobile, junction, memory)):
            raise ValueError("nonlinear memory fields must be finite")
        if any(np.any(x < 0.0) for x in (mobile, junction, memory)):
            raise ValueError("nonlinear memory fields must be nonnegative")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("time must be finite and nonnegative")
        object.__setattr__(self, "mobile_m2", mobile.copy())
        object.__setattr__(self, "junction_m2", junction.copy())
        object.__setattr__(self, "memory_m2", memory.copy())


def homogeneous_equilibrium_state(
    cells: int,
    mobile_density_m2: float,
    parameters: FrozenMemoryParameters,
) -> FrozenMemoryState:
    if cells < 8 or mobile_density_m2 <= 0.0:
        raise ValueError("at least eight cells and positive density are required")
    mobile = np.full((8, cells), mobile_density_m2)
    junction = np.empty((len(parameters.network.reactions), cells))
    for index, reaction in enumerate(parameters.network.reactions):
        forward, reverse = reaction.rate_constants_s_inv(parameters.temperature_K)
        first, second = reaction.parent_species
        junction[index] = (
            forward / reverse * mobile[first] * mobile[second]
            / reaction.reference_density_m2
        )
    return FrozenMemoryState(mobile, junction, junction.copy())


def _pack(state: FrozenMemoryState) -> np.ndarray:
    return np.concatenate((
        state.mobile_m2.ravel(), state.junction_m2.ravel(), state.memory_m2.ravel()
    ))


def _unpack(vector: np.ndarray, cells: int, reactions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_size = 8 * cells
    reaction_size = reactions * cells
    mobile = vector[:mobile_size].reshape(8, cells)
    junction = vector[mobile_size:mobile_size + reaction_size].reshape(reactions, cells)
    memory = vector[mobile_size + reaction_size:].reshape(reactions, cells)
    return mobile, junction, memory


def _spectral_derivative(field: np.ndarray, domain_m: float, order: int) -> np.ndarray:
    cells = field.shape[-1]
    wave = 2.0 * np.pi * np.fft.fftfreq(cells, d=domain_m / cells)
    multiplier = (1j * wave) ** order
    return np.fft.ifft(np.fft.fft(field, axis=-1) * multiplier, axis=-1).real


def _cell_velocities_m_s(memory_m2: np.ndarray, parameters: FrozenMemoryParameters) -> np.ndarray:
    cells = memory_m2.shape[1]
    velocities = np.empty((8, cells))
    resistance = parameters.forest_coefficients_Pa_m2 @ memory_m2
    for family, mechanism in enumerate(parameters.glide):
        stress = float(parameters.resolved_stress_Pa[family])
        effective = np.sign(stress) * np.maximum(abs(stress) - resistance[family], 0.0)
        enthalpy = mechanism.enthalpy
        h0 = enthalpy.enthalpy_scale_J(parameters.temperature_K)
        stress_scale = enthalpy.stress_scale_Pa(parameters.temperature_K)
        barrier = h0 * (
            enthalpy.floor_fraction
            + (1.0 - enthalpy.floor_fraction) * np.exp(
                -enthalpy.shape_a * (np.abs(effective) / stress_scale) ** enthalpy.shape_n
            )
        )
        entropy = mechanism.entropy.entropy_kB(parameters.temperature_K)
        prefactor = mechanism.attempt_frequency_s_inv * math.exp(entropy)
        loaded = prefactor * np.exp(
            -barrier / (1.380649e-23 * parameters.temperature_K)
        )
        unloaded = mechanism.one_way_rate_s_inv(0.0, parameters.temperature_K)
        velocity = parameters.event_lengths_m[family] * np.sign(effective) * (
            loaded - unloaded
        )
        velocities[family] = velocity
        velocities[4 + family] = -velocity
    return velocities


def frozen_memory_rhs(
    state: FrozenMemoryState,
    parameters: FrozenMemoryParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the nonlinear spectral semidiscrete equations."""

    mobile = state.mobile_m2
    junction = state.junction_m2
    memory = state.memory_m2
    velocities = _cell_velocities_m_s(memory, parameters)
    mobile_rate = -_spectral_derivative(mobile * velocities, parameters.domain_m, 1)
    mobile_rate += parameters.mobile_diffusivity_m2_s[:, None] * _spectral_derivative(
        mobile, parameters.domain_m, 2
    )
    junction_rate = np.zeros_like(junction)
    for index, reaction in enumerate(parameters.network.reactions):
        first, second = reaction.parent_species
        forward, reverse = reaction.rate_constants_s_inv(parameters.temperature_K)
        rate = (
            forward * mobile[first] * mobile[second] / reaction.reference_density_m2
            - reverse * junction[index]
        )
        mobile_rate[first] -= rate
        mobile_rate[second] -= rate
        junction_rate[index] += rate
        if reaction.product_velocity_m_s != 0.0:
            junction_rate[index] -= reaction.product_velocity_m_s * _spectral_derivative(
                junction[index], parameters.domain_m, 1
            )
        if reaction.product_diffusivity_m2_s != 0.0:
            junction_rate[index] += reaction.product_diffusivity_m2_s * _spectral_derivative(
                junction[index], parameters.domain_m, 2
            )
    memory_rate = (junction - memory) / parameters.memory_relaxation_s[:, None]
    return mobile_rate, junction_rate, memory_rate


def advance_frozen_memory(
    state: FrozenMemoryState,
    parameters: FrozenMemoryParameters,
    duration_s: float,
    *,
    rtol: float = 2.0e-7,
    atol_density_m2: float = 1.0e5,
) -> tuple[FrozenMemoryState, dict[str, float | int]]:
    """Advance one restartable interval with a stiff implicit integrator."""

    if duration_s <= 0.0 or not math.isfinite(duration_s):
        raise ValueError("duration must be finite and positive")
    cells = state.mobile_m2.shape[1]
    reactions = state.junction_m2.shape[0]

    def rhs(_time: float, vector: np.ndarray) -> np.ndarray:
        mobile, junction, memory = _unpack(vector, cells, reactions)
        # The accepted solution is checked for nonnegativity; small Newton
        # iterates may temporarily leave the positive cone.
        trial = object.__new__(FrozenMemoryState)
        object.__setattr__(trial, "mobile_m2", mobile)
        object.__setattr__(trial, "junction_m2", junction)
        object.__setattr__(trial, "memory_m2", memory)
        object.__setattr__(trial, "time_s", 0.0)
        mobile_rate, junction_rate, memory_rate = frozen_memory_rhs(trial, parameters)
        return np.concatenate((
            mobile_rate.ravel(), junction_rate.ravel(), memory_rate.ravel()
        ))

    solution = solve_ivp(
        rhs, (state.time_s, state.time_s + duration_s), _pack(state),
        method="BDF", rtol=rtol, atol=atol_density_m2,
    )
    if not solution.success:
        raise RuntimeError(f"nonlinear memory integration failed: {solution.message}")
    mobile, junction, memory = _unpack(solution.y[:, -1], cells, reactions)
    scale = max(float(np.max(state.mobile_m2)), 1.0)
    if min(float(np.min(mobile)), float(np.min(junction)), float(np.min(memory))) < -2.0e-9 * scale:
        raise RuntimeError("accepted nonlinear state violated nonnegativity")
    advanced = FrozenMemoryState(
        np.maximum(mobile, 0.0), np.maximum(junction, 0.0),
        np.maximum(memory, 0.0), float(solution.t[-1]),
    )
    return advanced, {
        "rhs_evaluations": int(solution.nfev),
        "jacobian_evaluations": int(solution.njev),
        "linear_decompositions": int(solution.nlu),
        "internal_steps": int(len(solution.t) - 1),
    }
