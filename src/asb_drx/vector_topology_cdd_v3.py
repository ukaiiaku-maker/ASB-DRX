"""Minimal signed-vector/junction CDD model for Mission-v3 falsification.

This module is intentionally smaller than a general higher-dimensional CDD
solver.  Eight mobile species represent the positive and negative populations
of four BCC Burgers families.  Every declared binary reaction creates one
explicit junction-product species.  A shared local line tangent is the
lowest-order vector alignment state.  With that restriction, the reaction
incidence matrix preserves the Nye tensor pointwise by Frank's rule.

The Fourier operator combines conservative aligned transport, positive
correlation diffusion, reversible mass-action junction kinetics, and the
linearization of an interpretable junction-dependent forest resistance.  It
contains no wavelength, negative mobility, or localization multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .arrhenius_v3 import ArrheniusMechanism, KB_J_PER_K
from .bertin_bcc import BURGERS_FAMILIES_CRYSTAL


_CHARACTERS = {"glissile", "sessile", "cross_slip", "climb"}


@dataclass(frozen=True)
class JunctionReaction:
    """One binary mobile reaction and its explicit product topology."""

    parent_species: tuple[int, int]
    character: str
    formation: ArrheniusMechanism
    reaction_free_energy_J: float
    reference_density_m2: float
    product_velocity_m_s: float = 0.0
    product_diffusivity_m2_s: float = 0.0

    def __post_init__(self) -> None:
        first, second = self.parent_species
        if not 0 <= first < 8 or not 0 <= second < 8 or first == second:
            raise ValueError("junction parents must be two distinct signed mobile species")
        if self.character not in _CHARACTERS:
            raise ValueError(f"character must be one of {sorted(_CHARACTERS)}")
        for name in ("reaction_free_energy_J", "product_velocity_m_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not math.isfinite(self.reference_density_m2) or self.reference_density_m2 <= 0.0:
            raise ValueError("reference_density_m2 must be finite and positive")
        if (
            not math.isfinite(self.product_diffusivity_m2_s)
            or self.product_diffusivity_m2_s < 0.0
        ):
            raise ValueError("product diffusivity must be finite and nonnegative")

    def rate_constants_s_inv(
        self, temperature_K: float, density_m2: float | None = None,
    ) -> tuple[float, float]:
        """Forward/reverse constants satisfying exact detailed balance.

        The forward transition state is the declared EXP-floor mechanism.
        The reverse activation free energy is its forward barrier minus the
        reaction-state free-energy difference.  A nonpositive reverse barrier
        is rejected because it requires an explicit barrierless/drag branch.
        """

        forward_barrier = self.formation.activation_free_energy_J(0.0, temperature_K)
        reverse_barrier = forward_barrier - self.reaction_free_energy_J
        if reverse_barrier <= 0.0:
            raise ValueError("junction reverse barrier is nonpositive")
        forward = self.formation.one_way_rate_s_inv(0.0, temperature_K, density_m2)
        reverse = forward * math.exp(
            self.reaction_free_energy_J / (KB_J_PER_K * temperature_K)
        )
        return forward, reverse


@dataclass(frozen=True)
class VectorTopologyNetwork:
    burgers_m: float
    reactions: tuple[JunctionReaction, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.burgers_m) or self.burgers_m <= 0.0:
            raise ValueError("burgers_m must be finite and positive")
        if not self.reactions:
            raise ValueError("at least one explicit junction reaction is required")

    @property
    def mobile_burgers_vectors_m(self) -> np.ndarray:
        base = self.burgers_m * BURGERS_FAMILIES_CRYSTAL
        return np.concatenate((base, -base), axis=0)

    @property
    def species_burgers_vectors_m(self) -> np.ndarray:
        mobile = self.mobile_burgers_vectors_m
        products = np.asarray([
            mobile[first] + mobile[second]
            for first, second in (reaction.parent_species for reaction in self.reactions)
        ])
        if np.any(np.linalg.norm(products, axis=1) <= 1.0e-15 * self.burgers_m):
            raise ValueError("a declared junction has a zero Burgers product")
        return np.concatenate((mobile, products), axis=0)

    @property
    def incidence(self) -> np.ndarray:
        matrix = np.zeros((8 + len(self.reactions), len(self.reactions)))
        for column, reaction in enumerate(self.reactions):
            matrix[reaction.parent_species[0], column] = -1.0
            matrix[reaction.parent_species[1], column] = -1.0
            matrix[8 + column, column] = 1.0
        return matrix

    @property
    def frank_residuals_m(self) -> np.ndarray:
        return self.species_burgers_vectors_m.T @ self.incidence


@dataclass(frozen=True)
class VectorTopologyState:
    """Two-dimensional fields with a lowest-order common line alignment."""

    mobile_m2: np.ndarray
    junction_m2: np.ndarray
    line_tangent_2d: np.ndarray
    dx_m: float
    dy_m: float

    def __post_init__(self) -> None:
        mobile = np.asarray(self.mobile_m2, dtype=float)
        junction = np.asarray(self.junction_m2, dtype=float)
        tangent = np.asarray(self.line_tangent_2d, dtype=float)
        if mobile.ndim != 3 or mobile.shape[0] != 8:
            raise ValueError("mobile_m2 must have shape (8,ny,nx)")
        if junction.ndim != 3 or junction.shape[1:] != mobile.shape[1:]:
            raise ValueError("junction_m2 must have shape (reactions,ny,nx)")
        if np.any(~np.isfinite(mobile)) or np.any(~np.isfinite(junction)):
            raise ValueError("topology populations must be finite")
        if np.any(mobile < 0.0) or np.any(junction < 0.0):
            raise ValueError("topology populations must be nonnegative")
        if tangent.shape != (2,) or not np.all(np.isfinite(tangent)):
            raise ValueError("line tangent must be a finite two-vector")
        norm = float(np.linalg.norm(tangent))
        if abs(norm - 1.0) > 2.0e-12:
            raise ValueError("line tangent must be unit length")
        for name in ("dx_m", "dy_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "mobile_m2", mobile.copy())
        object.__setattr__(self, "junction_m2", junction.copy())
        object.__setattr__(self, "line_tangent_2d", tangent.copy())

    def nye_tensor_m_inv(self, network: VectorTopologyNetwork) -> np.ndarray:
        if self.junction_m2.shape[0] != len(network.reactions):
            raise ValueError("state and network reaction counts differ")
        populations = np.concatenate((self.mobile_m2, self.junction_m2), axis=0)
        return np.einsum(
            "sa,i,syx->aiyx",
            network.species_burgers_vectors_m,
            self.line_tangent_2d,
            populations,
        )


@dataclass(frozen=True)
class TopologyReactionLedger:
    extent_m2: np.ndarray
    maximum_frank_source_residual_m_inv_s: float
    minimum_population_m2: float


def reaction_rates_m2_s(
    state: VectorTopologyState,
    network: VectorTopologyNetwork,
    temperature_K: float,
) -> np.ndarray:
    """Reversible normalized mass-action rates for every explicit channel."""

    if state.junction_m2.shape[0] != len(network.reactions):
        raise ValueError("state and network reaction counts differ")
    rates = np.empty_like(state.junction_m2)
    for index, reaction in enumerate(network.reactions):
        first, second = reaction.parent_species
        density = max(
            float(np.mean(state.mobile_m2[first] + state.mobile_m2[second])),
            reaction.reference_density_m2 * 1.0e-12,
        )
        forward, reverse = reaction.rate_constants_s_inv(temperature_K, density)
        rates[index] = (
            forward * state.mobile_m2[first] * state.mobile_m2[second]
            / reaction.reference_density_m2
            - reverse * state.junction_m2[index]
        )
    return rates


def advance_reactions(
    state: VectorTopologyState,
    network: VectorTopologyNetwork,
    temperature_K: float,
    dt_s: float,
) -> tuple[VectorTopologyState, TopologyReactionLedger]:
    """Advance the topology source, rejecting rather than clipping negativity."""

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    rates = reaction_rates_m2_s(state, network, temperature_K)
    source = np.einsum("sr,ryx->syx", network.incidence, rates)
    populations = np.concatenate((state.mobile_m2, state.junction_m2), axis=0)
    advanced = populations + dt_s * source
    tolerance = 2.0e-13 * max(float(np.max(populations)), 1.0)
    if float(np.min(advanced)) < -tolerance:
        raise RuntimeError("reaction interval violated population nonnegativity")
    advanced = np.maximum(advanced, 0.0)
    result = VectorTopologyState(
        advanced[:8], advanced[8:], state.line_tangent_2d,
        state.dx_m, state.dy_m,
    )
    frank_source = np.einsum(
        "as,syx->ayx", network.species_burgers_vectors_m.T, source
    )
    return result, TopologyReactionLedger(
        dt_s * rates,
        float(np.max(np.abs(frank_source))),
        float(np.min(advanced)),
    )


def arrhenius_forest_linearization(
    mechanisms: tuple[ArrheniusMechanism, ...],
    event_lengths_m: np.ndarray,
    resolved_stress_Pa: np.ndarray,
    junction_density_m2: np.ndarray,
    forest_coefficients_Pa_m2: np.ndarray,
    temperature_K: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and derivative from a declared junction forest resistance.

    ``tau_eff=sign(tau) max(|tau|-sum(h_ar p_r),0)`` with ``h`` in Pa m2.
    The returned derivative ``dv/dp`` has units m3/s.  Positive and negative
    signed populations use opposite velocities from the same family law.
    """

    stresses = np.asarray(resolved_stress_Pa, dtype=float)
    lengths = np.asarray(event_lengths_m, dtype=float)
    junction = np.asarray(junction_density_m2, dtype=float)
    coefficients = np.asarray(forest_coefficients_Pa_m2, dtype=float)
    if len(mechanisms) != 4 or stresses.shape != (4,) or lengths.shape != (4,):
        raise ValueError("four family mechanisms, stresses, and event lengths are required")
    if coefficients.shape != (4, junction.size) or np.any(coefficients < 0.0):
        raise ValueError("forest coefficients must have shape (4,reactions) and be nonnegative")
    velocities = np.zeros(8)
    derivatives = np.zeros((8, junction.size))
    for family in range(4):
        stress = float(stresses[family])
        resistance = float(coefficients[family] @ junction)
        magnitude = max(abs(stress) - resistance, 0.0)
        effective = math.copysign(magnitude, stress) if stress != 0.0 else 0.0
        velocity = mechanisms[family].glide_velocity_m_s(
            effective, temperature_K, float(lengths[family])
        )
        velocities[family] = velocity
        velocities[4 + family] = -velocity
        if magnitude > 0.0:
            increment = max(1.0e-6 * abs(effective), 1.0)
            upper = mechanisms[family].glide_velocity_m_s(
                effective + increment, temperature_K, float(lengths[family])
            )
            lower = mechanisms[family].glide_velocity_m_s(
                effective - increment, temperature_K, float(lengths[family])
            )
            dv_dstress = (upper - lower) / (2.0 * increment)
            derivative = -math.copysign(1.0, stress) * dv_dstress * coefficients[family]
            derivatives[family] = derivative
            derivatives[4 + family] = -derivative
    return velocities, derivatives


def reaction_transport_symbol_s_inv(
    wavevector_m_inv: np.ndarray,
    base_mobile_m2: np.ndarray,
    base_junction_m2: np.ndarray,
    network: VectorTopologyNetwork,
    temperature_K: float,
    line_tangent_2d: np.ndarray,
    mobile_velocity_m_s: np.ndarray,
    velocity_derivative_m3_s: np.ndarray,
    mobile_diffusivity_m2_s: np.ndarray,
) -> np.ndarray:
    """Return the complete isothermal aligned reaction--transport Jacobian."""

    k = np.asarray(wavevector_m_inv, dtype=float)
    mobile = np.asarray(base_mobile_m2, dtype=float)
    junction = np.asarray(base_junction_m2, dtype=float)
    tangent = np.asarray(line_tangent_2d, dtype=float)
    velocity = np.asarray(mobile_velocity_m_s, dtype=float)
    derivative = np.asarray(velocity_derivative_m3_s, dtype=float)
    diffusion = np.asarray(mobile_diffusivity_m2_s, dtype=float)
    reactions = len(network.reactions)
    if k.shape != (2,) or tangent.shape != (2,):
        raise ValueError("wavevector and line tangent must be two-vectors")
    if mobile.shape != (8,) or junction.shape != (reactions,):
        raise ValueError("base state has inconsistent species counts")
    if velocity.shape != (8,) or diffusion.shape != (8,):
        raise ValueError("mobile transport arrays must have length eight")
    if derivative.shape != (8, reactions):
        raise ValueError("velocity derivative must have shape (8,reactions)")
    species = 8 + reactions
    operator = np.zeros((species, species), dtype=complex)
    k_parallel = float(k @ tangent)
    k_squared = float(k @ k)
    operator[np.arange(8), np.arange(8)] += (
        -1j * k_parallel * velocity - k_squared * diffusion
    )
    operator[:8, 8:] += -1j * k_parallel * mobile[:, None] * derivative
    for index, reaction in enumerate(network.reactions):
        first, second = reaction.parent_species
        density = max(
            float(mobile[first] + mobile[second]),
            reaction.reference_density_m2 * 1.0e-12,
        )
        forward, reverse = reaction.rate_constants_s_inv(temperature_K, density)
        gradient = np.zeros(species)
        gradient[first] = forward * mobile[second] / reaction.reference_density_m2
        gradient[second] = forward * mobile[first] / reaction.reference_density_m2
        gradient[8 + index] = -reverse
        operator += np.outer(network.incidence[:, index], gradient)
        product = 8 + index
        operator[product, product] += (
            -1j * k_parallel * reaction.product_velocity_m_s
            - k_squared * reaction.product_diffusivity_m2_s
        )
    return operator


VECTOR_TOPOLOGY_PARAMETER_CLASSIFICATION = {
    "BCC_Burgers_families": "immutable_framework",
    "reaction_incidence": "declared_topology_hypothesis",
    "EXP_floor_transition_state": "immutable_kinetic_form",
    "reaction_free_energy_J": "physical_literature_bound",
    "forest_coefficients_Pa_m2": "physical_closure_hypothesis",
    "diffusivity_m2_s": "physical_literature_bound",
    "collective_DD_memory": "disabled_ablation_only",
}
