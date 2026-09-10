"""Signed Burgers-family transport and wall-pattern Gate B model.

The spatial patterning closure is a reduced, one-dimensional periodic member
of the signed-density continuum-dislocation family.  Positive and negative
line populations are primary nonnegative fields.  Their difference is the GND
density; the total density is not assigned a double-well free energy.

All default coefficients are generic verification fixtures.  They are not a
material calibration and the optional collective/DD-memory closure is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from .bertin_bcc import (
    BURGERS_FAMILIES_CRYSTAL,
    BertinBCCParameters,
    BertinBCCState,
    bertin_bcc_step,
)


_E1 = np.asarray([1.0, 0.0, 0.0])
_I3 = np.eye(3)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must be finite and nonzero")
    return vector / norm


def default_slip_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return four BCC slip directions and declared perpendicular normals."""

    directions = BURGERS_FAMILIES_CRYSTAL.copy()
    normals = np.empty_like(directions)
    probes = (np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 1.0, 0.0]))
    for index, direction in enumerate(directions):
        probe = probes[abs(float(np.dot(direction, probes[0]))) > 0.85]
        normals[index] = _unit(np.cross(direction, probe))
    return directions, normals


@dataclass(frozen=True)
class SignedTransportParameters:
    """Generic finite-wavelength signed-transport fixture in SI units."""

    domain_m: float = 16.0e-6
    burgers_m: float = 2.86e-10
    reference_density_m2: float = 4.8e14
    mobility_m2_s: float = 2.0e-12
    local_drive: float = 1.0
    nonlinear_saturation: float = 4.0
    selected_wavelength_m: float = 2.0e-6
    long_range_fraction: float = 0.10
    internal_stress_scale_Pa: float = 80.0e6
    collective_scale: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "domain_m",
            "burgers_m",
            "reference_density_m2",
            "mobility_m2_s",
            "local_drive",
            "nonlinear_saturation",
            "selected_wavelength_m",
            "internal_stress_scale_Pa",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.long_range_fraction < 0.5:
            raise ValueError("long_range_fraction must lie in [0, 0.5)")
        if self.collective_scale != 0.0:
            raise ValueError("DD collective closure is disabled in the Gate B baseline")

    @property
    def selected_wavenumber_m_inv(self) -> float:
        return 2.0 * math.pi / self.selected_wavelength_m

    @property
    def gradient_length2_m2(self) -> float:
        return self.local_drive / (2.0 * self.selected_wavenumber_m_inv**2)

    @property
    def long_range_m_inv2(self) -> float:
        k = self.selected_wavenumber_m_inv
        return self.long_range_fraction * self.local_drive * k * k


@dataclass(frozen=True)
class SignedTransportState:
    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    junction_plus_m2: np.ndarray
    junction_minus_m2: np.ndarray
    slip: np.ndarray
    plastic_distortion: np.ndarray
    orientation: np.ndarray
    time_s: float = 0.0
    accepted_steps: int = 0

    def __post_init__(self) -> None:
        plus = np.asarray(self.mobile_plus_m2, dtype=float)
        minus = np.asarray(self.mobile_minus_m2, dtype=float)
        jp = np.asarray(self.junction_plus_m2, dtype=float)
        jm = np.asarray(self.junction_minus_m2, dtype=float)
        slip = np.asarray(self.slip, dtype=float)
        beta = np.asarray(self.plastic_distortion, dtype=float)
        rotation = np.asarray(self.orientation, dtype=float)
        if plus.ndim != 2 or plus.shape[0] != 4:
            raise ValueError("signed populations must have shape (4, n)")
        if any(item.shape != plus.shape for item in (minus, jp, jm, slip)):
            raise ValueError("all family fields must share shape (4, n)")
        n = plus.shape[1]
        if beta.shape != (n, 3, 3) or rotation.shape != (n, 3, 3):
            raise ValueError("kinematic fields must have shape (n, 3, 3)")
        if any(not np.all(np.isfinite(item)) for item in (plus, minus, jp, jm, slip, beta, rotation)):
            raise ValueError("state must be finite")
        if any(np.any(item < 0.0) for item in (plus, minus, jp, jm)):
            raise ValueError("line populations must be nonnegative")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time and step count must be nonnegative")
        orthogonality = np.swapaxes(rotation, 1, 2) @ rotation
        if not np.allclose(orthogonality, _I3[None, :, :], atol=2.0e-11) or np.any(
            np.linalg.det(rotation) <= 0.0
        ):
            raise ValueError("orientation must contain proper rotations")
        for name, value in (
            ("mobile_plus_m2", plus), ("mobile_minus_m2", minus),
            ("junction_plus_m2", jp), ("junction_minus_m2", jm),
            ("slip", slip), ("plastic_distortion", beta), ("orientation", rotation),
        ):
            object.__setattr__(self, name, value.copy())

    @property
    def grid_points(self) -> int:
        return self.mobile_plus_m2.shape[1]

    @property
    def physical_grain_count(self) -> int:
        """Gate B has one crystal and has no label-allocation operation."""

        return 1


@dataclass(frozen=True)
class ReactionLedger:
    mobile_before_m_inv: float
    multiplication_added_m_inv: float
    annihilation_removed_m_inv: float
    locking_mobile_to_junction_m_inv: float
    mobile_after_m_inv: float
    junction_before_m_inv: float
    junction_after_m_inv: float
    mobile_balance_residual_m_inv: float
    junction_balance_residual_m_inv: float
    net_burgers_change_m_inv: float
    maximum_family_signed_change_m_inv: float
    burgers_vector_residual: float


def _wavenumbers(n: int, domain_m: float) -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(n, d=domain_m / n)


def spectral_derivative(field: np.ndarray, domain_m: float) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    k = _wavenumbers(field.shape[0], domain_m)
    return np.fft.ifft(1j * k * np.fft.fft(field)).real


def nye_tensor_1d(
    plastic_distortion: np.ndarray,
    domain_m: float,
    gradient_direction: np.ndarray = _E1,
) -> np.ndarray:
    """Compute alpha_ij=-epsilon_jkl n_k d(beta_il)/dx, in m^-1."""

    beta = np.asarray(plastic_distortion, dtype=float)
    if beta.ndim != 3 or beta.shape[1:] != (3, 3):
        raise ValueError("plastic_distortion must have shape (n, 3, 3)")
    nvec = _unit(gradient_direction)
    derivative = np.empty_like(beta)
    for i in range(3):
        for ell in range(3):
            derivative[:, i, ell] = spectral_derivative(beta[:, i, ell], domain_m)
    alpha = np.zeros_like(beta)
    # Cross-product matrix: (n cross row)_j = epsilon_jkl n_k row_l.
    for point in range(beta.shape[0]):
        for i in range(3):
            alpha[point, i] = -np.cross(nvec, derivative[point, i])
    return alpha


def _slip_from_signed_density(kappa_m2: np.ndarray, domain_m: float, burgers_m: float) -> np.ndarray:
    """Periodic zero-mean slip satisfying d(gamma_a)/dx=-b*kappa_a."""

    families, n = kappa_m2.shape
    k = _wavenumbers(n, domain_m)
    slip = np.zeros_like(kappa_m2)
    mask = k != 0.0
    for family in range(families):
        khat = np.fft.fft(kappa_m2[family])
        ghat = np.zeros(n, dtype=complex)
        ghat[mask] = -burgers_m * khat[mask] / (1j * k[mask])
        slip[family] = np.fft.ifft(ghat).real
    return slip


def kinematics_from_populations(
    plus_m2: np.ndarray,
    minus_m2: np.ndarray,
    domain_m: float,
    burgers_m: float,
    directions: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if directions is None or normals is None:
        directions, normals = default_slip_geometry()
    slip = _slip_from_signed_density(plus_m2 - minus_m2, domain_m, burgers_m)
    n = slip.shape[1]
    dyads = np.einsum("ai,aj->aij", directions, normals)
    beta = np.einsum("an,aij->nij", slip, dyads)
    lattice_spin = -0.5 * (beta - np.swapaxes(beta, 1, 2))
    vectors = np.stack(
        (lattice_spin[:, 2, 1], lattice_spin[:, 0, 2], lattice_spin[:, 1, 0]),
        axis=1,
    )
    angles = np.linalg.norm(vectors, axis=1)
    sine_coefficient = np.ones(n)
    cosine_coefficient = np.full(n, 0.5)
    mask = angles >= 1.0e-8
    sine_coefficient[mask] = np.sin(angles[mask]) / angles[mask]
    cosine_coefficient[mask] = (1.0 - np.cos(angles[mask])) / angles[mask] ** 2
    rotation = (
        _I3[None, :, :]
        + sine_coefficient[:, None, None] * lattice_spin
        + cosine_coefficient[:, None, None] * (lattice_spin @ lattice_spin)
    )
    return slip, beta, rotation


def initialize_signed_state(
    grid_points: int,
    parameters: SignedTransportParameters,
    *,
    perturbation: float = 0.02,
) -> SignedTransportState:
    if grid_points < 16 or grid_points % 2:
        raise ValueError("an even grid with at least 16 points is required")
    if not 0.0 <= perturbation < 0.2:
        raise ValueError("perturbation must lie in [0, 0.2)")
    x = np.arange(grid_points) * parameters.domain_m / grid_points
    q = np.zeros((4, grid_points))
    # Deterministic broadband signed seed; phases prevent family co-location.
    for family in range(4):
        for mode in range(1, 13):
            phase = 0.37 * family * mode
            q[family] += (perturbation / mode) * np.cos(
                2.0 * np.pi * mode * x / parameters.domain_m + phase
            )
    rho = parameters.reference_density_m2
    plus = 0.5 * rho * (1.0 + q)
    minus = 0.5 * rho * (1.0 - q)
    slip, beta, rotation = kinematics_from_populations(
        plus, minus, parameters.domain_m, parameters.burgers_m
    )
    zeros = np.zeros_like(plus)
    return SignedTransportState(plus, minus, zeros, zeros, slip, beta, rotation)


def signed_internal_stress_Pa(
    state: SignedTransportState, parameters: SignedTransportParameters
) -> np.ndarray:
    """GND local, gradient/backstress, and nonlocal elastic stress by family."""

    q = (state.mobile_plus_m2 - state.mobile_minus_m2) / parameters.reference_density_m2
    n = state.grid_points
    k = _wavenumbers(n, parameters.domain_m)
    result = np.zeros_like(q)
    for family in range(4):
        qhat = np.fft.fft(q[family])
        qxx = np.fft.ifft(-(k * k) * qhat).real
        psi_hat = np.zeros(n, dtype=complex)
        mask = k != 0.0
        psi_hat[mask] = qhat[mask] / (k[mask] ** 2)
        psi = np.fft.ifft(psi_hat).real
        chemical = (
            -parameters.local_drive * q[family]
            + parameters.nonlinear_saturation * q[family] ** 3
            - parameters.gradient_length2_m2 * qxx
            + parameters.long_range_m_inv2 * psi
        )
        result[family] = -parameters.internal_stress_scale_Pa * chemical
    return result


def signed_flux_m_s(
    state: SignedTransportState, parameters: SignedTransportParameters
) -> np.ndarray:
    """Normalized GND flux j_q=-M grad(mu), in m/s."""

    stress = signed_internal_stress_Pa(state, parameters)
    flux = np.empty_like(stress)
    for family in range(4):
        dimensionless_mu = -stress[family] / parameters.internal_stress_scale_Pa
        flux[family] = -parameters.mobility_m2_s * spectral_derivative(
            dimensionless_mu, parameters.domain_m
        )
    return flux


def plastic_slip_rate_from_signed_flux_s_inv(
    state: SignedTransportState, parameters: SignedTransportParameters
) -> np.ndarray:
    """Return dot(gamma)=b(J_plus-J_minus)=b*rho_ref*j_q."""

    return parameters.burgers_m * parameters.reference_density_m2 * signed_flux_m_s(
        state, parameters
    )


def pattern_step(
    state: SignedTransportState,
    dt_s: float,
    parameters: SignedTransportParameters,
) -> SignedTransportState:
    """ETD-Euler signed transport step with exact linear finite-k dynamics."""

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    rho = state.mobile_plus_m2 + state.mobile_minus_m2
    if np.max(np.abs(rho - parameters.reference_density_m2)) > 1.0e-10 * parameters.reference_density_m2:
        raise ValueError("baseline pattern step requires uniform declared total density")
    q = (state.mobile_plus_m2 - state.mobile_minus_m2) / parameters.reference_density_m2
    k = _wavenumbers(state.grid_points, parameters.domain_m)
    k2 = k * k
    linear = parameters.mobility_m2_s * (
        parameters.local_drive * k2
        - parameters.gradient_length2_m2 * k2 * k2
        - parameters.long_range_m_inv2 * (k != 0.0)
    )
    exponential = np.exp(linear * dt_s)
    phi = np.empty_like(linear)
    small = np.abs(linear) < 1.0e-30
    phi[small] = dt_s
    phi[~small] = np.expm1(linear[~small] * dt_s) / linear[~small]
    updated_q = np.empty_like(q)
    for family in range(4):
        qhat = np.fft.fft(q[family])
        nonlinear_hat = (
            -parameters.mobility_m2_s
            * parameters.nonlinear_saturation
            * k2
            * np.fft.fft(q[family] ** 3)
        )
        next_hat = exponential * qhat + phi * nonlinear_hat
        next_hat[0] = qhat[0]
        updated_q[family] = np.fft.ifft(next_hat).real
    if np.max(np.abs(updated_q)) >= 0.98:
        raise RuntimeError("signed pattern approached the population-positivity boundary")
    plus = 0.5 * parameters.reference_density_m2 * (1.0 + updated_q)
    minus = 0.5 * parameters.reference_density_m2 * (1.0 - updated_q)
    slip, beta, rotation = kinematics_from_populations(
        plus, minus, parameters.domain_m, parameters.burgers_m
    )
    return SignedTransportState(
        plus, minus, state.junction_plus_m2, state.junction_minus_m2,
        slip, beta, rotation, state.time_s + dt_s, state.accepted_steps + 1,
    )


def advance_pattern(
    state: SignedTransportState,
    duration_s: float,
    dt_s: float,
    parameters: SignedTransportParameters,
) -> SignedTransportState:
    steps = int(round(duration_s / dt_s))
    if steps <= 0 or abs(duration_s - steps * dt_s) > 1.0e-12 * duration_s:
        raise ValueError("duration must be a positive integer multiple of dt")
    current = state
    for _ in range(steps):
        current = pattern_step(current, dt_s, parameters)
    return current


def dispersion_rate_s_inv(k_m_inv: np.ndarray, parameters: SignedTransportParameters) -> np.ndarray:
    k = np.asarray(k_m_inv, dtype=float)
    rate = parameters.mobility_m2_s * (
        parameters.local_drive * k**2
        - parameters.gradient_length2_m2 * k**4
        - parameters.long_range_m_inv2
    )
    return np.where(k == 0.0, 0.0, rate)


def structure_factor_metrics(state: SignedTransportState, parameters: SignedTransportParameters) -> dict[str, float | None]:
    k = 2.0 * np.pi * np.fft.rfftfreq(state.grid_points, d=parameters.domain_m / state.grid_points)
    kappa = state.mobile_plus_m2 - state.mobile_minus_m2
    power = np.sum(np.abs(np.fft.rfft(kappa, axis=1)) ** 2, axis=0)
    power[0] = 0.0
    index = int(np.argmax(power))
    total = float(np.sum(power))
    if total == 0.0:
        return {
            "peak_wavenumber_m_inv": 0.0,
            "wall_spacing_m": None,
            "spectral_peak_fraction": 0.0,
        }
    return {
        "peak_wavenumber_m_inv": float(k[index]),
        "wall_spacing_m": float(2.0 * np.pi / k[index]),
        "spectral_peak_fraction": float(power[index] / total),
    }


def wall_diagnostics(state: SignedTransportState, parameters: SignedTransportParameters) -> dict[str, float | bool]:
    alpha = nye_tensor_1d(state.plastic_distortion, parameters.domain_m)
    gnd = np.linalg.norm(alpha, axis=(1, 2)) / parameters.burgers_m
    total = np.sum(state.mobile_plus_m2 + state.mobile_minus_m2, axis=0)
    rotation_gradient = np.empty((state.grid_points, 3, 3))
    for i in range(3):
        for j in range(3):
            rotation_gradient[:, i, j] = spectral_derivative(
                state.orientation[:, i, j], parameters.domain_m
            )
    structure = structure_factor_metrics(state, parameters)
    gnd_fraction = float(np.sqrt(np.mean(gnd**2)) / np.mean(total))
    orientation_gradient_rms_m_inv = float(np.sqrt(np.mean(rotation_gradient**2)))
    physical = bool(
        gnd_fraction > 0.01
        and orientation_gradient_rms_m_inv > 50.0
        and structure["spectral_peak_fraction"] > 0.25
    )
    if physical:
        classification = "gnd_bearing_wall"
    elif float(np.std(total) / np.mean(total)) > 0.05 and gnd_fraction <= 0.01:
        classification = "total_density_band_only"
    elif gnd_fraction > 0.01:
        classification = "diffuse_signed_structure"
    else:
        classification = "homogeneous_or_balanced_ssd"
    return {
        **structure,
        "gnd_rms_m2": float(np.sqrt(np.mean(gnd**2))),
        "gnd_to_total_ratio": gnd_fraction,
        "total_density_contrast": float(np.std(total) / np.mean(total)),
        "orientation_gradient_rms_m_inv": orientation_gradient_rms_m_inv,
        "physical_wall": physical,
        "classification": classification,
    }


def advect_periodic_upwind(
    density_m2: np.ndarray, velocity_m_s: float, dt_s: float, domain_m: float
) -> np.ndarray:
    density = np.asarray(density_m2, dtype=float)
    dx = domain_m / density.size
    cfl = velocity_m_s * dt_s / dx
    if abs(cfl) > 1.0 + 1.0e-14:
        raise ValueError("upwind CFL exceeds one")
    if cfl >= 0.0:
        updated = density - cfl * (density - np.roll(density, 1))
    else:
        updated = density - cfl * (np.roll(density, -1) - density)
    if np.any(updated < -1.0e-12 * max(float(np.max(density)), 1.0)):
        raise RuntimeError("upwind transport violated nonnegativity")
    return np.maximum(updated, 0.0)


def reaction_step(
    state: SignedTransportState,
    dt_s: float,
    parameters: SignedTransportParameters,
    *,
    multiplication_rate_m2_s: float = 0.0,
    annihilation_rate_s_inv: float = 0.0,
    locking_rate_s_inv: float = 0.0,
) -> tuple[SignedTransportState, ReactionLedger]:
    for value in (dt_s, multiplication_rate_m2_s, annihilation_rate_s_inv, locking_rate_s_inv):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("reaction step inputs must be finite and nonnegative")
    dx = parameters.domain_m / state.grid_points
    plus = state.mobile_plus_m2.copy()
    minus = state.mobile_minus_m2.copy()
    jp = state.junction_plus_m2.copy()
    jm = state.junction_minus_m2.copy()
    mobile_before = float(np.sum(plus + minus) * dx)
    junction_before = float(np.sum(jp + jm) * dx)
    added_each = 0.5 * multiplication_rate_m2_s * dt_s
    plus += added_each
    minus += added_each
    multiplication = float(2.0 * added_each * plus.size * dx)
    pair_fraction = -math.expm1(-annihilation_rate_s_inv * dt_s)
    removed_each = pair_fraction * np.minimum(plus, minus)
    plus -= removed_each
    minus -= removed_each
    annihilation = float(2.0 * np.sum(removed_each) * dx)
    lock_fraction = -math.expm1(-locking_rate_s_inv * dt_s)
    lock_plus = lock_fraction * plus
    lock_minus = lock_fraction * minus
    plus -= lock_plus
    minus -= lock_minus
    jp += lock_plus
    jm += lock_minus
    locking = float(np.sum(lock_plus + lock_minus) * dx)
    mobile_after = float(np.sum(plus + minus) * dx)
    junction_after = float(np.sum(jp + jm) * dx)
    family_signed_before = np.sum(
        state.mobile_plus_m2 - state.mobile_minus_m2
        + state.junction_plus_m2 - state.junction_minus_m2,
        axis=1,
    ) * dx
    family_signed_after = np.sum(plus - minus + jp - jm, axis=1) * dx
    family_change = family_signed_after - family_signed_before
    directions, _ = default_slip_geometry()
    burgers_residual = parameters.burgers_m * np.sum(
        family_change[:, None] * directions, axis=0
    )
    ledger = ReactionLedger(
        mobile_before, multiplication, annihilation, locking, mobile_after,
        junction_before, junction_after,
        mobile_after - (mobile_before + multiplication - annihilation - locking),
        junction_after - (junction_before + locking),
        float(np.sum(family_change)),
        float(np.max(np.abs(family_change))),
        float(np.linalg.norm(burgers_residual)),
    )
    slip, beta, rotation = kinematics_from_populations(
        plus, minus, parameters.domain_m, parameters.burgers_m
    )
    next_state = SignedTransportState(
        plus, minus, jp, jm, slip, beta, rotation,
        state.time_s + dt_s, state.accepted_steps + 1,
    )
    return next_state, ledger


def homogeneous_gate_a_step(
    states: tuple[BertinBCCState, ...],
    plus_m2: np.ndarray,
    minus_m2: np.ndarray,
    axial_rate_s_inv: float,
    strain_increment: float,
    temperature_K: float,
    parameters: BertinBCCParameters,
) -> tuple[tuple[BertinBCCState, ...], np.ndarray, np.ndarray]:
    """Exact Gate A reduction when spatial fluxes and Gate B reactions vanish."""

    plus = np.asarray(plus_m2, dtype=float)
    minus = np.asarray(minus_m2, dtype=float)
    if plus.shape != (4, len(states)) or minus.shape != plus.shape:
        raise ValueError("signed fields must have shape (4, number of cells)")
    advanced: list[BertinBCCState] = []
    next_plus = np.empty_like(plus)
    next_minus = np.empty_like(minus)
    for cell, state in enumerate(states):
        if not np.allclose(
            state.densities_m2, plus[:, cell] + minus[:, cell], rtol=2.0e-15, atol=0.0
        ):
            raise ValueError("signed populations do not match the Gate A family densities")
        updated, _ = bertin_bcc_step(
            state, axial_rate_s_inv, strain_increment, temperature_K, parameters
        )
        fraction = plus[:, cell] / state.densities_m2
        next_plus[:, cell] = fraction * updated.densities_m2
        next_minus[:, cell] = (1.0 - fraction) * updated.densities_m2
        advanced.append(updated)
    return tuple(advanced), next_plus, next_minus


def save_signed_checkpoint(path: str | Path, state: SignedTransportState) -> None:
    np.savez(
        Path(path), schema=np.asarray("asb-drx-signed-transport/v1"),
        mobile_plus_m2=state.mobile_plus_m2, mobile_minus_m2=state.mobile_minus_m2,
        junction_plus_m2=state.junction_plus_m2, junction_minus_m2=state.junction_minus_m2,
        slip=state.slip, plastic_distortion=state.plastic_distortion,
        orientation=state.orientation, time_s=np.asarray(state.time_s),
        accepted_steps=np.asarray(state.accepted_steps, dtype=np.int64),
    )


def load_signed_checkpoint(path: str | Path) -> SignedTransportState:
    with np.load(Path(path), allow_pickle=False) as archive:
        if str(archive["schema"]) != "asb-drx-signed-transport/v1":
            raise ValueError("unsupported signed-transport checkpoint schema")
        return SignedTransportState(
            archive["mobile_plus_m2"], archive["mobile_minus_m2"],
            archive["junction_plus_m2"], archive["junction_minus_m2"],
            archive["slip"], archive["plastic_distortion"], archive["orientation"],
            float(archive["time_s"]), int(archive["accepted_steps"]),
        )
