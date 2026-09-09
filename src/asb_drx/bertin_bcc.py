"""Bertin et al. BCC-Ta material-point Gate A reference implementation.

This module implements the published four-Burgers-vector pencil-glide model
from Acta Materialia 260 (2023) 119336.  Its Ta parameters are a verification
reference, not campaign production parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm


_I3 = np.eye(3)
_EZ = np.asarray([0.0, 0.0, 1.0])


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must have finite nonzero norm")
    return vector / norm


BURGERS_FAMILIES_CRYSTAL = np.asarray(
    [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, -1.0]],
    dtype=float,
)
BURGERS_FAMILIES_CRYSTAL /= np.linalg.norm(
    BURGERS_FAMILIES_CRYSTAL, axis=1
)[:, None]


def _all_110_normals() -> np.ndarray:
    normals: list[np.ndarray] = []
    for zero in range(3):
        nonzero = [index for index in range(3) if index != zero]
        for first in (-1.0, 1.0):
            for second in (-1.0, 1.0):
                vector = np.zeros(3)
                vector[nonzero] = (first, second)
                candidate = _unit(vector)
                if not any(np.allclose(candidate, item) for item in normals):
                    normals.append(candidate)
    return np.asarray(normals)


_NORMALS_110 = _all_110_normals()


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the deterministic minimum rotation mapping source to target."""

    source = _unit(source)
    target = _unit(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1.0e-14:
        if cosine > 0.0:
            return _I3.copy()
        probe = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(probe, source))) > 0.9:
            probe = np.asarray([0.0, 1.0, 0.0])
        axis = _unit(np.cross(source, probe))
        return rodrigues(axis, math.pi)
    axis = cross / sine
    return rodrigues(axis, math.atan2(sine, cosine))


def rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _unit(axis)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return _I3 + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (skew @ skew)


def initial_orientation(
    loading_axis_hkl: tuple[int, int, int],
    *,
    perturbation_axis_lab: tuple[float, float, float] | None = None,
    perturbation_deg: float = 0.0,
) -> np.ndarray:
    """Map a crystal loading direction onto the laboratory z axis."""

    orientation = rotation_between(np.asarray(loading_axis_hkl, dtype=float), _EZ)
    if perturbation_axis_lab is not None and perturbation_deg != 0.0:
        orientation = (
            rodrigues(np.asarray(perturbation_axis_lab), math.radians(perturbation_deg))
            @ orientation
        )
    return orientation


@dataclass(frozen=True)
class BertinBCCParameters:
    """Published BCC-Ta reference values from Bertin et al., Tables 2--3."""

    burgers_m: float = 2.86e-10
    taylor_alpha: float = 0.3
    velocity_exponent: float = 25.0
    velocity_T_m_s: float = 1.0
    velocity_AT_m_s: float = 0.1
    drag_velocity_m_s: float = 350.0
    drag_stress_Pa: float = 700.0e6
    activation_stress_Pa: float = 100.0e6
    activation_angle_deg: float = -21.0
    generation_coefficient_m_inv: float = 4.55e9
    generation_AT_slope: float = 0.1
    annihilation_coefficient: float = -1.146
    annihilation_reference_rate_s_inv: float = 7.91e9
    annihilation_reference_temperature_K: float = 7.31e-6
    inactive_relaxation_s_inv: float = 5.0e8
    activity_threshold: float = 0.01
    activity_sharpness: float = 100.0
    plastic_spin_scale: float = 1.0
    inactive_relaxation_scale: float = 1.0
    reference_axial_rate_min_s_inv: float = 1.0e6
    reference_axial_rate_max_s_inv: float = 3.0e8

    def __post_init__(self) -> None:
        positive = (
            "burgers_m",
            "taylor_alpha",
            "velocity_exponent",
            "velocity_T_m_s",
            "velocity_AT_m_s",
            "drag_velocity_m_s",
            "drag_stress_Pa",
            "activation_stress_Pa",
            "generation_coefficient_m_inv",
            "annihilation_reference_rate_s_inv",
            "annihilation_reference_temperature_K",
            "inactive_relaxation_s_inv",
            "activity_threshold",
            "activity_sharpness",
            "reference_axial_rate_min_s_inv",
            "reference_axial_rate_max_s_inv",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("generation_AT_slope", "plastic_spin_scale", "inactive_relaxation_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.reference_axial_rate_max_s_inv < self.reference_axial_rate_min_s_inv:
            raise ValueError("reference axial rate range is reversed")

    def elastic_constants_Pa(self, temperature_K: float) -> tuple[float, float, float]:
        temperatures = np.asarray([50.0, 100.0, 300.0, 600.0, 1000.0])
        if not math.isfinite(temperature_K) or not temperatures[0] <= temperature_K <= temperatures[-1]:
            raise ValueError("Bertin Ta elastic table is valid only from 50 to 1000 K")
        c11 = np.asarray([253.41, 254.92, 261.12, 264.55, 269.09]) * 1.0e9
        c12 = np.asarray([152.38, 156.13, 165.07, 171.23, 179.96]) * 1.0e9
        c44 = np.asarray([60.34, 58.59, 55.28, 53.31, 47.77]) * 1.0e9
        return tuple(
            float(np.interp(temperature_K, temperatures, values))
            for values in (c11, c12, c44)
        )


@dataclass(frozen=True)
class BertinBCCState:
    plastic_deformation_gradient: np.ndarray
    initial_orientation: np.ndarray
    densities_m2: np.ndarray
    axial_true_strain: float = 0.0
    time_s: float = 0.0
    accepted_steps: int = 0

    def __post_init__(self) -> None:
        fp = np.asarray(self.plastic_deformation_gradient, dtype=float)
        orientation = np.asarray(self.initial_orientation, dtype=float)
        densities = np.asarray(self.densities_m2, dtype=float)
        if fp.shape != (3, 3) or orientation.shape != (3, 3):
            raise ValueError("deformation gradient and orientation must be 3 by 3")
        if densities.shape != (4,):
            raise ValueError("exactly four Burgers-family densities are required")
        if not np.all(np.isfinite(fp)) or np.linalg.det(fp) <= 0.0:
            raise ValueError("plastic deformation gradient must be finite with positive determinant")
        if not np.allclose(orientation.T @ orientation, _I3, atol=2.0e-12) or np.linalg.det(orientation) <= 0.0:
            raise ValueError("initial_orientation must be a proper rotation")
        if not np.all(np.isfinite(densities)) or np.any(densities <= 0.0):
            raise ValueError("Burgers-family densities must be finite and positive")
        if not math.isfinite(self.axial_true_strain) or not math.isfinite(self.time_s):
            raise ValueError("strain and time must be finite")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time and accepted_steps must be nonnegative")
        object.__setattr__(self, "plastic_deformation_gradient", fp.copy())
        object.__setattr__(self, "initial_orientation", orientation.copy())
        object.__setattr__(self, "densities_m2", densities.copy())


@dataclass(frozen=True)
class BertinBCCResponse:
    cauchy_stress_Pa: np.ndarray
    orientation: np.ndarray
    loading_axis_crystal: np.ndarray
    shear_rates_s_inv: np.ndarray
    resolved_shear_Pa: np.ndarray
    effective_shear_Pa: np.ndarray
    mrssp_angles_rad: np.ndarray
    activity_fraction: np.ndarray
    inactive_weight: np.ndarray
    density_rates_m2_s: np.ndarray
    plastic_velocity_gradient_s_inv: np.ndarray
    plastic_spin_s_inv: np.ndarray


def total_deformation_gradient(axial_true_strain: float) -> np.ndarray:
    """Isochoric uniaxial reference path with loading along laboratory z."""

    return np.diag(
        [math.exp(-0.5 * axial_true_strain), math.exp(-0.5 * axial_true_strain), math.exp(axial_true_strain)]
    )


def _polar_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _cubic_second_piola(
    strain_reference: np.ndarray,
    initial_rotation: np.ndarray,
    constants: tuple[float, float, float],
) -> np.ndarray:
    strain = initial_rotation.T @ strain_reference @ initial_rotation
    c11, c12, c44 = constants
    stress = np.zeros((3, 3))
    for i in range(3):
        stress[i, i] = c11 * strain[i, i] + c12 * sum(
            strain[j, j] for j in range(3) if j != i
        )
    for i in range(3):
        for j in range(i + 1, 3):
            stress[i, j] = stress[j, i] = 2.0 * c44 * strain[i, j]
    return initial_rotation @ stress @ initial_rotation.T


def _nearest_110_angle(b_crystal: np.ndarray, m_crystal: np.ndarray) -> float:
    candidates = [
        normal for normal in _NORMALS_110 if abs(float(np.dot(normal, b_crystal))) < 1.0e-12
    ]
    if not candidates:
        raise RuntimeError("no {110} reference normal found")
    angles = [
        math.atan2(
            float(np.dot(b_crystal, np.cross(normal, m_crystal))),
            float(np.dot(normal, m_crystal)),
        )
        for normal in candidates
    ]
    return min(angles, key=abs)


def _safe_logistic_inactive(activity: float, threshold: float, sharpness: float) -> float:
    argument = sharpness * (activity - threshold)
    if argument >= 40.0:
        return math.exp(-argument)
    if argument <= -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(argument))


def evaluate_bertin_bcc(
    state: BertinBCCState,
    temperature_K: float,
    parameters: BertinBCCParameters,
) -> BertinBCCResponse:
    total = total_deformation_gradient(state.axial_true_strain)
    elastic = total @ np.linalg.inv(state.plastic_deformation_gradient)
    elastic_rotation = _polar_rotation(elastic)
    orientation = elastic_rotation @ state.initial_orientation
    elastic_strain = 0.5 * (elastic.T @ elastic - _I3)
    second_piola = _cubic_second_piola(
        elastic_strain, state.initial_orientation, parameters.elastic_constants_Pa(temperature_K)
    )
    cauchy = elastic @ second_piola @ elastic.T / np.linalg.det(elastic)

    total_density = float(np.sum(state.densities_m2))
    _, _, c44 = parameters.elastic_constants_Pa(temperature_K)
    critical_stress = (
        parameters.taylor_alpha * c44 * parameters.burgers_m * math.sqrt(total_density)
    )
    alpha_p = math.radians(parameters.activation_angle_deg)
    rates = np.zeros(4)
    resolved = np.zeros(4)
    effective = np.zeros(4)
    angles = np.zeros(4)
    normals_intermediate = np.zeros((4, 3))

    for index, b_crystal in enumerate(BURGERS_FAMILIES_CRYSTAL):
        b_current = orientation @ b_crystal
        pk = np.cross(cauchy @ b_current, b_current)
        pk_norm = float(np.linalg.norm(pk))
        if pk_norm <= 1.0e-20:
            continue
        m_current = _unit(np.cross(b_current, pk / pk_norm))
        m_crystal = orientation.T @ m_current
        chi = _nearest_110_angle(b_crystal, m_crystal)
        tau = float(b_current @ cauchy @ m_current)
        denominator = math.cos(chi - alpha_p)
        if denominator <= 0.0:
            raise RuntimeError("modified-Schmid activation denominator is nonpositive")
        tau_p = parameters.activation_stress_Pa / denominator
        tau_eff = max(abs(tau) - tau_p, 0.0)
        v0 = parameters.velocity_T_m_s + (3.0 / math.pi) * (
            math.pi / 6.0 - chi
        ) * (parameters.velocity_AT_m_s - parameters.velocity_T_m_s)
        power_velocity = v0 * (tau_eff / critical_stress) ** parameters.velocity_exponent
        drag_velocity = parameters.drag_velocity_m_s * (
            1.0 - math.exp(-tau_eff / parameters.drag_stress_Pa)
        )
        velocity = min(power_velocity, drag_velocity)
        rates[index] = state.densities_m2[index] * parameters.burgers_m * velocity * np.sign(tau)
        resolved[index] = tau
        effective[index] = tau_eff
        angles[index] = chi
        normals_intermediate[index] = elastic_rotation.T @ m_current

    total_rate = float(np.sum(np.abs(rates)))
    activity = np.abs(rates) / max(total_rate, np.finfo(float).tiny)
    inactive = np.asarray(
        [
            _safe_logistic_inactive(value, parameters.activity_threshold, parameters.activity_sharpness)
            for value in activity
        ]
    )
    if total_rate <= 0.0:
        k2 = 0.0
    else:
        k2 = parameters.annihilation_coefficient * math.log(
            total_rate / parameters.annihilation_reference_rate_s_inv
        ) * math.log(temperature_K / parameters.annihilation_reference_temperature_K)

    density_rates = np.zeros(4)
    lp_raw = np.zeros((3, 3))
    for index, (rate, density, chi) in enumerate(zip(rates, state.densities_m2, angles)):
        k1 = parameters.generation_coefficient_m_inv * (
            1.0 + parameters.generation_AT_slope / math.cos(chi - alpha_p)
        )
        density_rates[index] = (
            (k1 * math.sqrt(density) - k2 * density) * abs(rate)
            - parameters.inactive_relaxation_scale
            * inactive[index]
            * parameters.inactive_relaxation_s_inv
            * density
        )
        slip_direction_intermediate = state.initial_orientation @ BURGERS_FAMILIES_CRYSTAL[index]
        lp_raw += rate * np.outer(slip_direction_intermediate, normals_intermediate[index])

    symmetric = 0.5 * (lp_raw + lp_raw.T)
    spin = 0.5 * (lp_raw - lp_raw.T)
    lp = symmetric + parameters.plastic_spin_scale * spin
    return BertinBCCResponse(
        cauchy_stress_Pa=cauchy,
        orientation=orientation,
        loading_axis_crystal=orientation.T @ _EZ,
        shear_rates_s_inv=rates,
        resolved_shear_Pa=resolved,
        effective_shear_Pa=effective,
        mrssp_angles_rad=angles,
        activity_fraction=activity,
        inactive_weight=inactive,
        density_rates_m2_s=density_rates,
        plastic_velocity_gradient_s_inv=lp,
        plastic_spin_s_inv=parameters.plastic_spin_scale * spin,
    )


def _advance_density_exact(
    density: float,
    shear_rate: float,
    chi: float,
    total_shear_rate: float,
    inactive_weight: float,
    temperature_K: float,
    dt_s: float,
    parameters: BertinBCCParameters,
) -> float:
    alpha_p = math.radians(parameters.activation_angle_deg)
    k1 = parameters.generation_coefficient_m_inv * (
        1.0 + parameters.generation_AT_slope / math.cos(chi - alpha_p)
    )
    k2 = 0.0
    if total_shear_rate > 0.0:
        k2 = parameters.annihilation_coefficient * math.log(
            total_shear_rate / parameters.annihilation_reference_rate_s_inv
        ) * math.log(temperature_K / parameters.annihilation_reference_temperature_K)
    source = k1 * abs(shear_rate)
    sink = k2 * abs(shear_rate) + (
        parameters.inactive_relaxation_scale
        * inactive_weight
        * parameters.inactive_relaxation_s_inv
    )
    root = math.sqrt(density)
    if abs(sink) < 1.0e-30:
        updated_root = root + 0.5 * source * dt_s
    else:
        factor = math.exp(-0.5 * sink * dt_s)
        updated_root = root * factor + (source / sink) * (1.0 - factor)
    if not math.isfinite(updated_root) or updated_root <= 0.0:
        raise RuntimeError("density update left the positive finite domain")
    return updated_root**2


def bertin_bcc_step(
    state: BertinBCCState,
    axial_true_strain_rate_s_inv: float,
    strain_increment: float,
    temperature_K: float,
    parameters: BertinBCCParameters,
) -> tuple[BertinBCCState, BertinBCCResponse]:
    if not math.isfinite(axial_true_strain_rate_s_inv) or axial_true_strain_rate_s_inv == 0.0:
        raise ValueError("axial_true_strain_rate_s_inv must be finite and nonzero")
    if not (
        parameters.reference_axial_rate_min_s_inv
        <= abs(axial_true_strain_rate_s_inv)
        <= parameters.reference_axial_rate_max_s_inv
    ):
        raise ValueError("axial rate is outside the published Bertin reference envelope")
    if not math.isfinite(strain_increment) or strain_increment == 0.0:
        raise ValueError("strain_increment must be finite and nonzero")
    if strain_increment * axial_true_strain_rate_s_inv <= 0.0:
        raise ValueError("strain increment and rate must have the same sign")
    response = evaluate_bertin_bcc(state, temperature_K, parameters)
    dt_s = strain_increment / axial_true_strain_rate_s_inv
    total_shear_rate = float(np.sum(np.abs(response.shear_rates_s_inv)))
    densities = np.asarray(
        [
            _advance_density_exact(
                density,
                rate,
                chi,
                total_shear_rate,
                inactive,
                temperature_K,
                dt_s,
                parameters,
            )
            for density, rate, chi, inactive in zip(
                state.densities_m2,
                response.shear_rates_s_inv,
                response.mrssp_angles_rad,
                response.inactive_weight,
            )
        ]
    )
    fp = expm(response.plastic_velocity_gradient_s_inv * dt_s) @ state.plastic_deformation_gradient
    next_state = BertinBCCState(
        plastic_deformation_gradient=fp,
        initial_orientation=state.initial_orientation,
        densities_m2=densities,
        axial_true_strain=state.axial_true_strain + strain_increment,
        time_s=state.time_s + dt_s,
        accepted_steps=state.accepted_steps + 1,
    )
    return next_state, response


def advance_bertin_bcc(
    state: BertinBCCState,
    axial_true_strain_rate_s_inv: float,
    target_axial_true_strain: float,
    strain_increment_magnitude: float,
    temperature_K: float,
    parameters: BertinBCCParameters,
    *,
    sample_stride: int = 100,
) -> tuple[BertinBCCState, tuple[dict[str, object], ...]]:
    if strain_increment_magnitude <= 0.0 or not math.isfinite(strain_increment_magnitude):
        raise ValueError("strain_increment_magnitude must be finite and positive")
    direction = math.copysign(1.0, target_axial_true_strain - state.axial_true_strain)
    if direction * axial_true_strain_rate_s_inv <= 0.0:
        raise ValueError("rate must point toward target strain")
    current = state
    history: list[dict[str, object]] = []
    distance = abs(target_axial_true_strain - current.axial_true_strain)
    nominal_steps = int(round(distance / strain_increment_magnitude))
    aligned = abs(distance - nominal_steps * strain_increment_magnitude) <= (
        1.0e-11 * max(distance, strain_increment_magnitude)
    )
    if aligned:
        increments = [direction * strain_increment_magnitude] * nominal_steps
    else:
        full_steps = int(math.floor(distance / strain_increment_magnitude))
        remainder = distance - full_steps * strain_increment_magnitude
        increments = [direction * strain_increment_magnitude] * full_steps
        if remainder > 1.0e-14:
            increments.append(direction * remainder)
    for increment_index, increment in enumerate(increments):
        current, _ = bertin_bcc_step(
            current, axial_true_strain_rate_s_inv, increment, temperature_K, parameters
        )
        if current.accepted_steps % sample_stride == 0 or increment_index == len(increments) - 1:
            evaluated = evaluate_bertin_bcc(current, temperature_K, parameters)
            history.append(
                {
                    "axial_true_strain": current.axial_true_strain,
                    "time_s": current.time_s,
                    "axial_cauchy_stress_Pa": float(evaluated.cauchy_stress_Pa[2, 2]),
                    "total_density_m2": float(np.sum(current.densities_m2)),
                    "densities_m2": current.densities_m2.tolist(),
                    "loading_axis_crystal": evaluated.loading_axis_crystal.tolist(),
                    "shear_rates_s_inv": evaluated.shear_rates_s_inv.tolist(),
                    "activity_fraction": evaluated.activity_fraction.tolist(),
                    "plastic_spin_norm_s_inv": float(np.linalg.norm(evaluated.plastic_spin_s_inv)),
                }
            )
    return current, tuple(history)


def cubic_family_angle_deg(vector: np.ndarray, family: str) -> float:
    """Minimum unsigned angle to a cubic <001>, <101>, or <111> family."""

    vector = _unit(vector)
    if family not in {"001", "101", "111"}:
        raise ValueError("family must be 001, 101, or 111")
    prototypes = {
        "001": (1, 0, 0),
        "101": (1, 0, 1),
        "111": (1, 1, 1),
    }
    base = prototypes[family]
    candidates = set()
    from itertools import permutations, product

    for permuted in permutations(base):
        for signs in product((-1, 1), repeat=3):
            candidate = tuple(value * sign for value, sign in zip(permuted, signs))
            if candidate != (0, 0, 0):
                candidates.add(candidate)
    maximum = max(abs(float(np.dot(vector, _unit(np.asarray(item))))) for item in candidates)
    return math.degrees(math.acos(float(np.clip(maximum, -1.0, 1.0))))


def save_bertin_checkpoint(path: str | Path, state: BertinBCCState) -> None:
    np.savez(
        Path(path),
        schema=np.asarray("asb-drx-bertin-bcc-checkpoint/v1"),
        plastic_deformation_gradient=state.plastic_deformation_gradient,
        initial_orientation=state.initial_orientation,
        densities_m2=state.densities_m2,
        axial_true_strain=np.asarray(state.axial_true_strain),
        time_s=np.asarray(state.time_s),
        accepted_steps=np.asarray(state.accepted_steps, dtype=np.int64),
    )


def load_bertin_checkpoint(path: str | Path) -> BertinBCCState:
    with np.load(Path(path), allow_pickle=False) as archive:
        schema = str(archive["schema"])
        if schema != "asb-drx-bertin-bcc-checkpoint/v1":
            raise ValueError(f"unsupported checkpoint schema: {schema}")
        return BertinBCCState(
            plastic_deformation_gradient=archive["plastic_deformation_gradient"],
            initial_orientation=archive["initial_orientation"],
            densities_m2=archive["densities_m2"],
            axial_true_strain=float(archive["axial_true_strain"]),
            time_s=float(archive["time_s"]),
            accepted_steps=int(archive["accepted_steps"]),
        )
