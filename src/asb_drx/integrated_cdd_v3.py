"""Integrated four-family Arrhenius mechanics and staggered signed CDD.

This is a new Mission-v3 production candidate.  The historical Bertin model
is not called: only its multiplicative-kinematics architecture is retained.
The one-dimensional periodic reduction deliberately starts without grain or
phase allocation and without a collective DD closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .arrhenius_v3 import ArrheniusMechanism
from .cdd_flux_v3 import (
    logarithmic_mean,
    variational_correlation_energy_J_m3,
    variational_correlation_fluxes,
)
from .staggered_cdd import (
    StaggeredFluxLedger,
    StaggeredSignedState,
    advance_staggered_flux_with_ledger,
    cell_centered_slip,
    face_traction_from_cells,
    work_conjugacy_residual_J_m3,
)


def _proper_polar_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


@dataclass(frozen=True)
class IntegratedCDDParameters:
    glide: ArrheniusMechanism
    shear_modulus_Pa: float
    volumetric_heat_capacity_J_m3_K: float
    line_energy_J_m: float
    slip_dyads_crystal: np.ndarray
    correlation_mobility_m_Pa_s: float
    backstress_coefficient: float
    diffusion_coefficient: float
    reference_density_m2: float
    density_floor_m2: float = 1.0e8

    def __post_init__(self) -> None:
        for name in (
            "shear_modulus_Pa", "volumetric_heat_capacity_J_m3_K",
            "line_energy_J_m", "correlation_mobility_m_Pa_s",
            "reference_density_m2", "density_floor_m2",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("backstress_coefficient", "diffusion_coefficient"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        dyads = np.asarray(self.slip_dyads_crystal, dtype=float)
        if dyads.shape != (4, 3, 3) or np.any(~np.isfinite(dyads)):
            raise ValueError("slip_dyads_crystal must have shape (4,3,3)")
        if np.max(np.abs(np.trace(dyads, axis1=1, axis2=2))) > 1.0e-12:
            raise ValueError("slip dyads must be isochoric")
        object.__setattr__(self, "slip_dyads_crystal", dyads.copy())


@dataclass(frozen=True)
class IntegratedCDDState:
    signed: StaggeredSignedState
    plastic_deformation_gradient: np.ndarray
    orientation: np.ndarray
    temperature_K: np.ndarray
    applied_shear: float = 0.0
    time_s: float = 0.0
    accepted_steps: int = 0

    def __post_init__(self) -> None:
        families, cells = self.signed.mobile_plus_m2.shape
        if families != 4:
            raise ValueError("the integrated state requires four Burgers families")
        fp = np.asarray(self.plastic_deformation_gradient, dtype=float)
        rotation = np.asarray(self.orientation, dtype=float)
        temperature = np.asarray(self.temperature_K, dtype=float)
        if fp.shape != (cells, 3, 3) or rotation.shape != fp.shape:
            raise ValueError("Fp and orientation must have shape (cells,3,3)")
        if temperature.shape != (cells,) or np.any(temperature <= 0.0):
            raise ValueError("temperature_K must be positive and cellwise")
        if np.any(~np.isfinite(fp)) or np.any(np.linalg.det(fp) <= 0.0):
            raise ValueError("Fp must be finite with positive determinant")
        if np.any(~np.isfinite(rotation)):
            raise ValueError("orientation must be finite")
        identity = np.eye(3)
        for index in range(cells):
            if not np.allclose(rotation[index].T @ rotation[index], identity, atol=2.0e-11):
                raise ValueError("orientation must be a proper rotation")
            if np.linalg.det(rotation[index]) <= 0.0:
                raise ValueError("orientation must be a proper rotation")
            expected = _proper_polar_rotation(fp[index])
            if not np.allclose(rotation[index], expected, atol=2.0e-11):
                raise ValueError("orientation must equal the polar rotation of Fp")
        if not math.isfinite(self.applied_shear) or not math.isfinite(self.time_s):
            raise ValueError("applied shear and time must be finite")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time and accepted steps must be nonnegative")
        object.__setattr__(self, "plastic_deformation_gradient", fp.copy())
        object.__setattr__(self, "orientation", rotation.copy())
        object.__setattr__(self, "temperature_K", temperature.copy())

    @property
    def physical_grain_count(self) -> int:
        return 1


@dataclass(frozen=True)
class IntegratedCDDLedger:
    external_work_J_m3: float
    elastic_energy_change_J_m3: float
    plastic_work_J_m3: float
    correlation_energy_change_J_m3: float
    stored_line_energy_change_J_m3: float
    heat_J_m3: float
    thermal_energy_change_J_m3: float
    total_energy_residual_J_m3: float
    work_projection_residual_J_m3: float
    flux: StaggeredFluxLedger


@dataclass(frozen=True)
class IntegratedCDDStep:
    state: IntegratedCDDState
    ledger: IntegratedCDDLedger
    resolved_shear_Pa: np.ndarray
    plus_face_flux_m_inv_s: np.ndarray
    minus_face_flux_m_inv_s: np.ndarray
    accepted_dt_s: float
    halvings: int


def _macro_plastic_shear(state: IntegratedCDDState, parameters: IntegratedCDDParameters) -> np.ndarray:
    # Work-conjugate scalar weights for laboratory xz antiplane shear.
    weights = parameters.slip_dyads_crystal[:, 0, 2]
    return np.sum(weights[:, None] * cell_centered_slip(state.signed.face_slip), axis=0)


def _common_stress_Pa(state: IntegratedCDDState, parameters: IntegratedCDDParameters) -> float:
    return parameters.shear_modulus_Pa * (
        state.applied_shear - float(np.mean(_macro_plastic_shear(state, parameters)))
    )


def integrated_cdd_step(
    state: IntegratedCDDState,
    applied_shear_rate_s_inv: float,
    proposed_dt_s: float,
    parameters: IntegratedCDDParameters,
    *,
    maximum_halvings: int = 40,
) -> IntegratedCDDStep:
    """Advance one energy-checked explicit coupled interval.

    The correlation subsystem is variational but currently explicit.  Step
    rejection, rather than clipping, handles its stiffness until the IMEX
    implementation is qualified.
    """

    if not math.isfinite(applied_shear_rate_s_inv):
        raise ValueError("applied_shear_rate_s_inv must be finite")
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    signed = state.signed
    old_stress = _common_stress_Pa(state, parameters)
    cells = signed.mobile_plus_m2.shape[1]
    resolved = np.empty((4, cells))
    for family in range(4):
        for cell in range(cells):
            dyad_lab = (
                state.orientation[cell] @ parameters.slip_dyads_crystal[family]
                @ state.orientation[cell].T
            )
            resolved[family, cell] = old_stress * dyad_lab[0, 2]
    resolved_face = 0.5 * (resolved + np.roll(resolved, -1, axis=1))
    temperature_face = 0.5 * (
        state.temperature_K + np.roll(state.temperature_K, -1)
    )
    plus_face = logarithmic_mean(
        signed.mobile_plus_m2 + parameters.density_floor_m2,
        np.roll(signed.mobile_plus_m2, -1, axis=1) + parameters.density_floor_m2,
    ) - parameters.density_floor_m2
    minus_face = logarithmic_mean(
        signed.mobile_minus_m2 + parameters.density_floor_m2,
        np.roll(signed.mobile_minus_m2, -1, axis=1) + parameters.density_floor_m2,
    ) - parameters.density_floor_m2
    plus_face = np.maximum(plus_face, 0.0)
    minus_face = np.maximum(minus_face, 0.0)
    physical_plus = np.zeros_like(resolved)
    physical_minus = np.zeros_like(resolved)
    for family, cell in np.ndindex(resolved.shape):
        physical_plus[family, cell] = plus_face[family, cell] * parameters.glide.net_rate_s_inv(
            float(resolved_face[family, cell]), float(temperature_face[cell])
        )
        physical_minus[family, cell] = minus_face[family, cell] * parameters.glide.net_rate_s_inv(
            float(-resolved_face[family, cell]), float(temperature_face[cell])
        )
    correlation_plus, correlation_minus = variational_correlation_fluxes(
        signed.mobile_plus_m2, signed.mobile_minus_m2,
        np.full_like(signed.mobile_plus_m2, parameters.correlation_mobility_m_Pa_s),
        np.full(cells, parameters.shear_modulus_Pa), signed.burgers_m, signed.dx_m,
        backstress_coefficient=parameters.backstress_coefficient,
        diffusion_coefficient=parameters.diffusion_coefficient,
        reference_density_m2=parameters.reference_density_m2,
        density_floor_m2=parameters.density_floor_m2,
    )
    flux_plus = physical_plus + correlation_plus
    flux_minus = physical_minus + correlation_minus
    old_corr = float(np.mean(variational_correlation_energy_J_m3(
        signed.mobile_plus_m2, signed.mobile_minus_m2,
        np.full(cells, parameters.shear_modulus_Pa), signed.burgers_m,
        backstress_coefficient=parameters.backstress_coefficient,
        diffusion_coefficient=parameters.diffusion_coefficient,
        reference_density_m2=parameters.reference_density_m2,
        density_floor_m2=parameters.density_floor_m2,
    )))
    old_line = parameters.line_energy_J_m * float(np.mean(
        signed.mobile_plus_m2 + signed.mobile_minus_m2
        + signed.locked_plus_m2 + signed.locked_minus_m2
        + signed.wall_plus_m2 + signed.wall_minus_m2
    ))
    weights = parameters.slip_dyads_crystal[:, 0, 2]
    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        try:
            advanced_signed, flux_ledger = advance_staggered_flux_with_ledger(
                signed, flux_plus, flux_minus, dt_s
            )
        except RuntimeError:
            dt_s *= 0.5
            continue
        if flux_ledger.clipping_added_m2 != 0.0:
            dt_s *= 0.5
            continue
        slip_increment_face = advanced_signed.face_slip - signed.face_slip
        slip_increment_cell = cell_centered_slip(slip_increment_face)
        macro_increment = np.sum(weights[:, None] * slip_increment_cell, axis=0)
        applied_increment = applied_shear_rate_s_inv * dt_s
        new_applied = state.applied_shear + applied_increment
        new_mean_plastic = float(np.mean(_macro_plastic_shear(
            IntegratedCDDState(
                advanced_signed, state.plastic_deformation_gradient,
                state.orientation, state.temperature_K, new_applied,
                state.time_s, state.accepted_steps,
            ), parameters
        )))
        new_stress = parameters.shear_modulus_Pa * (new_applied - new_mean_plastic)
        mean_stress = 0.5 * (old_stress + new_stress)
        external = mean_stress * applied_increment
        elastic_change = (new_stress**2 - old_stress**2) / (2.0 * parameters.shear_modulus_Pa)
        plastic_work = mean_stress * float(np.mean(macro_increment))
        new_corr = float(np.mean(variational_correlation_energy_J_m3(
            advanced_signed.mobile_plus_m2, advanced_signed.mobile_minus_m2,
            np.full(cells, parameters.shear_modulus_Pa), signed.burgers_m,
            backstress_coefficient=parameters.backstress_coefficient,
            diffusion_coefficient=parameters.diffusion_coefficient,
            reference_density_m2=parameters.reference_density_m2,
            density_floor_m2=parameters.density_floor_m2,
        )))
        new_line = parameters.line_energy_J_m * float(np.mean(
            advanced_signed.mobile_plus_m2 + advanced_signed.mobile_minus_m2
            + advanced_signed.locked_plus_m2 + advanced_signed.locked_minus_m2
            + advanced_signed.wall_plus_m2 + advanced_signed.wall_minus_m2
        ))
        correlation_change = new_corr - old_corr
        line_change = new_line - old_line
        heat = plastic_work - correlation_change - line_change
        scale = max(abs(external), abs(elastic_change), abs(plastic_work), abs(correlation_change), 1.0)
        if heat < -1.0e-12 * scale:
            dt_s *= 0.5
            continue
        heat = max(heat, 0.0)
        temperature = state.temperature_K + heat / parameters.volumetric_heat_capacity_J_m3_K
        fp = state.plastic_deformation_gradient.copy()
        orientation = state.orientation.copy()
        for cell in range(cells):
            increment_lp = np.zeros((3, 3))
            for family in range(4):
                dyad_lab = (
                    state.orientation[cell] @ parameters.slip_dyads_crystal[family]
                    @ state.orientation[cell].T
                )
                increment_lp += slip_increment_cell[family, cell] * dyad_lab
            fp[cell] = expm(increment_lp) @ fp[cell]
            orientation[cell] = _proper_polar_rotation(fp[cell])
        projection_residual = work_conjugacy_residual_J_m3(
            np.broadcast_to((mean_stress * weights)[:, None], slip_increment_face.shape),
            slip_increment_face,
        ) / cells
        thermal_change = parameters.volumetric_heat_capacity_J_m3_K * float(
            np.mean(temperature - state.temperature_K)
        )
        residual = external - elastic_change - correlation_change - line_change - thermal_change
        thermal_roundoff = (
            16.0 * np.finfo(float).eps
            * parameters.volumetric_heat_capacity_J_m3_K
            * float(np.max(temperature))
        )
        if abs(residual) > max(2.0e-11 * scale, thermal_roundoff):
            dt_s *= 0.5
            continue
        advanced = IntegratedCDDState(
            advanced_signed, fp, orientation, temperature, new_applied,
            state.time_s + dt_s, state.accepted_steps + 1,
        )
        return IntegratedCDDStep(
            advanced,
            IntegratedCDDLedger(
                external, elastic_change, plastic_work, correlation_change,
                line_change, heat, thermal_change, residual,
                projection_residual, flux_ledger,
            ),
            resolved, flux_plus, flux_minus, dt_s, halvings,
        )
    raise RuntimeError("no invariant-admissible integrated CDD step found")


INTEGRATED_PARAMETER_CLASSIFICATION = {
    "Arrhenius_form": "immutable_framework",
    "slip_dyads_crystal": "immutable_framework",
    "correlation_mobility_m_Pa_s": "generic_development_parameter",
    "backstress_coefficient": "generic_development_parameter",
    "diffusion_coefficient": "generic_development_parameter",
    "density_floor_m2": "numerical_regularization",
    "collective_DD_closure": "disabled_ablation_only",
}


CHECKPOINT_SCHEMA = "asb-drx-integrated-cdd-v3/v1"


def save_integrated_cdd_checkpoint(path: str | Path, state: IntegratedCDDState) -> None:
    """Persist every authoritative field required for an exact restart."""

    np.savez(
        Path(path), schema=np.array(CHECKPOINT_SCHEMA),
        mobile_plus_m2=state.signed.mobile_plus_m2,
        mobile_minus_m2=state.signed.mobile_minus_m2,
        locked_plus_m2=state.signed.locked_plus_m2,
        locked_minus_m2=state.signed.locked_minus_m2,
        wall_plus_m2=state.signed.wall_plus_m2,
        wall_minus_m2=state.signed.wall_minus_m2,
        face_slip=state.signed.face_slip,
        burgers_m=np.array(state.signed.burgers_m),
        dx_m=np.array(state.signed.dx_m),
        plastic_deformation_gradient=state.plastic_deformation_gradient,
        orientation=state.orientation,
        temperature_K=state.temperature_K,
        applied_shear=np.array(state.applied_shear),
        time_s=np.array(state.time_s),
        accepted_steps=np.array(state.accepted_steps, dtype=np.int64),
    )


def load_integrated_cdd_checkpoint(path: str | Path) -> IntegratedCDDState:
    with np.load(Path(path), allow_pickle=False) as archive:
        if str(archive["schema"]) != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported integrated CDD checkpoint schema")
        signed = StaggeredSignedState(
            archive["mobile_plus_m2"], archive["mobile_minus_m2"],
            archive["face_slip"], float(archive["burgers_m"]),
            float(archive["dx_m"]), archive["locked_plus_m2"],
            archive["locked_minus_m2"], archive["wall_plus_m2"],
            archive["wall_minus_m2"],
        )
        return IntegratedCDDState(
            signed, archive["plastic_deformation_gradient"],
            archive["orientation"], archive["temperature_K"],
            float(archive["applied_shear"]), float(archive["time_s"]),
            int(archive["accepted_steps"]),
        )
