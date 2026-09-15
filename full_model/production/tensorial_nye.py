"""Authoritative 2.5-D plastic distortion and tensorial Nye kinematics.

Fields vary in ``x`` and ``y`` but Burgers vectors, plane normals, line
alignment, plastic distortion, and Nye tensors retain three components.  The
same accepted slip increment updates ``beta_p`` and the family alignment
vectors.  The two Nye constructions therefore follow independent discrete
paths while sharing one kinematic event.

Sign convention::

    alpha_beta[i,j] = -epsilon[j,k,l] d_k beta_p[i,l]
    alpha_rho[i,j]  = sum_a b_a[i] rho1_a[j] + alpha_excess[i,j]

All derivatives are periodic spectral derivatives.  Density/alignment has
units m^-2, plastic distortion and slip are dimensionless, and Nye has m^-1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SlipSystem3D:
    burgers_vector_m: np.ndarray
    plane_normal: np.ndarray
    name: str

    def __post_init__(self):
        b = np.asarray(self.burgers_vector_m, dtype=float)
        n = np.asarray(self.plane_normal, dtype=float)
        if b.shape != (3,) or n.shape != (3,) or not self.name:
            raise ValueError("slip system requires named three-vectors")
        if np.any(~np.isfinite(b)) or np.any(~np.isfinite(n)):
            raise ValueError("slip-system vectors must be finite")
        bn, nn = np.linalg.norm(b), np.linalg.norm(n)
        if bn <= 0.0 or nn <= 0.0 or abs(float(b@n))/(bn*nn) > 2e-14:
            raise ValueError("Burgers vector and plane normal must be nonzero and orthogonal")
        object.__setattr__(self, "burgers_vector_m", b.copy())
        object.__setattr__(self, "plane_normal", n/nn)

    @property
    def burgers_m(self):
        return float(np.linalg.norm(self.burgers_vector_m))

    @property
    def slip_direction(self):
        return self.burgers_vector_m/self.burgers_m


def bcc_four_family_systems(burgers_m=2.48e-10):
    """Four declared BCC <111> families with admissible {110}/{101} planes."""
    b = float(burgers_m)
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError("Burgers magnitude must be positive")
    directions = np.asarray(((1, 1, 1), (1, -1, 1),
                             (-1, 1, 1), (1, 1, -1)), dtype=float)
    normals = np.asarray(((1, -1, 0), (1, 1, 0),
                          (1, 0, 1), (1, 0, 1)), dtype=float)
    return tuple(SlipSystem3D(
        b*direction/np.linalg.norm(direction), normal,
        f"bcc-b{index}")
        for index, (direction, normal) in enumerate(zip(directions, normals)))


def rotation_z(angle_rad):
    angle = np.asarray(angle_rad, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    result = np.zeros(angle.shape+(3, 3))
    result[..., 0, 0] = c; result[..., 0, 1] = -s
    result[..., 1, 0] = s; result[..., 1, 1] = c
    result[..., 2, 2] = 1.0
    return result


def rotated_system_fields(systems, orientation_rad):
    """Return spatial Burgers/slip and plane-normal fields in the lab frame."""
    R = rotation_z(orientation_rad)
    b0 = np.stack([system.burgers_vector_m for system in systems])
    n0 = np.stack([system.plane_normal for system in systems])
    burgers = np.einsum("...ij,aj->...ai", R, b0)
    normals = np.einsum("...ij,aj->...ai", R, n0)
    directions = burgers/np.linalg.norm(b0, axis=1)[None, None, :, None]
    return burgers, directions, normals


def spectral_derivatives(field, spacing_m):
    f = np.asarray(field, dtype=float)
    if f.ndim < 2 or spacing_m <= 0.0:
        raise ValueError("periodic field and positive spacing required")
    nx, ny = f.shape[:2]
    kx = 2*np.pi*np.fft.fftfreq(nx, d=spacing_m)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=spacing_m)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    axes = (0, 1)
    spectrum = np.fft.fftn(f, axes=axes)
    trailing = (1,)*(f.ndim-2)
    dx = np.real(np.fft.ifftn(1j*KX.reshape(KX.shape+trailing)*spectrum,
                              axes=axes))
    dy = np.real(np.fft.ifftn(1j*KY.reshape(KY.shape+trailing)*spectrum,
                              axes=axes))
    return dx, dy


def plastic_distortion_from_slip(slip, systems, orientation_rad=None):
    gamma = np.asarray(slip, dtype=float)
    if gamma.ndim != 3 or gamma.shape[2] != len(systems):
        raise ValueError("slip must have grid x family layout")
    if orientation_rad is None:
        orientation_rad = np.zeros(gamma.shape[:2])
    _, directions, normals = rotated_system_fields(systems, orientation_rad)
    return np.einsum("...a,...ai,...aj->...ij", gamma, directions, normals)


def nye_from_plastic_distortion(beta_p, spacing_m):
    beta = np.asarray(beta_p, dtype=float)
    if beta.ndim != 4 or beta.shape[2:] != (3, 3):
        raise ValueError("plastic distortion must have shape (nx,ny,3,3)")
    bx, by = spectral_derivatives(beta, spacing_m)
    alpha = np.zeros_like(beta)
    alpha[..., :, 0] = -by[..., :, 2]
    alpha[..., :, 1] = bx[..., :, 2]
    alpha[..., :, 2] = by[..., :, 0]-bx[..., :, 1]
    return alpha


def alignment_increment_from_slip(slip_increment, systems, orientation_rad,
                                  spacing_m):
    """Discrete first-order alignment increment implied by accepted slip."""
    dgamma = np.asarray(slip_increment, dtype=float)
    if dgamma.ndim != 3 or dgamma.shape[2] != len(systems):
        raise ValueError("slip increment must have grid x family layout")
    gx, gy = spectral_derivatives(dgamma, spacing_m)
    _, _, normals = rotated_system_fields(systems, orientation_rad)
    bmag = np.asarray([system.burgers_m for system in systems])
    alignment = np.empty(dgamma.shape+(3,))
    alignment[..., 0] = -normals[..., 2]*gy/bmag
    alignment[..., 1] = normals[..., 2]*gx/bmag
    alignment[..., 2] = (normals[..., 0]*gy-normals[..., 1]*gx)/bmag
    return alignment


def nye_from_alignment(alignment_m2, systems, orientation_rad,
                       boundary_excess_m1=None):
    alignment = np.asarray(alignment_m2, dtype=float)
    if alignment.ndim != 4 or alignment.shape[2:] != (len(systems), 3):
        raise ValueError("alignment must have shape (nx,ny,family,3)")
    burgers, _, _ = rotated_system_fields(systems, orientation_rad)
    alpha = np.einsum("...ai,...aj->...ij", burgers, alignment)
    if boundary_excess_m1 is not None:
        excess = np.asarray(boundary_excess_m1, dtype=float)
        if excess.shape != alpha.shape:
            raise ValueError("boundary excess must match Nye tensor")
        alpha = alpha+excess
    return alpha


def divergence_of_nye(alpha, spacing_m):
    tensor = np.asarray(alpha, dtype=float)
    if tensor.ndim != 4 or tensor.shape[2:] != (3, 3):
        raise ValueError("Nye tensor must have shape (nx,ny,3,3)")
    dx, dy = spectral_derivatives(tensor, spacing_m)
    return dx[..., :, 0]+dy[..., :, 1]


def frank_bilby_closure_from_orientations(left_angle_rad, right_angle_rad,
                                           tangent_vector):
    """Dimensionless Burgers closure per unit contour length, independently.

    The expression is evaluated from the two lattice rotations and the
    material tangent only; it does not use plastic distortion or Nye data.
    """
    tangent = np.asarray(tangent_vector, dtype=float)
    if tangent.shape != (3,):
        raise ValueError("Frank--Bilby tangent must be a three-vector")
    left = rotation_z(float(left_angle_rad))
    right = rotation_z(float(right_angle_rad))
    return (right-left)@tangent


def integrated_nye_closure(alpha_m1, normal_axis, spacing_m, line_axis=2,
                           transverse_slice=None):
    """Integrate the declared Nye line column through a boundary normal."""
    alpha = np.asarray(alpha_m1, dtype=float)
    if alpha.ndim != 4 or alpha.shape[2:] != (3, 3):
        raise ValueError("Nye tensor must have shape (nx,ny,3,3)")
    axis = int(normal_axis)
    if axis not in (0, 1) or int(line_axis) not in (0, 1, 2):
        raise ValueError("invalid spatial normal or line axis")
    if transverse_slice is None:
        transverse_slice = alpha.shape[1-axis]//2
    section = (alpha[:, transverse_slice, :, int(line_axis)] if axis == 0
               else alpha[transverse_slice, :, :, int(line_axis)])
    return np.sum(section, axis=0)*float(spacing_m)


@dataclass(frozen=True)
class TensorialKinematicState:
    slip: np.ndarray
    beta_p: np.ndarray
    alignment_m2: np.ndarray
    family_nye_m1: np.ndarray

    def validate(self, systems):
        gamma = np.asarray(self.slip, dtype=float)
        beta = np.asarray(self.beta_p, dtype=float)
        align = np.asarray(self.alignment_m2, dtype=float)
        family_nye = np.asarray(self.family_nye_m1, dtype=float)
        if gamma.ndim != 3 or gamma.shape[2] != len(systems):
            raise ValueError("state slip layout is invalid")
        if beta.shape != gamma.shape[:2]+(3, 3):
            raise ValueError("state plastic distortion layout is invalid")
        if align.shape != gamma.shape+(3,):
            raise ValueError("state alignment layout is invalid")
        if family_nye.shape != gamma.shape+(3, 3):
            raise ValueError("state family Nye layout is invalid")
        if any(np.any(~np.isfinite(a)) for a in (gamma, beta, align, family_nye)):
            raise ValueError("tensorial kinematic state must be finite")
        return gamma, beta, align, family_nye


def initialize_tensorial_state(shape, systems):
    nx, ny = map(int, shape)
    return TensorialKinematicState(
        np.zeros((nx, ny, len(systems))), np.zeros((nx, ny, 3, 3)),
        np.zeros((nx, ny, len(systems), 3)),
        np.zeros((nx, ny, len(systems), 3, 3)))


def accept_slip_increment(state, slip_increment, systems, orientation_rad,
                          spacing_m):
    gamma, beta, alignment, family_nye = state.validate(systems)
    increment = np.asarray(slip_increment, dtype=float)
    if increment.shape != gamma.shape:
        raise ValueError("accepted slip increment has wrong shape")
    dbeta = plastic_distortion_from_slip(increment, systems, orientation_rad)
    dalignment = alignment_increment_from_slip(
        increment, systems, orientation_rad, spacing_m)
    # The first alignment vector is sufficient for spatially fixed crystal
    # axes.  When orientation varies, Curl(s tensor n) also contains connection
    # terms from gradients of the rotated basis.  Retaining the family-resolved
    # Nye increment is the declared equivalent minimal alignment-tensor state;
    # it closes those terms without inventing dislocation content.
    dfamily_nye = np.stack([
        nye_from_plastic_distortion(
            plastic_distortion_from_slip(
                np.where(np.arange(len(systems))[None, None, :] == family,
                         increment, 0.0),
                systems, orientation_rad),
            spacing_m)
        for family in range(len(systems))], axis=2)
    return TensorialKinematicState(
        gamma+increment, beta+dbeta, alignment+dalignment,
        family_nye+dfamily_nye)


def consistency_metrics(state, systems, orientation_rad, spacing_m):
    gamma, beta, alignment, family_nye = state.validate(systems)
    alpha_beta = nye_from_plastic_distortion(beta, spacing_m)
    alpha_rho = np.sum(family_nye, axis=2)
    alpha_first_moment = nye_from_alignment(
        alignment, systems, orientation_rad)
    residual = alpha_beta-alpha_rho
    scale = max(float(np.sqrt(np.mean(alpha_beta*alpha_beta))), 1.0)
    divergence = divergence_of_nye(alpha_rho, spacing_m)
    div_scale = max(scale/spacing_m, 1.0)
    return {
        "alpha_beta_m1": alpha_beta,
        "alpha_rho_m1": alpha_rho,
        "alpha_first_moment_m1": alpha_first_moment,
        "connection_excess_m1": alpha_rho-alpha_first_moment,
        "residual_m1": residual,
        "relative_rms_residual": float(np.sqrt(np.mean(residual*residual))/scale),
        "divergence_m2": divergence,
        "relative_divergence_rms": float(np.sqrt(np.mean(divergence*divergence))/div_scale),
        "maximum_alignment_m2": float(np.max(np.linalg.norm(alignment, axis=-1))),
    }


@dataclass(frozen=True)
class JunctionTopology:
    parent_a: int
    parent_b: int
    sign_a: int
    sign_b: int
    product_burgers_m: np.ndarray
    parent_line_directions: np.ndarray
    product_line_direction: np.ndarray
    product_line_multiplicity: float
    character: str
    delta_free_energy_J_m: float


def make_junction_topology(systems, parent_a, parent_b, sign_a=1, sign_b=1,
                           line_a=None, line_b=None, line_tension_J_m=1.0):
    if parent_a == parent_b or sign_a not in (-1, 1) or sign_b not in (-1, 1):
        raise ValueError("junction parents/signs are invalid")
    a, b = systems[parent_a], systems[parent_b]
    ba = sign_a*a.burgers_vector_m; bb = sign_b*b.burgers_vector_m
    product = ba+bb
    if line_a is None:
        line_a = np.cross(a.plane_normal, a.slip_direction)
    if line_b is None:
        line_b = np.cross(b.plane_normal, b.slip_direction)
    parent_lines = np.stack((np.asarray(line_a, dtype=float),
                             np.asarray(line_b, dtype=float)))
    parent_lines /= np.maximum(np.linalg.norm(parent_lines, axis=1)[:, None], 1e-300)
    line = np.sum(parent_lines, axis=0)
    if np.linalg.norm(line) < 1e-14:
        line = np.cross(a.plane_normal, b.plane_normal)
    line_multiplicity = float(np.linalg.norm(line))
    line /= max(line_multiplicity, 1e-300)
    before = float(ba@ba+bb@bb)
    after = float(product@product)
    delta = float(line_tension_J_m)*(after-before)/max(
        0.5*(a.burgers_m+b.burgers_m)**2, 1e-300)
    glissile = any(
        np.linalg.norm(np.cross(product, system.burgers_vector_m))
        <= 1e-12*max(np.linalg.norm(product)*system.burgers_m, 1e-300)
        for system in systems)
    character = "glissile" if glissile else ("sessile" if after <= before else "unfavorable")
    return JunctionTopology(parent_a, parent_b, sign_a, sign_b,
                            product, parent_lines, line, line_multiplicity,
                            character, delta)


def junction_closure_metrics(topology, systems):
    parent_burgers = (
        topology.sign_a*systems[topology.parent_a].burgers_vector_m
        +topology.sign_b*systems[topology.parent_b].burgers_vector_m)
    line_before = np.sum(topology.parent_line_directions, axis=0)
    line_after = (topology.product_line_multiplicity
                  *topology.product_line_direction)
    return {
        "frank_rule_residual_m": float(np.linalg.norm(
            parent_burgers-topology.product_burgers_m)),
        "line_node_residual": float(np.linalg.norm(line_before-line_after)),
    }


def detailed_balance_rates(forward_rate_s, delta_free_energy_J_m,
                           segment_length_m, temperature_K, kB_J_K=1.380649e-23):
    """Return forward/reverse rates satisfying kf/kr=exp(-Delta F/kT)."""
    kf = float(forward_rate_s)
    if kf <= 0.0 or segment_length_m <= 0.0 or temperature_K <= 0.0:
        raise ValueError("detailed-balance inputs must be positive")
    delta = float(delta_free_energy_J_m)*float(segment_length_m)
    kr = kf*math.exp(delta/(float(kB_J_K)*float(temperature_K)))
    return kf, kr
