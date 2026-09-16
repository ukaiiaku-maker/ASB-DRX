"""Stable, error-controlled Allen--Cahn phase proposals.

The production phase fields live on the Gibbs simplex.  This module advances
all labels simultaneously, projects the variational force onto the simplex
tangent space, and uses embedded IMEX step doubling with both a spectral
diffusion stability bound and a free-energy descent check.  It changes the
time discretization only; mobility and free-energy parameters are inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy import ndimage


Array = np.ndarray


@dataclass
class PhaseProposalDiagnostics:
    requested_dt_s: float
    integrated_dt_s: float
    minimum_substep_dt_s: float
    maximum_substep_dt_s: float
    accepted_substeps: int
    rejected_substeps: int
    maximum_error_estimate: float
    maximum_abs_increment: float
    maximum_force_residual_Pa: float
    simplex_sum_error: float
    simplex_minimum: float
    simplex_maximum: float
    energy_before_J_m: float
    energy_after_J_m: float
    energy_change_J_m: float
    diffusion_cfl_limit_s: float
    topology_projected_pixel_updates: int = 0

    def to_dict(self):
        return asdict(self)


def project_simplex(fields: Array) -> Array:
    """Euclidean projection of each pixel onto ``eta_i >= 0, sum eta_i=1``."""
    values = np.asarray(fields, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("phase fields require a nonempty phase axis")
    flat = values.reshape((-1, values.shape[-1]))
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)-1.0
    indices = np.arange(1, values.shape[-1]+1, dtype=float)[None, :]
    support = ordered-cumulative/indices > 0.0
    rho = np.maximum(np.sum(support, axis=1)-1, 0)
    theta = cumulative[np.arange(flat.shape[0]), rho]/(rho+1.0)
    projected = np.maximum(flat-theta[:, None], 0.0)
    return projected.reshape(values.shape)


def _periodic_component_labels(mask: Array):
    value = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(
        value, structure=np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0))))
    parent = np.arange(count+1)

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        if left and right:
            a, b = find(int(left)), find(int(right))
            if a != b:
                parent[b] = a

    for first, last in zip(labels[0, :], labels[-1, :]):
        union(first, last)
    for first, last in zip(labels[:, 0], labels[:, -1]):
        union(first, last)
    roots = np.zeros(count+1, dtype=int)
    remap = {}
    for label in range(1, count+1):
        root = find(label)
        if root not in remap:
            remap[root] = len(remap)+1
        roots[label] = remap[root]
    return roots[labels], len(remap)


def preserve_existing_pair_components(
        before: Array, candidate: Array, *, parent_label: int,
        child_label: int):
    """Project disconnected sign births out of an existing two-phase pair.

    A translated front remains connected to the phase it advances.  A new
    zero-level island has a candidate sign component with no pixel in the
    corresponding pre-step phase.  Reverting only such components is the
    active-set projection for the no-nucleation/no-label-allocation contract.
    """
    old = np.asarray(before, dtype=float)
    trial = np.asarray(candidate, dtype=float).copy()
    old_child = old[:, :, child_label] >= old[:, :, parent_label]
    new_child = trial[:, :, child_label] >= trial[:, :, parent_label]
    reverted = 0
    for target, old_target in ((new_child, old_child),
                               (~new_child, ~old_child)):
        labels, count = _periodic_component_labels(target)
        overlap = np.bincount(
            labels.ravel(), weights=old_target.ravel(), minlength=count+1)
        orphan_ids = np.flatnonzero(overlap[1:] == 0.0)+1
        if orphan_ids.size:
            orphan = np.isin(labels, orphan_ids)
            trial[orphan, :] = old[orphan, :]
            reverted += int(np.count_nonzero(orphan))
    # A bridge can split an existing opposite-sign region without creating a
    # disconnected component of its own.  Such a pinch-off is also outside
    # the existing-boundary/no-nucleation contract.  Revert only sign-changing
    # pixels for this substep; diffuse relaxation that preserves sign remains.
    guarded_child = trial[:, :, child_label] >= trial[:, :, parent_label]
    _, guarded_child_count = _periodic_component_labels(guarded_child)
    _, old_child_count = _periodic_component_labels(old_child)
    _, guarded_parent_count = _periodic_component_labels(~guarded_child)
    _, old_parent_count = _periodic_component_labels(~old_child)
    if (guarded_child_count != old_child_count
            or guarded_parent_count != old_parent_count):
        changed = guarded_child != old_child
        reverted += int(np.count_nonzero(changed))
        trial[changed, :] = old[changed, :]
    return trial, reverted


def spectral_phase_energy(
        eta: Array, *, spacing_m: float, kappa_J_m: float,
        bulk_barrier_J_m3: float, stored_energy_density: Callable[[Array], Array] | None = None,
        extra_energy_density: Callable[[Array], Array] | None = None) -> float:
    """Evaluate the declared periodic multiphase free energy per unit depth."""
    fields = np.asarray(eta, dtype=float)
    nx, ny, _ = fields.shape
    kx = 2.0*np.pi*np.fft.fftfreq(nx, d=spacing_m)
    ky = 2.0*np.pi*np.fft.fftfreq(ny, d=spacing_m)
    k2 = kx[:, None]**2+ky[None, :]**2
    # Parseval gives the domain mean directly.
    gradient_mean = 0.0
    for phase in range(fields.shape[2]):
        spectrum = np.fft.fft2(fields[:, :, phase])
        gradient_mean += np.sum(k2*np.abs(spectrum)**2)/(nx*ny)**2
    pair_sum = 0.5*((np.sum(fields**2, axis=2))**2
                    -np.sum(fields**4, axis=2))
    density_mean = 0.5*kappa_J_m*float(np.real(gradient_mean))
    density_mean += bulk_barrier_J_m3*float(np.mean(pair_sum))
    if stored_energy_density is not None:
        density_mean += float(np.mean(stored_energy_density(fields)))
    if extra_energy_density is not None:
        density_mean += float(np.mean(extra_energy_density(fields)))
    return density_mean*(nx*spacing_m)*(ny*spacing_m)


def _constrained_rate(force: Array, mobility, active_mask: Array | None) -> Array:
    driving = np.asarray(force, dtype=float)
    tangent = driving-np.mean(driving, axis=2, keepdims=True)
    if np.ndim(mobility) == 2:
        rate = -np.asarray(mobility, dtype=float)[:, :, None]*tangent
    else:
        rate = -np.asarray(mobility, dtype=float)*tangent
    if active_mask is not None:
        rate = np.where(np.asarray(active_mask, dtype=bool)[:, :, None], rate, 0.0)
    return rate


def adaptive_phase_proposal(
        eta: Array, *, dt_s: float, spacing_m: float, mobility,
        kappa_J_m: float, force: Callable[[Array], Array],
        energy: Callable[[Array], float], active_mask: Array | None = None,
        local_force: Callable[[Array], Array] | None = None,
        admissibility_projector: Callable[[Array, Array], tuple[Array, int]] | None = None,
        absolute_tolerance: float = 2.0e-6,
        relative_tolerance: float = 2.0e-4,
        maximum_abs_substep: float = 5.0e-3,
        diffusion_safety: float = 0.80, maximum_substeps: int = 8192,
        minimum_dt_fraction: float = 2.0**-24,
        energy_relative_tolerance: float = 2.0e-10):
    """Advance one macro step with embedded error and energy control.

    The spectral CFL limit is based on the largest resolved periodic wave
    number.  With ``local_force`` supplied, diffusion is backward Euler in
    Fourier space and one full step is compared with two half steps.  Accepted
    steps satisfy the embedded local-error target, the declared per-substep
    increment bound, simplex bounds, and nonincrease of the supplied
    frozen-state free energy.
    """
    initial = np.asarray(eta, dtype=float)
    if initial.ndim != 3 or not np.all(np.isfinite(initial)):
        raise ValueError("eta must be a finite (nx, ny, nphase) array")
    if dt_s < 0.0 or spacing_m <= 0.0 or kappa_J_m < 0.0:
        raise ValueError("invalid phase proposal scales")
    if dt_s == 0.0:
        value = project_simplex(initial)
        e0 = float(energy(value))
        diag = PhaseProposalDiagnostics(
            0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0,
            float(np.max(np.abs(force(value)))),
            float(np.max(np.abs(np.sum(value, axis=2)-1.0))),
            float(np.min(value)), float(np.max(value)), e0, e0, 0.0, np.inf)
        return value, diag

    mobility_array = np.asarray(mobility, dtype=float)
    maximum_mobility = float(np.max(np.abs(mobility_array)))
    if not np.isfinite(maximum_mobility):
        raise ValueError("mobility must be finite")
    implicit_diffusion = local_force is not None
    if implicit_diffusion and np.ndim(mobility_array) != 0:
        raise ValueError("implicit spectral diffusion requires scalar mobility")
    if maximum_mobility*kappa_J_m > 0.0:
        # |k|max^2=2(pi/dx)^2 and RK2's real-axis stability interval is 2.
        diffusion_limit = (diffusion_safety*spacing_m**2
                           /(np.pi**2*maximum_mobility*kappa_J_m))
    else:
        diffusion_limit = dt_s

    value = project_simplex(initial)
    start = value.copy()
    elapsed = 0.0
    h = dt_s if implicit_diffusion else min(dt_s, diffusion_limit)
    minimum_h = max(dt_s*minimum_dt_fraction, np.finfo(float).tiny)
    accepted = rejected = 0
    min_used = np.inf
    max_used = 0.0
    max_error = 0.0
    max_residual = 0.0
    projected_pixels = 0
    energy_before = float(energy(value))
    current_energy = energy_before

    while elapsed < dt_s*(1.0-8.0*np.finfo(float).eps):
        if accepted+rejected >= maximum_substeps:
            raise RuntimeError("phase proposal exceeded maximum_substeps")
        h = min(h, dt_s-elapsed)
        if not implicit_diffusion:
            h = min(h, diffusion_limit)
        first_force = np.asarray(force(value), dtype=float)
        first_rate = _constrained_rate(
            np.asarray(local_force(value), dtype=float)
            if implicit_diffusion else first_force,
            mobility, active_mask)
        rate_max = float(np.max(np.abs(first_rate)))
        if rate_max > 0.0:
            h = min(h, maximum_abs_substep/rate_max)
        def _implicit_diffuse(_value, _h):
            if not implicit_diffusion or kappa_J_m == 0.0:
                return project_simplex(_value)
            nx, ny, phases = _value.shape
            kx = 2.0*np.pi*np.fft.fftfreq(nx, d=spacing_m)
            ky = 2.0*np.pi*np.fft.fftfreq(ny, d=spacing_m)
            denominator = (1.0+_h*float(mobility_array)*kappa_J_m
                           *(kx[:, None]**2+ky[None, :]**2))
            result = np.empty_like(_value)
            for phase in range(phases):
                result[:, :, phase] = np.real(np.fft.ifft2(
                    np.fft.fft2(_value[:, :, phase])/denominator))
            return project_simplex(result)

        euler = _implicit_diffuse(value+h*first_rate, h)
        if active_mask is not None:
            euler = np.where(
                np.asarray(active_mask, dtype=bool)[:, :, None], euler, value)
        if admissibility_projector is not None:
            euler, projected = admissibility_projector(value, euler)
            projected_pixels += int(projected)
        if implicit_diffusion:
            half = _implicit_diffuse(value+0.5*h*first_rate, 0.5*h)
            if active_mask is not None:
                half = np.where(
                    np.asarray(active_mask, dtype=bool)[:, :, None], half, value)
            if admissibility_projector is not None:
                half, projected = admissibility_projector(value, half)
                projected_pixels += int(projected)
            half_rate = _constrained_rate(
                np.asarray(local_force(half), dtype=float), mobility, active_mask)
            candidate = _implicit_diffuse(half+0.5*h*half_rate, 0.5*h)
        else:
            second_force = np.asarray(force(euler), dtype=float)
            second_rate = _constrained_rate(second_force, mobility, active_mask)
            candidate = project_simplex(value+0.5*h*(first_rate+second_rate))
        if active_mask is not None:
            candidate = np.where(
                np.asarray(active_mask, dtype=bool)[:, :, None], candidate, value)
        if admissibility_projector is not None:
            candidate, projected = admissibility_projector(value, candidate)
            projected_pixels += int(projected)
        error = float(np.max(np.abs(candidate-euler)))
        scale = absolute_tolerance+relative_tolerance*max(
            float(np.max(np.abs(value))), float(np.max(np.abs(candidate))))
        increment = float(np.max(np.abs(candidate-value)))
        candidate_energy = float(energy(candidate))
        energy_allowance = energy_relative_tolerance*max(abs(current_energy), 1e-300)
        valid = (np.all(np.isfinite(candidate)) and error <= scale
                 and increment <= maximum_abs_substep*(1.0+1e-12)
                 and candidate_energy <= current_energy+energy_allowance)
        if not valid:
            rejected += 1
            h *= 0.5
            if h < minimum_h:
                raise RuntimeError(
                    "phase proposal cannot satisfy error/energy controls")
            continue
        value = candidate
        current_energy = candidate_energy
        elapsed += h
        accepted += 1
        min_used = min(min_used, h)
        max_used = max(max_used, h)
        max_error = max(max_error, error)
        max_residual = max(max_residual, float(np.max(np.abs(first_force))))
        if error < 0.125*scale:
            h = min(2.0*h, dt_s if implicit_diffusion else diffusion_limit)

    diagnostics = PhaseProposalDiagnostics(
        requested_dt_s=float(dt_s), integrated_dt_s=float(elapsed),
        minimum_substep_dt_s=float(min_used), maximum_substep_dt_s=float(max_used),
        accepted_substeps=accepted, rejected_substeps=rejected,
        maximum_error_estimate=max_error,
        maximum_abs_increment=float(np.max(np.abs(value-start))),
        maximum_force_residual_Pa=max_residual,
        simplex_sum_error=float(np.max(np.abs(np.sum(value, axis=2)-1.0))),
        simplex_minimum=float(np.min(value)), simplex_maximum=float(np.max(value)),
        energy_before_J_m=energy_before, energy_after_J_m=current_energy,
        energy_change_J_m=current_energy-energy_before,
        diffusion_cfl_limit_s=float(diffusion_limit))
    diagnostics.topology_projected_pixel_updates = int(projected_pixels)
    return value, diagnostics


def phase_structure_diagnostics(
        before: Array, after: Array, *, spacing_m: float,
        parent_label: int = 0, child_label: int = 1) -> dict:
    """Return resolution diagnostics without modifying the phase proposal."""
    old = np.asarray(before, dtype=float)
    new = np.asarray(after, dtype=float)
    phi0 = old[:, :, child_label]-old[:, :, parent_label]
    phi1 = new[:, :, child_label]-new[:, :, parent_label]
    nx, ny = phi1.shape
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    radius = np.sqrt(kx[:, None]**2+ky[None, :]**2)
    high = radius >= (1.0/3.0)*float(np.max(radius))

    def high_fraction(field):
        power = np.abs(np.fft.fft2(field-np.mean(field)))**2
        return float(np.sum(power[high])/max(float(np.sum(power)), 1e-300))

    gx = (np.roll(phi1, -1, axis=0)-np.roll(phi1, 1, axis=0))/(2*spacing_m)
    gy = (np.roll(phi1, -1, axis=1)-np.roll(phi1, 1, axis=1))/(2*spacing_m)
    interface = np.abs(phi1) <= 0.2
    inverse_gradient = np.divide(
        2.0, np.hypot(gx, gy), out=np.full_like(phi1, np.nan),
        where=np.hypot(gx, gy) > 0.0)
    thickness = inverse_gradient[interface]
    return {
        "sign_flip_pixel_count": int(np.count_nonzero((phi0 >= 0.0) != (phi1 >= 0.0))),
        "new_child_sign_pixel_count": int(np.count_nonzero((phi0 < 0.0) & (phi1 >= 0.0))),
        "new_parent_sign_pixel_count": int(np.count_nonzero((phi0 >= 0.0) & (phi1 < 0.0))),
        "pure_core_pixel_count": int(np.count_nonzero(np.abs(phi1) >= 0.9)),
        "minimum_phase": float(np.min(new)),
        "maximum_phase": float(np.max(new)),
        "high_k_fraction_before": high_fraction(phi0),
        "high_k_fraction_after": high_fraction(phi1),
        "profile_thickness_median_m": (
            float(np.nanmedian(thickness)) if thickness.size else None),
        "profile_thickness_median_cells": (
            float(np.nanmedian(thickness)/spacing_m) if thickness.size else None),
    }
