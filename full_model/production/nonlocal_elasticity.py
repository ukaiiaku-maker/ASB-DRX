"""Periodic 2-D isotropic elastic solve used by full-model mechanics.

Plastic distortion is work-conjugate through its symmetric in-plane part.
The zero Fourier mode is prescribed by ``mean_strain`` and all nonzero modes
are relaxed by the same Moulinec--Suquet Green operator as the recovery driver.
"""

from __future__ import annotations

import numpy as np


def isotropic_constants(c11_pa, c12_pa, c44_pa):
    mu = (float(c11_pa)-float(c12_pa)+3.0*float(c44_pa))/5.0
    lam = (float(c11_pa)+4.0*float(c12_pa)-2.0*float(c44_pa))/5.0
    return lam, mu


def stiffness_2d(c11_pa, c12_pa, c44_pa):
    lam, mu = isotropic_constants(c11_pa, c12_pa, c44_pa)
    tensor = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    tensor[i, j, k, l] = (
                        lam*(i == j)*(k == l)
                        +mu*((i == k)*(j == l)+(i == l)*(j == k)))
    return tensor


def stiffness_3d(c11_pa, c12_pa, c44_pa):
    """Isotropic 3-D stiffness using the retained Voigt shear average."""
    lam, mu = isotropic_constants(c11_pa, c12_pa, c44_pa)
    identity = np.eye(3)
    return (lam*np.einsum("ij,kl->ijkl", identity, identity)
            +mu*(np.einsum("ik,jl->ijkl", identity, identity)
                 +np.einsum("il,jk->ijkl", identity, identity)))


def green_operator(shape, spacing_m, c11_pa, c12_pa, c44_pa):
    nx, ny = map(int, shape)
    kx = 2*np.pi*np.fft.fftfreq(nx, d=float(spacing_m))
    ky = 2*np.pi*np.fft.fftfreq(ny, d=float(spacing_m))
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx*kx+ky*ky
    k2nz = k2.copy(); k2nz[0, 0] = 1.0
    lam, mu = isotropic_constants(c11_pa, c12_pa, c44_pa)
    gamma = np.zeros((2, 2, 2, 2, nx, ny))
    vectors = (kx, ky)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    ki, kj, kk, kl = vectors[i], vectors[j], vectors[k], vectors[l]
                    gamma[i, j, k, l] = (
                        .25*((i == k)*kj*kl+(j == k)*ki*kl
                             +(i == l)*kj*kk+(j == l)*ki*kk)/(mu*k2nz)
                        -(lam+mu)/(mu*(lam+2*mu))*ki*kj*kk*kl/k2nz**2)
    gamma[:, :, :, :, 0, 0] = 0.0
    return gamma


def solve_periodic_eigenstrain(eigenstrain, mean_strain, spacing_m,
                               c11_pa, c12_pa, c44_pa, iterations=3):
    inelastic = np.asarray(eigenstrain, dtype=float)
    mean = np.asarray(mean_strain, dtype=float)
    if inelastic.ndim != 4 or inelastic.shape[2:] != (2, 2) or mean.shape != (2, 2):
        raise ValueError("eigenstrain and mean-strain layouts are invalid")
    c4 = stiffness_2d(c11_pa, c12_pa, c44_pa)
    gamma = green_operator(inelastic.shape[:2], spacing_m,
                           c11_pa, c12_pa, c44_pa)
    strain = np.zeros_like(inelastic)
    for i in range(2):
        for j in range(2):
            strain[:, :, i, j] = mean[i, j]
    for _ in range(int(iterations)):
        stress = np.einsum("ijkl,...kl->...ij", c4, strain-inelastic)
        stress_hat = np.zeros_like(stress, dtype=complex)
        correction_hat = np.zeros_like(stress_hat)
        for i in range(2):
            for j in range(2):
                stress_hat[:, :, i, j] = np.fft.fft2(stress[:, :, i, j])
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        correction_hat[:, :, i, j] += (
                            gamma[i, j, k, l]*stress_hat[:, :, k, l])
        for i in range(2):
            for j in range(2):
                strain[:, :, i, j] -= np.real(np.fft.ifft2(
                    correction_hat[:, :, i, j]))
                strain[:, :, i, j] += mean[i, j]-np.mean(strain[:, :, i, j])
        strain = .5*(strain+np.swapaxes(strain, -1, -2))
    stress = np.einsum("ijkl,...kl->...ij", c4, strain-inelastic)
    return stress, strain


def solve_periodic_eigenstrain_3d_z_invariant(
        eigenstrain, mean_strain, spacing_m,
        c11_pa, c12_pa, c44_pa):
    """Equilibrate a full 3-D strain/stress field on a periodic 2-D grid.

    The fluctuation displacement has three components and depends on x and y
    only: ``partial_z u_fluctuation = 0``.  ``mean_strain`` prescribes every
    zero-mode strain component.  In particular, a zero 33 entry is a
    plane-strain constraint and does not impose zero sigma_33.
    """
    inelastic = np.asarray(eigenstrain, dtype=float)
    mean = np.asarray(mean_strain, dtype=float)
    if (inelastic.ndim != 4 or inelastic.shape[2:] != (3, 3)
            or mean.shape != (3, 3)):
        raise ValueError("full-tensor eigenstrain and mean strain require 3x3 layouts")
    if not np.all(np.isfinite(inelastic)) or not np.all(np.isfinite(mean)):
        raise ValueError("full-tensor elastic inputs must be finite")
    inelastic = .5*(inelastic+np.swapaxes(inelastic, -1, -2))
    mean = .5*(mean+mean.T)
    nx, ny = inelastic.shape[:2]
    kx = 2*np.pi*np.fft.fftfreq(nx, d=float(spacing_m))[:, None]
    ky = 2*np.pi*np.fft.fftfreq(ny, d=float(spacing_m))[None, :]
    wave = np.zeros((nx, ny, 3))
    wave[..., 0] = kx
    wave[..., 1] = ky
    k2 = np.sum(wave*wave, axis=-1)
    active = k2 > 0.0
    lam, mu = isotropic_constants(c11_pa, c12_pa, c44_pa)
    c4 = stiffness_3d(c11_pa, c12_pa, c44_pa)
    eigen_hat = np.fft.fftn(inelastic, axes=(0, 1))
    polarization_hat = np.einsum("ijkl,...kl->...ij", c4, eigen_hat)
    traction_hat = np.einsum("...j,...ij->...i", wave, polarization_hat)
    displacement_hat = np.zeros((nx, ny, 3), dtype=complex)
    # Inverse of mu*k^2 I + (lambda+mu) k tensor k.
    unit = np.zeros_like(wave)
    unit[active] = wave[active]/np.sqrt(k2[active])[:, None]
    longitudinal = (lam+mu)/(lam+2.0*mu)
    projected = traction_hat-longitudinal*unit*np.einsum(
        "...i,...i->...", unit, traction_hat)[..., None]
    displacement_hat[active] = (-1j*projected[active]
                                 /(mu*k2[active, None]))
    strain_hat = .5j*(
        np.einsum("...i,...j->...ij", wave, displacement_hat)
        +np.einsum("...j,...i->...ij", wave, displacement_hat))
    strain_hat[0, 0] = mean*float(nx*ny)
    strain = np.fft.ifftn(strain_hat, axes=(0, 1)).real
    strain += mean-np.mean(strain, axis=(0, 1))
    strain = .5*(strain+np.swapaxes(strain, -1, -2))
    stress = np.einsum("ijkl,...kl->...ij", c4, strain-inelastic)
    return stress, strain


def z_invariant_equilibrium_residual(stress, spacing_m):
    """Return div(sigma) for a z-invariant full-tensor stress field."""
    value = np.asarray(stress, dtype=float)
    if value.ndim != 4 or value.shape[2:] != (3, 3):
        raise ValueError("equilibrium residual requires a 3x3 stress field")
    nx, ny = value.shape[:2]
    kx = 2*np.pi*np.fft.fftfreq(nx, d=float(spacing_m))[:, None]
    ky = 2*np.pi*np.fft.fftfreq(ny, d=float(spacing_m))[None, :]
    result = np.zeros(value.shape[:2]+(3,))
    for component in range(3):
        sx = np.fft.fftn(value[..., component, 0], axes=(0, 1))
        sy = np.fft.fftn(value[..., component, 1], axes=(0, 1))
        result[..., component] = np.fft.ifftn(
            1j*kx*sx+1j*ky*sy, axes=(0, 1)).real
    return result


def elastic_energy_density(stress, strain, eigenstrain):
    elastic = np.asarray(strain)-np.asarray(eigenstrain)
    return .5*np.einsum("...ij,...ij->...", np.asarray(stress), elastic)
