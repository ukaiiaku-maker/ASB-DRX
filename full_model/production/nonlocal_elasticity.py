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


def elastic_energy_density(stress, strain, eigenstrain):
    elastic = np.asarray(strain)-np.asarray(eigenstrain)
    return .5*np.einsum("...ij,...ij->...", np.asarray(stress), elastic)
