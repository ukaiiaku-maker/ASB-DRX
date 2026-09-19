"""Spectral discrete complex for Mura-consistent Nye/plastic flow.

The authoritative convention is ``alpha = -Curl(beta_p)``.  A single plastic
flow tensor ``J`` therefore advances both fields through
``beta_dot = J`` and ``alpha_dot = -Curl(J)``.  No reconstruction or
post-step projection is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .tensorial_nye import (
        divergence_of_nye, nye_from_plastic_distortion,
        rotated_system_fields)
except ImportError:  # pragma: no cover - direct production-script execution
    from tensorial_nye import (
        divergence_of_nye, nye_from_plastic_distortion,
        rotated_system_fields)


@dataclass(frozen=True)
class MuraIncrement:
    plastic_flow_rate_s: np.ndarray
    nye_rate_m1_s: np.ndarray
    div_curl_residual_m2_s: np.ndarray


def plastic_flow_from_signed_alignment(plus_alignment_m2,
                                       minus_alignment_m2,
                                       velocity_plus_m_s,
                                       velocity_minus_m_s,
                                       systems, orientation_rad):
    """Return ``J=sum b tensor (v x kappa)`` from one signed face event.

    The velocity-cross-line convention is the one that gives positive
    ``s tensor n`` slip for a positive population whose declared line is
    ``n cross s``.  The negative population carries ``-b`` and is therefore
    subtracted explicitly.
    """
    kp = np.asarray(plus_alignment_m2, dtype=float)
    km = np.asarray(minus_alignment_m2, dtype=float)
    vp = np.asarray(velocity_plus_m_s, dtype=float)
    vm = np.asarray(velocity_minus_m_s, dtype=float)
    expected = np.asarray(orientation_rad).shape+(len(systems), 3)
    if (kp.shape != expected or km.shape != expected or vp.shape != expected
            or vm.shape != expected or any(np.any(~np.isfinite(x))
                                            for x in (kp, km, vp, vm))):
        raise ValueError("signed alignment and velocity fields are inconsistent")
    burgers, _, _ = rotated_system_fields(systems, orientation_rad)
    swept = np.cross(vp, kp)-np.cross(vm, km)
    return np.einsum("...ai,...al->...il", burgers, swept)


def family_plastic_flow_from_signed_alignment(
        plus_alignment_m2, minus_alignment_m2,
        velocity_plus_m_s, velocity_minus_m_s, systems, orientation_rad):
    """Family-resolved plastic flow from the same signed line event.

    Returns ``(nx, ny, family, 3, 3)``.  Summing over the family axis is
    exactly :func:`plastic_flow_from_signed_alignment`.
    """
    kp = np.asarray(plus_alignment_m2, dtype=float)
    km = np.asarray(minus_alignment_m2, dtype=float)
    vp = np.asarray(velocity_plus_m_s, dtype=float)
    vm = np.asarray(velocity_minus_m_s, dtype=float)
    expected = np.asarray(orientation_rad).shape+(len(systems), 3)
    if (kp.shape != expected or km.shape != expected or vp.shape != expected
            or vm.shape != expected or any(np.any(~np.isfinite(x))
                                            for x in (kp, km, vp, vm))):
        raise ValueError("signed alignment and velocity fields are inconsistent")
    burgers, _, _ = rotated_system_fields(systems, orientation_rad)
    swept = np.cross(vp, kp)-np.cross(vm, km)
    return np.einsum("...ai,...al->...ail", burgers, swept)


def vector_curl_2d(vector, spacing_m):
    """Spectral curl of a 3-vector field varying periodically in x and y."""
    value = np.asarray(vector, dtype=float)
    if value.ndim < 3 or value.shape[-1] != 3:
        raise ValueError("vector field must end in three components")
    dx, dy = _spectral_derivatives_local(value, spacing_m)
    result = np.empty_like(value)
    result[..., 0] = dy[..., 2]
    result[..., 1] = -dx[..., 2]
    result[..., 2] = dx[..., 1]-dy[..., 0]
    return result


def _spectral_derivatives_local(field, spacing_m):
    value = np.asarray(field, dtype=float)
    if value.ndim < 2 or spacing_m <= 0.0:
        raise ValueError("periodic field and positive spacing required")
    nx, ny = value.shape[:2]
    kx = 2*np.pi*np.fft.fftfreq(nx, d=spacing_m)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=spacing_m)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    trailing = (1,)*(value.ndim-2)
    spectrum = np.fft.fftn(value, axes=(0, 1))
    dx = np.real(np.fft.ifftn(
        1j*kx.reshape(kx.shape+trailing)*spectrum, axes=(0, 1)))
    dy = np.real(np.fft.ifftn(
        1j*ky.reshape(ky.shape+trailing)*spectrum, axes=(0, 1)))
    return dx, dy


def dealiased_product_2d(left, right):
    """Return a periodic 3/2-padded product on the original cell grid.

    Both arguments must share their first two (spatial) axes.  Trailing axes
    broadcast normally.  Padding is performed before multiplication; this is
    intentionally different from filtering an already aliased product.
    """
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape[:2] != b.shape[:2] or a.ndim != b.ndim:
        raise ValueError("dealiased factors require matching spatial layouts")
    try:
        shape = np.broadcast_shapes(a.shape, b.shape)
    except ValueError as error:
        raise ValueError("dealiased factors are not broadcast-compatible") from error
    a = np.broadcast_to(a, shape); b = np.broadcast_to(b, shape)
    nx, ny = shape[:2]
    mx = int(np.ceil(1.5*nx)); my = int(np.ceil(1.5*ny))
    # Preserve parity so the centered Fourier crop is unambiguous.
    mx += (mx-nx) % 2; my += (my-ny) % 2

    def pad(value):
        spectrum = np.fft.fftshift(np.fft.fftn(value, axes=(0, 1)), axes=(0, 1))
        padded = np.zeros((mx, my)+shape[2:], dtype=complex)
        i0 = (mx-nx)//2; j0 = (my-ny)//2
        padded[i0:i0+nx, j0:j0+ny] = spectrum
        return np.fft.ifftn(
            np.fft.ifftshift(padded, axes=(0, 1)), axes=(0, 1)).real*(mx*my)/(nx*ny)

    product = pad(a)*pad(b)
    spectrum = np.fft.fftshift(
        np.fft.fftn(product, axes=(0, 1)), axes=(0, 1))
    i0 = (mx-nx)//2; j0 = (my-ny)//2
    cropped = spectrum[i0:i0+nx, j0:j0+ny]
    return np.fft.ifftn(
        np.fft.ifftshift(cropped, axes=(0, 1)), axes=(0, 1)).real*(nx*ny)/(mx*my)


def dealiased_cross_2d(left, right):
    """Cross product whose nonlinear component products are padded first."""
    a = np.asarray(left, dtype=float); b = np.asarray(right, dtype=float)
    if a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("dealiased cross factors require identical 3-vector layouts")
    result = np.empty_like(a)
    result[..., 0] = (dealiased_product_2d(a[..., 1], b[..., 2])
                      -dealiased_product_2d(a[..., 2], b[..., 1]))
    result[..., 1] = (dealiased_product_2d(a[..., 2], b[..., 0])
                      -dealiased_product_2d(a[..., 0], b[..., 2]))
    result[..., 2] = (dealiased_product_2d(a[..., 0], b[..., 1])
                      -dealiased_product_2d(a[..., 1], b[..., 0]))
    return result


def dealiased_signed_mura_products(plus_alignment_m2, minus_alignment_m2,
                                   velocity_plus_m_s, velocity_minus_m_s,
                                   spacing_m):
    """Return filtered swept products and their exact spectral Mura rates."""
    kp = np.asarray(plus_alignment_m2, dtype=float)
    km = np.asarray(minus_alignment_m2, dtype=float)
    vp = np.asarray(velocity_plus_m_s, dtype=float)
    vm = np.asarray(velocity_minus_m_s, dtype=float)
    if kp.shape != km.shape or vp.shape != kp.shape or vm.shape != kp.shape:
        raise ValueError("alignment and velocity fields must share a layout")
    swept_plus = dealiased_cross_2d(vp, kp)
    swept_minus = dealiased_cross_2d(vm, km)
    return (swept_plus, swept_minus,
            -vector_curl_2d(swept_plus, spacing_m),
            -vector_curl_2d(swept_minus, spacing_m))


def dealiased_scalar_transport_rate(density_m2, velocity_m_s, spacing_m):
    """Conservative scalar rate paired with the declared Mura velocity.

    With this codebase's ``beta_dot=b tensor (v cross kappa)`` and
    ``alpha=-Curl(beta)``, the leading transport velocity of ``kappa`` is
    ``-v``.  The scalar flux therefore uses ``-rho*v`` as well.  Its periodic
    integral is zero to roundoff; line stretching is a separate source.
    """
    rho = np.asarray(density_m2, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    if velocity.shape != rho.shape+(3,):
        raise ValueError("scalar density and Mura velocity layouts disagree")
    flux_x = -dealiased_product_2d(rho, velocity[..., 0])
    flux_y = -dealiased_product_2d(rho, velocity[..., 1])
    dx, _ = _spectral_derivatives_local(flux_x, spacing_m)
    _, dy = _spectral_derivatives_local(flux_y, spacing_m)
    return -(dx+dy)


def family_plastic_flow_from_swept_products(swept_plus, swept_minus,
                                             systems, orientation_rad):
    """Family plastic flow from already accepted/de-aliased swept products."""
    sp = np.asarray(swept_plus, dtype=float)
    sm = np.asarray(swept_minus, dtype=float)
    expected = np.asarray(orientation_rad).shape+(len(systems), 3)
    if sp.shape != expected or sm.shape != expected:
        raise ValueError("swept products require grid x family x vector layout")
    burgers, _, _ = rotated_system_fields(systems, orientation_rad)
    return np.einsum("...ai,...al->...ail", burgers, sp-sm)


def signed_alignment_mura_rates(plus_alignment_m2, minus_alignment_m2,
                                velocity_plus_m_s, velocity_minus_m_s,
                                spacing_m):
    """Return plus/minus alignment rates generated by one Mura flux.

    For either Burgers sign, ``kappa_dot=-curl(v cross kappa)``.  The signed
    difference then generates exactly the family plastic-flow curl.
    """
    plus = np.asarray(plus_alignment_m2, dtype=float)
    minus = np.asarray(minus_alignment_m2, dtype=float)
    vp = np.asarray(velocity_plus_m_s, dtype=float)
    vm = np.asarray(velocity_minus_m_s, dtype=float)
    if plus.shape != minus.shape or vp.shape != plus.shape or vm.shape != plus.shape:
        raise ValueError("alignment and velocity fields must share a layout")
    return (-vector_curl_2d(np.cross(vp, plus), spacing_m),
            -vector_curl_2d(np.cross(vm, minus), spacing_m))


def accept_family_mura_step(beta_p, family_alpha_m1, family_flow_rate_s,
                            dt_s, spacing_m, *, relative_tolerance=2e-11):
    """Atomically accept family flow, total beta, and family-resolved Nye."""
    beta = np.asarray(beta_p, dtype=float)
    alpha = np.asarray(family_alpha_m1, dtype=float)
    flow = np.asarray(family_flow_rate_s, dtype=float)
    dt = float(dt_s)
    if (flow.ndim != 5 or flow.shape[-2:] != (3, 3)
            or alpha.shape != flow.shape or beta.shape != flow.shape[:2]+(3, 3)
            or dt < 0.0):
        raise ValueError("family Mura step fields or time increment are invalid")
    family_rate = np.stack([
        nye_from_plastic_distortion(flow[..., family, :, :], spacing_m)
        for family in range(flow.shape[2])], axis=2)
    new_beta = beta+dt*np.sum(flow, axis=2)
    new_alpha = alpha+dt*family_rate
    initial_offset = (np.sum(alpha, axis=2)
                      -nye_from_plastic_distortion(beta, spacing_m))
    curl_beta = nye_from_plastic_distortion(new_beta, spacing_m)
    alpha_total = np.sum(new_alpha, axis=2)
    # A previously declared loop/node/boundary source is a persistent offset,
    # not an error in the next source-free Mura increment.
    residual = alpha_total-curl_beta-initial_offset
    scale = max(float(np.sqrt(np.mean(curl_beta**2))), 1.0)
    relative = float(np.sqrt(np.mean(residual**2))/scale)
    if relative > float(relative_tolerance):
        raise RuntimeError("accepted family Mura step violates alpha=-Curl(beta_p)")
    div_curl = divergence_of_nye(np.sum(family_rate, axis=2), spacing_m)
    return new_beta, new_alpha, {
        "dual_nye_relative_rms": relative,
        "increment_residual_rms_m1": float(np.sqrt(np.mean((
            dt*np.sum(family_rate, axis=2)
            -nye_from_plastic_distortion(dt*np.sum(flow, axis=2), spacing_m))**2))),
        "div_curl_rms_m2_s": float(np.sqrt(np.mean(div_curl**2))),
        "cumulative_declared_source_rms_m1": float(
            np.sqrt(np.mean(initial_offset**2))),
        "accepted_step_hard_invariant_passed": True,
        "post_step_projection_used": False,
    }


def mura_increment(plastic_flow_rate_s, spacing_m):
    flow = np.asarray(plastic_flow_rate_s, dtype=float)
    if flow.ndim != 4 or flow.shape[-2:] != (3, 3):
        raise ValueError("plastic flow must have shape (nx,ny,3,3)")
    nye_rate = nye_from_plastic_distortion(flow, spacing_m)
    div_curl = divergence_of_nye(nye_rate, spacing_m)
    return MuraIncrement(flow, nye_rate, div_curl)


def accept_mura_step(beta_p, alpha_m1, plastic_flow_rate_s, dt_s, spacing_m,
                     *, relative_tolerance=2e-11):
    """Atomically accept a flow only when the dual-Nye identity remains closed."""
    beta = np.asarray(beta_p, dtype=float)
    alpha = np.asarray(alpha_m1, dtype=float)
    flow = np.asarray(plastic_flow_rate_s, dtype=float)
    dt = float(dt_s)
    if beta.shape != alpha.shape or beta.shape != flow.shape or dt < 0.0:
        raise ValueError("Mura step fields or time increment are invalid")
    operator = mura_increment(flow, spacing_m)
    new_beta = beta+dt*flow
    new_alpha = alpha+dt*operator.nye_rate_m1_s
    curl_beta = nye_from_plastic_distortion(new_beta, spacing_m)
    residual = new_alpha-curl_beta
    scale = max(float(np.sqrt(np.mean(curl_beta**2))), 1.0)
    relative = float(np.sqrt(np.mean(residual**2))/scale)
    if relative > float(relative_tolerance):
        raise RuntimeError("accepted Mura step violates alpha=-Curl(beta_p)")
    return new_beta, new_alpha, {
        "dual_nye_relative_rms": relative,
        "div_curl_rms_m2_s": float(np.sqrt(np.mean(
            operator.div_curl_residual_m2_s**2))),
        "accepted_step_hard_invariant_passed": True,
    }
