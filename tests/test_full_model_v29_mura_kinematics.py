import numpy as np
import pytest

from full_model.production.mura_kinematics import (
    accept_mura_step, mura_increment, plastic_flow_from_signed_alignment)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, divergence_of_nye,
    nye_from_plastic_distortion)


def _periodic_tensor(n=24, scale=1.0):
    x = np.arange(n)[:, None]*2*np.pi/n
    y = np.arange(n)[None, :]*2*np.pi/n
    value = np.zeros((n, n, 3, 3))
    value[..., 0, 0] = scale*np.sin(2*x+3*y)
    value[..., 1, 2] = scale*np.cos(x-2*y)
    value[..., 2, 1] = scale*np.sin(3*x-y)
    return value


def test_periodic_discrete_div_curl_identity_is_machine_precision():
    flow = _periodic_tensor()
    result = mura_increment(flow, 2e-9)
    curl_scale = max(np.sqrt(np.mean(result.nye_rate_m1_s**2))/2e-9, 1.0)
    assert np.sqrt(np.mean(result.div_curl_residual_m2_s**2))/curl_scale < 2e-14


def test_one_flux_updates_beta_and_nye_with_hard_identity():
    beta = _periodic_tensor(scale=2e-4)
    alpha = nye_from_plastic_distortion(beta, 3e-9)
    flow = _periodic_tensor(scale=4e2)
    new_beta, new_alpha, audit = accept_mura_step(
        beta, alpha, flow, 2e-9, 3e-9)
    np.testing.assert_allclose(
        new_alpha, nye_from_plastic_distortion(new_beta, 3e-9), rtol=2e-12, atol=2e-8)
    assert audit["accepted_step_hard_invariant_passed"]
    div = divergence_of_nye(new_alpha, 3e-9)
    assert np.sqrt(np.mean(div**2)) < 2e4


def test_signed_population_exchange_covariance_of_mura_flux():
    n = 8
    systems = bcc_four_family_systems()
    shape = (n, n, len(systems), 3)
    rng = np.random.default_rng(12)
    kp = rng.normal(size=shape)*1e12
    km = rng.normal(size=shape)*1e12
    vp = rng.normal(size=shape)*1e-4
    vm = rng.normal(size=shape)*1e-4
    angle = np.zeros((n, n))
    direct = plastic_flow_from_signed_alignment(kp, km, vp, vm, systems, angle)
    exchanged = plastic_flow_from_signed_alignment(km, kp, vm, vp, systems, angle)
    np.testing.assert_allclose(exchanged, -direct, rtol=2e-15, atol=2e-15)
