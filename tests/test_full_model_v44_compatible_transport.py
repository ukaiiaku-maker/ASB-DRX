import numpy as np

from full_model.production.lattice_line_geometry import (
    geometry_link_nye_mimetic, geometry_link_nye_spectral_transfer,
    geometry_surface_nye_mimetic,
)
from full_model.production.mura_kinematics import (
    dealiased_product_2d, dealiased_signed_mura_products,
    family_plastic_flow_from_swept_products,
)
from full_model.production.tensorial_nye import (
    nye_from_plastic_distortion, rotated_system_fields,
)
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from tests.test_full_model_v24_mechanical_wall import mechanical_fixture
from tests.test_full_model_v43_lattice_geometry import prepared_loop


def test_three_halves_product_preserves_constant_and_removes_alias_wraparound():
    n = 24
    x = 2*np.pi*np.arange(n)/n
    constant = np.ones((n, n))*2.5
    wave = np.sin(9*x)[:, None]*np.ones((1, n))
    np.testing.assert_allclose(dealiased_product_2d(constant, wave), 2.5*wave,
                               atol=2e-14, rtol=2e-14)
    # mode 9 times mode 9 contains modes 0 and 18. Mode 18 is outside the
    # retained band and must not wrap to mode 6 as an aliased physical product.
    product = dealiased_product_2d(wave, wave)
    spectrum = np.fft.fft(product[:, 0])/n
    assert abs(spectrum[6]) < 2e-13
    assert abs(spectrum[0]-.5) < 2e-13


def test_compatible_production_transaction_closes_fluxes_and_nye():
    args = mechanical_fixture()
    state, ledger = accepted_v24_mechanical_step(
        *args, dt_s=1e-9, topology_route_enabled=False,
        mura_transport_operator="compatible_dealiased")
    transport = ledger["transport_capture"]
    assert transport["operator"] == (
        "v44_compatible_dealiased_mura_transport_and_capture")
    assert ledger["nye_suboperator_audit"]["accepted_step_hard_invariant_passed"]
    assert np.min(state.density.mobile_plus_m2) >= 0.0
    assert np.min(state.density.mobile_minus_m2) >= 0.0
    for sign in ("plus", "minus"):
        row = transport["sign"][sign]
        assert abs(row["scalar_transport_integral_residual_m"]) < 1e-12
        assert abs(row["global_scalar_residual_line_per_thickness"]) < 1e-12
        assert np.max(np.abs(
            row["global_alignment_residual_line_per_thickness"])) < 1e-12


def test_dealiased_moment_and_plastic_flow_are_one_nonlinear_product():
    args = mechanical_fixture(16)
    state, systems, spacing = args[0], args[3], args[5].spacing_m
    rng = np.random.default_rng(44)
    velocity = rng.normal(size=state.reservoir_alignment.mobile_plus_m2.shape)
    velocity = np.concatenate((velocity[..., :2],
                               np.zeros(velocity.shape[:-1]+(1,))), axis=-1)
    sp, sm, rp, rm = dealiased_signed_mura_products(
        state.reservoir_alignment.mobile_plus_m2,
        state.reservoir_alignment.mobile_minus_m2,
        velocity, -velocity, spacing)
    flow = family_plastic_flow_from_swept_products(
        sp, sm, systems, state.common.orientation_rad)
    dt = 1e-12
    alpha_increment = np.sum([
        nye_from_plastic_distortion(dt*flow[..., a, :, :], spacing)
        for a in range(len(systems))], axis=0)
    burgers, _, _ = rotated_system_fields(systems, state.common.orientation_rad)
    reservoir_increment = dt*np.einsum(
        "...ai,...aj->...ij", burgers, rp-rm)
    np.testing.assert_allclose(alpha_increment, reservoir_increment,
                               atol=2e-11, rtol=2e-11)


def test_independent_link_and_surface_reconstructions_commute():
    state, args, _ = prepared_loop(16, (7, 7))
    links = geometry_link_nye_mimetic(state.geometry)
    surface = geometry_surface_nye_mimetic(state.geometry)
    np.testing.assert_array_equal(links, surface)
    assert np.sqrt(np.mean(links**2)) > 0.0
    # This is deliberately independent of the production spectral curl. Its
    # transfer error is finite and is reported by the V44 campaign, not hidden
    # by defining both sides with the same curl.
    spectral = np.sum(state.common.family_nye_m1, axis=2)
    assert np.sqrt(np.mean((np.sum(links, axis=2)-spectral)**2)) > 0.0
    transferred = geometry_link_nye_spectral_transfer(state.geometry)
    np.testing.assert_allclose(np.sum(transferred, axis=2), spectral,
                               atol=2e-10, rtol=2e-12)
