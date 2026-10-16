from dataclasses import replace

import numpy as np

from full_model.production.conservative_dipole_wall_organization import (
    ConservativeDipoleWallParameters, _checkerboard_pair_separation_pass,
    _dipole_capture_domain,
)
from tests.test_full_model_common_tensorial_wall import fixture


def test_pair_separation_is_nonnegative_and_conserves_each_signed_inventory():
    _, _, _, owner, _ = fixture(8)
    owner = replace(
        owner, mobile_plus_m2=np.full_like(owner.mobile_plus_m2, 1.0e13),
        mobile_minus_m2=np.full_like(owner.mobile_minus_m2, 1.0e13),
        wall_plus_m2=np.full_like(owner.wall_plus_m2, 1.0e12),
        wall_minus_m2=np.full_like(owner.wall_minus_m2, 1.0e12))
    shape = owner.mobile_plus_m2.shape
    chemical = {
        "mobile_plus_mu_J_m": np.full(shape, 2.0e-9),
        "mobile_minus_mu_J_m": np.full(shape, 2.0e-9),
        "wall_plus_mu_J_m": np.full(shape, 1.0e-9),
        "wall_minus_mu_J_m": np.full(shape, 1.0e-9),
    }
    plus_before = np.sum(
        owner.mobile_plus_m2+owner.wall_plus_m2, dtype=np.longdouble)
    minus_before = np.sum(
        owner.mobile_minus_m2+owner.wall_minus_m2, dtype=np.longdouble)
    result, moved, _ = _checkerboard_pair_separation_pass(
        owner, chemical, np.full((8, 8), 1000.0),
        np.ones((8, 8)),
        np.ones((8, 8), dtype=bool), 2e-8, 0, 0,
        ConservativeDipoleWallParameters())
    assert moved > 0.0
    assert min(np.min(result.mobile_plus_m2), np.min(result.mobile_minus_m2),
               np.min(result.wall_plus_m2), np.min(result.wall_minus_m2)) >= 0.0
    np.testing.assert_allclose(
        np.sum(result.mobile_plus_m2+result.wall_plus_m2,
               dtype=np.longdouble), plus_before, rtol=1e-15, atol=1.0)
    np.testing.assert_allclose(
        np.sum(result.mobile_minus_m2+result.wall_minus_m2,
               dtype=np.longdouble), minus_before, rtol=1e-15, atol=1.0)


def test_pair_separation_is_inactive_when_event_is_uphill():
    _, _, _, owner, _ = fixture(8)
    shape = owner.mobile_plus_m2.shape
    chemical = {
        "mobile_plus_mu_J_m": np.full(shape, 1.0e-9),
        "mobile_minus_mu_J_m": np.full(shape, 1.0e-9),
        "wall_plus_mu_J_m": np.full(shape, 2.0e-9),
        "wall_minus_mu_J_m": np.full(shape, 2.0e-9),
    }
    result, moved, _ = _checkerboard_pair_separation_pass(
        owner, chemical, np.full((8, 8), 1000.0),
        np.ones((8, 8)),
        np.ones((8, 8), dtype=bool), 2e-8, 0, 0,
        ConservativeDipoleWallParameters())
    assert moved == 0.0
    np.testing.assert_array_equal(result.mobile_plus_m2, owner.mobile_plus_m2)
    np.testing.assert_array_equal(result.wall_minus_m2, owner.wall_minus_m2)


def test_pair_separation_conserves_support_weighted_owner_inventory():
    _, _, _, owner, _ = fixture(8)
    owner = replace(
        owner, mobile_plus_m2=np.full_like(owner.mobile_plus_m2, 1.0e13),
        mobile_minus_m2=np.full_like(owner.mobile_minus_m2, 1.0e13))
    shape = owner.mobile_plus_m2.shape
    chemical = {
        "mobile_plus_mu_J_m": np.full(shape, 2.0e-9),
        "mobile_minus_mu_J_m": np.full(shape, 2.0e-9),
        "wall_plus_mu_J_m": np.full(shape, 1.0e-9),
        "wall_minus_mu_J_m": np.full(shape, 1.0e-9),
    }
    # Even rows are the parity-zero donors.  Raw-density transfer across this
    # support jump would not conserve the common physical inventory eta*rho.
    support = np.ones((8, 8))
    support[::2] = .2
    plus_before = np.sum(
        support[..., None]
        *(owner.mobile_plus_m2+owner.wall_plus_m2), dtype=np.longdouble)
    minus_before = np.sum(
        support[..., None]
        *(owner.mobile_minus_m2+owner.wall_minus_m2), dtype=np.longdouble)
    result, moved, _ = _checkerboard_pair_separation_pass(
        owner, chemical, np.full((8, 8), 1000.0), support,
        np.ones((8, 8), dtype=bool), 2e-8, 0, 0,
        ConservativeDipoleWallParameters())
    assert moved > 0.0
    np.testing.assert_allclose(np.sum(
        support[..., None]
        *(result.mobile_plus_m2+result.wall_plus_m2), dtype=np.longdouble),
        plus_before, rtol=2e-15, atol=1.0)
    np.testing.assert_allclose(np.sum(
        support[..., None]
        *(result.mobile_minus_m2+result.wall_minus_m2), dtype=np.longdouble),
        minus_before, rtol=2e-15, atol=1.0)


def test_capture_reach_is_physical_and_includes_diffuse_boundary_donor():
    support = np.ones((8, 8))
    support[0, :] = .2
    order = np.zeros((8, 8))
    order[1, :] = 1.0
    spacing = 3.0e-7
    parameters = ConservativeDipoleWallParameters(
        wall_capture_distance_m=spacing)
    domain = _dipole_capture_domain(support, order, spacing, parameters)
    assert np.all(domain[0, :])
    assert np.all(domain[1, :])
    assert not np.any(domain[3, :])


def test_multicell_loop_span_preserves_each_signed_inventory():
    _, _, _, owner, _ = fixture(8)
    owner = replace(
        owner, mobile_plus_m2=np.full_like(owner.mobile_plus_m2, 1.0e13),
        mobile_minus_m2=np.full_like(owner.mobile_minus_m2, 1.0e13))
    shape = owner.mobile_plus_m2.shape
    chemical = {
        "mobile_plus_mu_J_m": np.full(shape, 2.0e-9),
        "mobile_minus_mu_J_m": np.full(shape, 2.0e-9),
        "wall_plus_mu_J_m": np.full(shape, 1.0e-9),
        "wall_minus_mu_J_m": np.full(shape, 1.0e-9),
    }
    plus_before = np.sum(
        owner.mobile_plus_m2+owner.wall_plus_m2, dtype=np.longdouble)
    minus_before = np.sum(
        owner.mobile_minus_m2+owner.wall_minus_m2, dtype=np.longdouble)
    result, moved, _ = _checkerboard_pair_separation_pass(
        owner, chemical, np.full((8, 8), 1000.0),
        np.ones((8, 8)),
        np.ones((8, 8), dtype=bool), 2e-8, 0, 0,
        ConservativeDipoleWallParameters(), offset_cells=2)
    assert moved > 0.0
    np.testing.assert_allclose(
        np.sum(result.mobile_plus_m2+result.wall_plus_m2,
               dtype=np.longdouble), plus_before, rtol=1e-15, atol=1.0)
    np.testing.assert_allclose(
        np.sum(result.mobile_minus_m2+result.wall_minus_m2,
               dtype=np.longdouble), minus_before, rtol=1e-15, atol=1.0)
