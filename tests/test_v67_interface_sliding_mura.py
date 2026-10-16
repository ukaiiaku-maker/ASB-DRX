import numpy as np
import pytest

from full_model.production.interface_sliding_mura import (
    minimal_plastic_distortion_for_alpha_z,
    project_tangential_interface_burgers_to_sliding,
)
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from tests.test_v67_interface_structural_absorption import _fixture


def test_minimal_inverse_curl_reproduces_zero_mean_manufactured_nye():
    n = 16
    spacing = 1.5e-7
    x = 2*np.pi*np.arange(n)/n
    alpha = np.zeros((n, n, 3))
    alpha[..., 0] = np.sin(x)[:, None]
    alpha[..., 1] = .4*np.cos(2*x)[None, :]
    beta = minimal_plastic_distortion_for_alpha_z(alpha, spacing)
    reconstructed = nye_from_plastic_distortion(beta, spacing)
    np.testing.assert_allclose(reconstructed[..., :, 2], alpha,
                               rtol=0.0, atol=2e-9)
    np.testing.assert_allclose(reconstructed[..., :, :2], 0.0,
                               rtol=0.0, atol=2e-9)


def test_periodic_inverse_curl_rejects_nonzero_topological_mean():
    with pytest.raises(ValueError, match="zero-integral"):
        minimal_plastic_distortion_for_alpha_z(
            np.ones((8, 8, 3)), 1.5e-7)


def test_tangential_projection_closes_interface_to_sliding_nye_balance():
    state, _, spacing, _, wall, _ = _fixture("tangent")
    result = project_tangential_interface_burgers_to_sliding(
        state, spacing_m=spacing, burgers_m=wall.burgers_m,
        reaction_fraction=.4)
    assert result.feasible
    assert result.dual_nye_relative_rms < 2e-11
    assert result.maximum_carrier_deficit_m2 == 0.0
    assert result.removed_carrier_line_cell_sum_m2 > 0.0
    np.testing.assert_allclose(result.integrated_burgers_balance_m, 0.0,
                               rtol=0.0, atol=1e-20)
    before = state.interfaces[0].boundary_burgers_m1
    after = result.candidate_state.interfaces[0].boundary_burgers_m1
    np.testing.assert_allclose(
        np.sum(before-after, axis=-2),
        result.removed_interface_burgers_m1, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(result.candidate_state.supports, state.supports)
    assert result.candidate_state.grain_ids == state.grain_ids


def test_tangential_sliding_projection_has_sign_exchange_symmetry():
    state, _, spacing, _, wall, _ = _fixture("tangent")
    positive = project_tangential_interface_burgers_to_sliding(
        state, spacing_m=spacing, burgers_m=wall.burgers_m,
        reaction_fraction=.6)
    component = state.interfaces[0]
    negative_state = type(state)(
        state.grain_ids, state.supports, state.owners,
        (type(component)(
            component.component_id, component.grain_a_id, component.grain_b_id,
            component.first_passage_fraction, component.revisit_fraction,
            component.boundary_plus_m2, component.boundary_minus_m2,
            component.boundary_junction_m2, component.exposure_s,
            component.cumulative_work_J, component.cumulative_heat_J,
            -component.boundary_burgers_m1,
            component.cumulative_signed_sweep_fraction,
            component.cumulative_absolute_sweep_fraction,
            component.periodic_winding_x, component.periodic_winding_y),),
        state.ledger, state.schema)
    negative = project_tangential_interface_burgers_to_sliding(
        negative_state, spacing_m=spacing, burgers_m=wall.burgers_m,
        reaction_fraction=.6)
    np.testing.assert_allclose(
        negative.additional_plastic_distortion,
        -positive.additional_plastic_distortion, rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(
        negative.candidate_state.interfaces[0].boundary_burgers_m1,
        -positive.candidate_state.interfaces[0].boundary_burgers_m1,
        rtol=0.0, atol=1e-12)
