from dataclasses import replace

import numpy as np

from full_model.production.extensive_wall import (
    orientation_gradient_frank_bilby_target_m1,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from full_model.production.wall_orientation_relaxation import (
    evaluate_wall_orientation_relaxation,
)
from tests.test_full_model_common_tensorial_wall import fixture


def test_orientation_relaxation_uses_arrhenius_fraction_not_caller_cap():
    systems, topologies, parameters, owner, _ = fixture(8)
    spacing = parameters.spacing_m
    coordinate = np.arange(8)[:, None]
    prescribed = (.02*np.sin(2*np.pi*coordinate/8)
                  *np.ones((1, 8)))
    alpha = orientation_gradient_frank_bilby_target_m1(
        prescribed, spacing)
    burgers = np.stack([item.burgers_vector_m for item in systems])
    signed = np.einsum(
        "ai,...i->...a", np.linalg.pinv(burgers.T), alpha[..., :, 2])
    plus = np.maximum(signed, 0.0)
    minus = np.maximum(-signed, 0.0)
    zero_family = np.zeros_like(owner.mobile_plus_m2)
    owner = replace(
        owner, mobile_plus_m2=zero_family, mobile_minus_m2=zero_family,
        forest_plus_m2=zero_family, forest_minus_m2=zero_family,
        wall_plus_m2=plus, wall_minus_m2=minus,
        wall_order=np.ones(owner.wall_order.shape),
        orientation_rad=np.zeros(owner.orientation_rad.shape),
        temperature_K=np.full(owner.temperature_K.shape, 1000.0))
    state = MultiGrainCommonState(
        (1,), np.ones((1,)+owner.orientation_rad.shape), (owner,))
    parameters = replace(
        parameters, wall_nye_match_coefficient_J_m=2.0e-5)
    decision = evaluate_wall_orientation_relaxation(
        state, interval_s=2.0e-5, spacing_m=spacing,
        represented_thickness_m=5.0e-10, systems=systems,
        wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems})
    assert decision.accepted
    fraction = decision.owner_fraction_by_id[1]
    assert .05 < fraction <= 1.0
    diagnostic = decision.owner_compatibility[1]
    final_span = np.ptp(decision.committed_state.owners[0].orientation_rad)
    assert np.isclose(
        final_span, fraction*diagnostic.reconstructed_orientation_span_rad,
        rtol=2e-3)
    assert decision.delta_helmholtz_J < 0.0
    assert decision.relative_first_law_residual < 1.0e-10
    np.testing.assert_array_equal(
        decision.committed_state.owners[0].wall_plus_m2,
        state.owners[0].wall_plus_m2)
    np.testing.assert_array_equal(
        decision.committed_state.owners[0].wall_minus_m2,
        state.owners[0].wall_minus_m2)


def test_orientation_relaxation_can_preserve_structural_owner_mean():
    systems, topologies, parameters, owner, _ = fixture(8)
    spacing = parameters.spacing_m
    coordinate = np.arange(8)[:, None]
    prescribed = .02*np.sin(2*np.pi*coordinate/8)*np.ones((1, 8))
    alpha = orientation_gradient_frank_bilby_target_m1(prescribed, spacing)
    burgers = np.stack([item.burgers_vector_m for item in systems])
    signed = np.einsum(
        "ai,...i->...a", np.linalg.pinv(burgers.T), alpha[..., :, 2])
    zero_family = np.zeros_like(owner.mobile_plus_m2)
    owner = replace(
        owner, mobile_plus_m2=zero_family, mobile_minus_m2=zero_family,
        forest_plus_m2=zero_family, forest_minus_m2=zero_family,
        wall_plus_m2=np.maximum(signed, 0.0),
        wall_minus_m2=np.maximum(-signed, 0.0),
        wall_order=np.ones(owner.wall_order.shape),
        orientation_rad=np.full(owner.orientation_rad.shape, .17),
        temperature_K=np.full(owner.temperature_K.shape, 1000.0))
    state = MultiGrainCommonState(
        (1,), np.ones((1,)+owner.orientation_rad.shape), (owner,))
    parameters = replace(
        parameters, wall_nye_match_coefficient_J_m=2.0e-5)
    decision = evaluate_wall_orientation_relaxation(
        state, interval_s=2e-5, spacing_m=spacing,
        represented_thickness_m=5e-10, systems=systems,
        wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems},
        preserve_owner_mean_orientation=True)
    assert decision.accepted
    assert np.isclose(
        np.mean(decision.committed_state.owners[0].orientation_rad), .17,
        rtol=0.0, atol=1e-15)
    assert np.ptp(decision.committed_state.owners[0].orientation_rad) > 0.0
