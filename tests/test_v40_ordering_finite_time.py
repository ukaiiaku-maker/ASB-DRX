from dataclasses import replace

import numpy as np

from full_model.production.extensive_wall import (
    _attempt_rate_s, accepted_ordering_step,
)
from tests.test_v39_stiff_ordering import _sparse_wall_case


def _compact_active_case():
    state, density, systems, topologies, parameters, target, stress = (
        _sparse_wall_case())
    plus = np.zeros_like(density.wall_tangle_plus_m2)
    minus = np.zeros_like(density.wall_tangle_minus_m2)
    plus[7, 7, :] = 1.2e13
    minus[7, 7, :] = 9.0e12
    density = replace(
        density, wall_tangle_plus_m2=plus, wall_tangle_minus_m2=minus)
    attempt = float(np.max(_attempt_rate_s(
        np.max(np.abs(stress), axis=2), state.common.temperature_K,
        parameters)))
    return (state, density, systems, topologies, parameters, target, stress,
            attempt)


def _step(case, parameters, duration):
    state, density, systems, topologies, _, target, stress, _ = case
    return accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, parameters, duration)


def test_asymptotic_switch_overlaps_finite_time_bdf():
    case = _compact_active_case()
    parameters = case[4]
    duration = parameters.ordering_asymptotic_minimum_attempt_exposure / case[7]
    finite = _step(case, replace(
        parameters, ordering_integration_method="finite_time_bdf"), duration)
    selected = _step(case, replace(
        parameters, ordering_integration_method="implicit_backward_euler"),
        duration)
    assert finite[1]["integration_method"] == "bounded_finite_time_bdf"
    assert selected[1]["integration_method"] == "bounded_convex_asymptotic"
    assert finite[1]["finite_time_local_error_control_passed"] is True
    assert finite[1]["finite_time_kinetic_accuracy_certified_by_this_solve"] is False
    assert selected[1]["finite_time_kinetic_accuracy_certified_by_this_solve"] is False
    assert "not_a_finite_time_kinetic_error_bound" in selected[1][
        "asymptotic_endpoint_diagnostic_semantics"]
    for name in ("wall_tangle_plus_m2", "wall_tangle_minus_m2",
                 "wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        np.testing.assert_allclose(
            getattr(selected[0], name), getattr(finite[0], name),
            rtol=2e-6, atol=1e-3)


def test_finite_time_bdf_respects_rate_time_rescaling():
    case = _compact_active_case()
    parameters = replace(case[4], ordering_integration_method="finite_time_bdf")
    duration = 20.0 / case[7]
    baseline = _step(case, parameters, duration)
    multiplier = 4.0
    scaled = _step(case, replace(
        parameters,
        ordering_attempt_frequency_s=(
            multiplier * parameters.ordering_attempt_frequency_s)),
        duration / multiplier)
    for name in ("wall_tangle_plus_m2", "wall_tangle_minus_m2",
                 "wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        np.testing.assert_allclose(
            getattr(scaled[0], name), getattr(baseline[0], name),
            rtol=2e-7, atol=1e-3)
