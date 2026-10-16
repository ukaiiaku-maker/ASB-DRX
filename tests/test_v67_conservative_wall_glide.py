from dataclasses import replace

import numpy as np

from full_model.production.conservative_wall_glide import (
    evaluate_conservative_wall_glide,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from tests.test_full_model_common_tensorial_wall import fixture


def test_local_wall_glide_is_downhill_nonnegative_and_inventory_conservative():
    systems, topologies, parameters, owner, _ = fixture(8)
    checker = (np.indices((8, 8)).sum(axis=0) % 2)[..., None]
    plus = np.broadcast_to(1.0e12+checker*1.5e14,
                           owner.wall_plus_m2.shape).copy()
    minus = np.broadcast_to(1.0e12+(1-checker)*1.2e14,
                            owner.wall_minus_m2.shape).copy()
    owner = replace(
        owner, wall_plus_m2=plus, wall_minus_m2=minus,
        wall_order=np.ones((8, 8)),
        temperature_K=np.full((8, 8), 1000.0))
    state = MultiGrainCommonState((1,), np.ones((1, 8, 8)), (owner,))
    before_plus = np.sum(plus, axis=(0, 1), dtype=np.longdouble)
    before_minus = np.sum(minus, axis=(0, 1), dtype=np.longdouble)
    decision = evaluate_conservative_wall_glide(
        state, interval_s=2.0e-5, mean_strain=np.zeros((2, 2)),
        spacing_m=parameters.spacing_m,
        represented_thickness_m=5.0e-10, systems=systems,
        wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems})
    assert decision.accepted
    assert decision.delta_helmholtz_J < 0.0
    assert decision.relative_first_law_residual < 1.0e-10
    result = decision.committed_state.owners[0]
    assert np.min(result.wall_plus_m2) >= 0.0
    assert np.min(result.wall_minus_m2) >= 0.0
    np.testing.assert_allclose(
        np.sum(result.wall_plus_m2, axis=(0, 1), dtype=np.longdouble),
        before_plus, rtol=2e-15, atol=1.0)
    np.testing.assert_allclose(
        np.sum(result.wall_minus_m2, axis=(0, 1), dtype=np.longdouble),
        before_minus, rtol=2e-15, atol=1.0)
