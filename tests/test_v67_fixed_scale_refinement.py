import numpy as np

from full_model.production.fixed_scale_refinement import (
    refine_multigrain_common_state,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from tests.test_full_model_common_tensorial_wall import fixture


def test_fixed_scale_refinement_preserves_support_and_line_inventory():
    systems, topologies, parameters, owner, _ = fixture(8)
    x, y = np.indices((8, 8))
    first = .25+.20*np.sin(2*np.pi*x/8)*np.cos(2*np.pi*y/8)
    supports = np.stack((first, 1.0-first))
    state = MultiGrainCommonState((1, 2), supports, (owner, owner))
    refined, audit = refine_multigrain_common_state(state, 2)
    assert refined.supports.shape == (2, 16, 16)
    np.testing.assert_allclose(np.sum(refined.supports, axis=0), 1.0,
                               rtol=0.0, atol=2e-16)
    assert audit["maximum_relative_line_inventory_error"] < 2e-15
    assert refined.grain_ids == state.grain_ids
