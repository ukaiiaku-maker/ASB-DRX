import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _smooth_voronoi_supports,
)


def _transition_cells(n):
    supports = _smooth_voronoi_supports(
        n, ((.25*n, .5*n), (.75*n, .5*n)), width_cells=.05*n)
    profile = supports[0, :, n//2]
    return int(np.count_nonzero((profile > .1) & (profile < .9)))


def test_declared_physical_width_refines_in_cells():
    coarse = _transition_cells(64)
    fine = _transition_cells(128)
    assert coarse >= 4
    assert 1.7 <= fine/coarse <= 2.3


def test_distance_softmax_is_a_partition_with_pure_cores():
    supports = _smooth_voronoi_supports(
        64, ((.2*64, .5*64), (.7*64, .25*64), (.7*64, .75*64)),
        width_cells=2.0)
    np.testing.assert_allclose(np.sum(supports, axis=0), 1.0, atol=1e-14)
    assert np.all(supports >= 0.0)
    assert np.all(np.max(supports, axis=(1, 2)) > .99)
