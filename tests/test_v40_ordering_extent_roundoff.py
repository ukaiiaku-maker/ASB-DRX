from dataclasses import replace

import numpy as np

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.wall_topology_supply import apply_signed_ordering_extent


def test_nearly_complete_ordering_preserves_saturated_alignment_bound():
    state, _, _, _, systems, topologies, _, _, _, _ = build_case(16)
    shape = state.density.wall_tangle_plus_m2.shape
    tangle = np.full(shape, 1.0e13)
    ordered = np.full(shape, 1.0e9)
    density = replace(
        state.density, wall_tangle_plus_m2=tangle,
        wall_ordered_plus_m2=ordered)
    direction = np.zeros(shape+(3,)); direction[..., 0] = 1.0
    alignment = replace(
        state.reservoir_alignment,
        wall_tangle_plus_m2=tangle[..., None]*direction,
        wall_ordered_plus_m2=ordered[..., None]*direction)
    extent = tangle*(1.0-1.0e-15)
    zero = np.zeros(shape)
    updated, aligned, ledger = apply_signed_ordering_extent(
        density, alignment, extent, zero, systems,
        state.common.orientation_rad, topologies)
    aligned.validate(updated, len(systems))
    assert np.allclose(
        updated.wall_tangle_plus_m2+updated.wall_ordered_plus_m2,
        tangle+ordered, rtol=2e-16, atol=0.0)
    assert np.allclose(
        aligned.wall_tangle_plus_m2+aligned.wall_ordered_plus_m2,
        alignment.wall_tangle_plus_m2+alignment.wall_ordered_plus_m2,
        rtol=2e-16, atol=0.0)
    assert np.max(np.abs(ledger["total_nye_residual_m1"])) < 1e-8
