import numpy as np
from dataclasses import replace

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.common_front_state import (
    _minimum_change_bounded_moments,
    _reference_minimum_change_bounded_moments,
)
from full_model.production.extensive_wall import accepted_ordering_step


def _case(seed=17, shape=(13, 11, 4)):
    rng = np.random.default_rng(seed)
    raw_weights = rng.random((3,)+shape[:2])
    raw_weights[2, :3] = 0.0
    weights = raw_weights/np.sum(raw_weights, axis=0)
    bounds = [1.0+rng.random(shape) for _ in range(3)]
    feasible = []
    for bound in bounds:
        direction = rng.normal(size=shape+(3,))
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True),
                                1e-300)
        feasible.append(direction*bound[..., None]*rng.random(shape+(1,)))
    target = sum(weight[..., None, None]*moment
                 for weight, moment in zip(weights, feasible))
    baselines = [rng.normal(size=shape+(3,))*bound[..., None]
                 for bound in bounds]
    return baselines, bounds, list(weights), target


def test_compact_bounded_moment_map_matches_reference_and_constraints():
    baselines, bounds, weights, target = _case()
    reference = _reference_minimum_change_bounded_moments(
        baselines, bounds, weights, target)
    compact = _minimum_change_bounded_moments(
        baselines, bounds, weights, target)
    scale = max(float(np.max(np.abs(target))), 1.0)
    for actual, expected, bound, weight in zip(
            compact, reference, bounds, weights):
        assert np.allclose(actual, expected, rtol=2e-12, atol=2e-12*scale)
        mask = np.broadcast_to(
            weight[..., None] > 64*np.finfo(float).eps, bound.shape)
        active_violation = (np.linalg.norm(actual, axis=-1)-bound)[mask]
        assert np.max(active_violation) <= 2e-12*scale
    mixture = sum(weight[..., None, None]*moment
                  for weight, moment in zip(weights, compact))
    assert np.max(np.abs(mixture-target)) <= 2e-12*scale


def test_compact_bounded_moment_map_full_polarization_limit():
    shape = (5, 7, 2)
    weights = [np.full(shape[:2], 0.2), np.full(shape[:2], 0.3),
               np.full(shape[:2], 0.5)]
    bounds = [np.full(shape, 2.0), np.full(shape, 3.0), np.full(shape, 4.0)]
    direction = np.zeros(shape+(3,)); direction[..., 1] = 1.0
    target = sum(weight[..., None, None]*bound[..., None]
                 for weight, bound in zip(weights, bounds))*direction
    baselines = [np.zeros_like(target) for _ in range(3)]
    result = _minimum_change_bounded_moments(
        baselines, bounds, weights, target)
    for moment, bound in zip(result, bounds):
        assert np.allclose(moment, bound[..., None]*direction)


def test_ordering_subcycle_covers_same_clock_and_alignment_conservatively():
    (state, _, _, _, systems, topologies, _, extensive, _, _) = build_case(16)
    shape = state.density.wall_tangle_plus_m2.shape
    density = replace(
        state.density,
        wall_tangle_plus_m2=np.full(shape, 5e13),
        wall_tangle_minus_m2=np.full(shape, 4e13),
        wall_ordered_plus_m2=np.full(shape, 1e13),
        wall_ordered_minus_m2=np.full(shape, 2e13))
    extensive = replace(
        extensive, ordering_internal_substep_s=5e-13,
        ordering_internal_max_substeps=128)
    zero_target = np.zeros(state.common.orientation_rad.shape+(3, 3))
    stress = np.full(state.common.slip.shape, 7e8)
    one = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad,
        zero_target, stress, state.common.temperature_K, extensive, 4e-12,
        alignment=state.reservoir_alignment)
    half = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad,
        zero_target, stress, state.common.temperature_K, extensive, 2e-12,
        alignment=state.reservoir_alignment)
    two = accepted_ordering_step(
        half[0], systems, topologies, state.common.orientation_rad,
        zero_target, stress, state.common.temperature_K, extensive, 2e-12,
        alignment=half[1])
    for name in ("wall_tangle_plus_m2", "wall_tangle_minus_m2",
                 "wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        assert np.array_equal(getattr(one[0], name), getattr(two[0], name))
        assert np.array_equal(getattr(one[1], name), getattr(two[1], name))
    assert one[2]["complete_elapsed_time_s"] == 4e-12
    assert one[2]["discarded_reaction_time_s"] == 0.0
    for sign in ("plus", "minus"):
        before = (getattr(density, f"wall_tangle_{sign}_m2")
                  +getattr(density, f"wall_ordered_{sign}_m2"))
        after = (getattr(one[0], f"wall_tangle_{sign}_m2")
                 +getattr(one[0], f"wall_ordered_{sign}_m2"))
        assert np.max(np.abs(before-after)) <= 0.05
