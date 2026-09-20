from dataclasses import replace
import json

import numpy as np
import pytest

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.analysis.run_v39_common_horizon import run_case
from full_model.production.extensive_wall import (
    accepted_ordering_step, extensive_wall_energy_components_J_m3,
)


def _sparse_wall_case():
    state, _, _, _, systems, topologies, _, parameters, _, _ = build_case(16)
    shape = state.density.wall_tangle_plus_m2.shape
    plus = np.zeros(shape); minus = np.zeros(shape)
    plus[7:9, :, :] = 1.2e13
    minus[7:9, :, :] = 9.0e12
    density = replace(
        state.density,
        wall_tangle_plus_m2=plus, wall_tangle_minus_m2=minus,
        wall_ordered_plus_m2=np.zeros(shape),
        wall_ordered_minus_m2=np.zeros(shape))
    target = np.zeros(state.common.orientation_rad.shape+(3, 3))
    stress = np.full(state.common.slip.shape, 7e8)
    return state, density, systems, topologies, parameters, target, stress


def test_stiff_ordering_asymptote_is_bounded_conservative_and_dissipative():
    state, density, systems, topologies, parameters, target, stress = (
        _sparse_wall_case())
    parameters = replace(
        parameters, ordering_integration_method="implicit_backward_euler",
        ordering_internal_substep_s=5e-13)
    before = extensive_wall_energy_components_J_m3(
        density, systems, topologies, state.common.orientation_rad, target,
        parameters)["total"]
    updated, alignment, ledger, _ = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, parameters, 5e-4,
        alignment=state.reservoir_alignment)
    after = extensive_wall_energy_components_J_m3(
        updated, systems, topologies, state.common.orientation_rad, target,
        parameters)["total"]
    assert ledger["integration_method"] == "bounded_convex_asymptotic"
    assert ledger["complete_elapsed_time_s"] == 5e-4
    assert ledger["discarded_reaction_time_s"] == 0.0
    assert ledger["asymptotic_endpoint_inventory_change_bound_relative"] < 1e-3
    assert np.sum(after) <= np.sum(before)+1e-10*max(abs(np.sum(before)), 1.0)
    for sign in ("plus", "minus"):
        total0 = (getattr(density, f"wall_tangle_{sign}_m2")
                  +getattr(density, f"wall_ordered_{sign}_m2"))
        total1 = (getattr(updated, f"wall_tangle_{sign}_m2")
                  +getattr(updated, f"wall_ordered_{sign}_m2"))
        assert np.array_equal(total0, total1)
    alignment.validate(updated, len(systems))
    assert np.max(np.abs(ledger["topology_ledger"][
        "total_nye_residual_m1"])) < 1e-12


def test_low_exposure_dispatch_uses_selected_finite_time_backend():
    state, density, systems, topologies, parameters, target, stress = (
        _sparse_wall_case())
    reference_parameters = replace(
        parameters, ordering_integration_method="finite_time_bdf",
        ordering_finite_time_backend="dense_bdf_oracle")
    selected = replace(
        parameters, ordering_integration_method="implicit_backward_euler",
        ordering_finite_time_backend="dense_bdf_oracle")
    reference = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, reference_parameters, 4e-10)
    actual = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, selected, 4e-10)
    assert actual[1]["stiff_dispatch"] == "finite_time_dense_bdf_oracle"
    for name in ("wall_tangle_plus_m2", "wall_tangle_minus_m2",
                 "wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        assert np.array_equal(getattr(actual[0], name),
                              getattr(reference[0], name))


def test_post_front_partial_restart_does_not_repeat_front(tmp_path):
    failed = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="injected post-front failure"):
        run_case(failed, grid=16, macro_dt_s=1e-3, intervals=1,
                 inject_post_front_failure=True)
    terminal = json.loads((failed/"terminal.json").read_text())
    assert terminal["accepted_complete_macros"] == 0
    assert terminal["partial_stage"] == "POST_MURA_PENDING"
    partial = failed/"partial_000001_post_front.npz"
    resumed = run_case(tmp_path/"resumed", grid=16, macro_dt_s=1e-3,
                       intervals=1, restart=partial)
    continuous = run_case(tmp_path/"continuous", grid=16, macro_dt_s=1e-3,
                          intervals=1)
    assert resumed["records"][0]["resumed_without_repeating_front"]
    assert resumed["physical_time_s"] == 1e-3
    assert resumed["records"][0]["mura_operator_exposure_s"] == 1e-3
    with np.load(tmp_path/"resumed"/"checkpoint_000001.npz") as left, np.load(
            tmp_path/"continuous"/"checkpoint_000001.npz") as right:
        shared = sorted(set(left.files)&set(right.files)-{
            "v39_stage_metadata_json"})
        for key in shared:
            assert np.array_equal(left[key], right[key]), key


def test_completed_macro_restart_can_adapt_macro_dt(tmp_path):
    first = run_case(tmp_path/"first", grid=16, macro_dt_s=2e-5,
                     intervals=1)
    continued = run_case(
        tmp_path/"continued", grid=16, macro_dt_s=1e-5, intervals=3,
        restart=tmp_path/"first"/"checkpoint_000001.npz")
    assert first["physical_time_s"] == 2e-5
    assert continued["physical_time_s"] == 4e-5
    assert continued["macro_dt_values_s"] == [1e-5, 2e-5]
    assert [row["macro_dt_s"] for row in continued["records"]] == [
        2e-5, 1e-5, 1e-5]


def test_pending_stage_restart_rejects_macro_dt_change(tmp_path):
    failed = tmp_path/"failed-adaptive"
    with pytest.raises(RuntimeError, match="injected post-front failure"):
        run_case(failed, grid=16, macro_dt_s=2e-5, intervals=1,
                 inject_post_front_failure=True)
    with pytest.raises(ValueError, match="requires its original macro dt"):
        run_case(tmp_path/"invalid", grid=16, macro_dt_s=1e-5, intervals=1,
                 restart=failed/"partial_000001_post_front.npz")


def test_front_direction_is_explicit_and_bounded(tmp_path):
    with pytest.raises(ValueError, match="front_direction"):
        run_case(tmp_path/"invalid-direction", front_direction=0)
