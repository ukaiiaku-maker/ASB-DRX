import numpy as np

from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from tests.test_full_model_v24_mechanical_wall import mechanical_fixture


def test_complete_mura_budget_closes_with_nonnegative_physical_heat():
    state, ledger = accepted_v24_mechanical_step(
        *mechanical_fixture(), dt_s=1e-9,
        mura_work_budget_mode="energy_limited")
    del state
    balance = ledger["mura_balance_ledger"]
    budget = ledger["mura_work_budget"]["accepted"]
    assert budget["admissible"]
    assert budget["plastic_work_not_double_counted"]
    assert np.min(balance["deposited_heat_increment_J_m3"]) >= 0.0
    scale = max(abs(balance["recoverable_elastic_energy_release_J_m3_cells"]), 1.0)
    assert abs(balance[
        "global_work_minus_heat_storage_residual_J_m3_cells"]) < 2e-14*scale


def test_legacy_reject_mode_is_exact_old_line_creation_budget_control():
    state, ledger = accepted_v24_mechanical_step(
        *mechanical_fixture(), dt_s=1e-9,
        mura_work_budget_mode="legacy_reject")
    del state
    balance = ledger["mura_balance_ledger"]
    np.testing.assert_array_equal(
        balance["stored_line_energy_increment_J_m3"],
        balance["line_creation_energy_increment_J_m3"])
    work = float(np.sum(balance["plastic_work_increment_J_m3"]))
    heat = float(np.sum(balance["deposited_heat_increment_J_m3"]))
    stored = float(np.sum(balance["stored_line_energy_increment_J_m3"]))
    assert abs(work-heat-stored) < 2e-14*max(abs(work), 1.0)
    np.testing.assert_array_equal(ledger["mura_family_event_scales"], np.ones(4))
    assert ledger["mura_event_scale"] == 1.0


def test_energy_limited_complete_restart_remains_bitwise_exact():
    from full_model.production.v24_mechanical_wall import (
        mechanical_checkpoint_arrays, mechanical_from_checkpoint_arrays,
    )
    args = mechanical_fixture()
    first, _ = accepted_v24_mechanical_step(
        *args, dt_s=1e-9, mura_work_budget_mode="energy_limited")
    restored = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), args[3], args[4])
    continuous, _ = accepted_v24_mechanical_step(
        first, *args[1:], dt_s=1e-9,
        mura_work_budget_mode="energy_limited")
    restarted, _ = accepted_v24_mechanical_step(
        restored, *args[1:], dt_s=1e-9,
        mura_work_budget_mode="energy_limited")
    for group in ("common", "density", "reservoir_alignment"):
        left = getattr(continuous, group); right = getattr(restarted, group)
        for name in left.__dict__:
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
