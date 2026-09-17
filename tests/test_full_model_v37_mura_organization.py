import numpy as np

from full_model.analysis.run_v30_mura_tier_b1_case import compact_metrics
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from tests.test_full_model_v24_mechanical_wall import mechanical_fixture


def test_v37_organization_metrics_are_finite_and_topology_separated():
    args = mechanical_fixture()
    records = []
    for topology_on in (False, True):
        state, ledger = accepted_v24_mechanical_step(
            *args, dt_s=1e-9, topology_route_enabled=topology_on,
            mura_work_budget_mode="energy_limited_feasible_extents")
        metrics = compact_metrics(
            state, ledger, args[3], args[4], args[5].spacing_m)
        for key in (
                "ordered_line_m2_cells", "total_wall_line_m2_cells",
                "ordered_fraction_global", "ordered_fraction_wall_local",
                "ordered_polarization_maximum",
                "ordered_polarization_wall_local_mean",
                "junction_line_m2_cells"):
            assert np.isfinite(metrics[key])
            assert metrics[key] >= 0.0
        assert metrics["accepted_step_hard_invariant_passed"]
        records.append(metrics)
    # The disabled topology route and explicit topology route are separate
    # physical comparators, not aliases that silently exercise one operator.
    assert records[0]["ordered_line_m2_cells"] != records[1][
        "ordered_line_m2_cells"]
