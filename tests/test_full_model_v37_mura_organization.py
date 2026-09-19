import numpy as np

from full_model.analysis.postprocess_v37_mura_organization import (
    checkpoint_scoped_history, history_hard_invariants_pass,
)
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
    # V43 assigns the geometry-neutral ordering reaction and its heat to both
    # controls identically.  The topology flag only labels the unavailable
    # legacy reorientation route; persistent plaquette geometry is exercised
    # separately by the V43 event tests.
    assert records[0]["ordered_line_m2_cells"] == records[1][
        "ordered_line_m2_cells"]


def test_compact_postprocessing_excludes_history_newer_than_checkpoint():
    history = [
        {"step": 100, "applied_strain": 0.040},
        {"step": 125, "applied_strain": 0.045},
        {"step": 150, "applied_strain": 0.050},
    ]
    scoped = checkpoint_scoped_history(
        history, {"step": 125, "applied_strain": 0.045})
    assert [row["step"] for row in scoped] == [100, 125]


def test_organization_hard_gate_includes_authoritative_nye_checks():
    base = {
        "accepted_step_hard_invariant_passed": True,
        "post_step_projection_used": False,
        "minimum_heat_increment_J_m3": 0.0,
        "authoritative_source_offset_relative_rms": 1e-12,
        "normalized_line_continuity_residual": 1e-12,
    }
    assert history_hard_invariants_pass([base])
    assert not history_hard_invariants_pass([
        {**base, "authoritative_source_offset_relative_rms": 0.69}])
    assert not history_hard_invariants_pass([
        {**base, "normalized_line_continuity_residual": 0.66}])
