from full_model.analysis.postprocess_v36_recurrent_response import (
    build_decision, relative_difference,
)


def _case(name, sweep, velocity, *, intervals=1, grid=16, dt=1e-5):
    data = {
        "source_commit": "a"*40,
        "configuration": {
            "grid": grid, "dt_s": dt, "protocol": "hold",
        },
        "completed_intervals": intervals,
        "physical_time_s": intervals*dt,
        "accepted_front_intervals": intervals,
        "cumulative_signed_sweep_m3": sweep,
        "gross_expected_event_count": 2.0,
        "net_expected_event_count": 1.0,
        "classification": "STORED_ENERGY_DRIVEN_MIGRATION_WITHOUT_APPLIED_FRONT_WORK",
        "records": [{
            "net_velocity_m_s": velocity, "front_published": True,
        }],
    }
    return {"path": f"/{name}/result.json", "sha256": "b"*64,
            "data": data}


def test_relative_difference_is_symmetric():
    assert relative_difference(9.5, 10.0) == relative_difference(10.0, 9.5)
    assert relative_difference(9.5, 10.0) == 0.05


def test_decision_requires_long_n128_n192_pair_for_scientific_gate():
    cases = {
        "hold_dt1e5_20": _case("a", 1.0, 1.0, intervals=20),
        "hold_dt5e6_40": _case("b", .98, 1.0, intervals=40, dt=5e-6),
        "hold_n128_pf00625_1": _case("c", 1.0, 1.0, grid=128),
        "hold_n192_pf00625_1": _case("d", 1.0, .97, grid=192),
    }
    pending = build_decision(cases)
    assert pending["fixture_passed"]
    assert not pending["scientific_gate_passed"]
    assert pending["classification"] == (
        "INITIAL_RATE_QUALIFIED_LONG_HORIZON_RESOLUTION_PENDING")

    cases["hold_n128_pf00625_10"] = _case(
        "e", 2.0, 1.0, intervals=10, grid=128)
    cases["hold_n192_pf00625_10"] = _case(
        "f", 1.94, .97, intervals=10, grid=192)
    qualified = build_decision(cases)
    assert qualified["scientific_gate_passed"]
    assert qualified["classification"] == (
        "RESOLVED_ZERO_WORK_RECURRENT_RESPONSE_QUALIFIED")
