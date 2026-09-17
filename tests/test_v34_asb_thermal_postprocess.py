from full_model.analysis.postprocess_v34_asb_thermal import (
    actual_semantics,
    matched_causal_effect,
    process_records,
    read_json_if_present,
)


def summary(**updates):
    base = {
        "step": 100, "sim_time_s": 1e-6, "nominal_strain": 0.1,
        "stress_Pa": 2e9, "active_fraction": 0.5,
        "temperature_max_K": 1200.0, "softening_fraction": 0.1,
        "effective_width_m": 2e-6, "causal_interpretation_valid": True,
    }
    return base | updates


def record(source="a"*40, prefix="b"*64):
    return {"source_commit": source, "shared_checkpoint_sha256": prefix}


def test_actual_semantics_distinguishes_all_boundary_operators():
    assert actual_semantics({
        "thermal_control_semantics": "exact_prescribed_temperature",
        "k_thermal": 0.0, "T_bath_coupling": 0.0,
    }) == "EXACT_PRESCRIBED_TEMPERATURE_WITH_THERMOSTAT_EXPORT"
    assert actual_semantics({
        "thermal_control_semantics": "auto", "k_thermal": 0.15,
        "T_bath_coupling": 0.0,
    }) == "FINITE_CONDUCTION_PERIODIC_INSULATED"
    assert actual_semantics({
        "thermal_control_semantics": "auto", "k_thermal": 0.15,
        "T_bath_coupling": 2e10,
    }) == "FINITE_BATH"
    assert actual_semantics({
        "thermal_control_semantics": "auto", "k_thermal": 0.0,
        "T_bath_coupling": 0.0,
    }) == "NO_CONDUCTION_LOCAL_ADIABATIC"


def test_launch_provenance_readers(tmp_path):
    assert read_json_if_present(tmp_path/"missing.json") == {}
    (tmp_path/"launch.json").write_text('{"state":"RUNNING"}\n')
    assert read_json_if_present(tmp_path/"launch.json")["state"] == "RUNNING"
    (tmp_path/"processes.tsv").write_text(
        "0\t123\t2026-09-16T00:00:00Z\n1\t456\t2026-09-16T00:01:00Z\n")
    assert process_records(tmp_path/"processes.tsv") == [
        {"case_id": 0, "pid": 123, "launched_utc": "2026-09-16T00:00:00Z"},
        {"case_id": 1, "pid": 456, "launched_utc": "2026-09-16T00:01:00Z"},
    ]


def test_matched_effect_has_units_provenance_and_no_interpolation():
    effect = matched_causal_effect(
        summary(stress_Pa=2.2e9), summary(), record("c"*40), record())
    assert effect["status"] == "MATCHED_EXACT"
    assert effect["numeric_effects"]["delta_stress_Pa"] == 2e8
    assert effect["units"]["delta_stress_Pa"] == "Pa"
    assert effect["interpolation"] == {
        "used": False, "method": "none", "status": "NOT_REQUIRED_EXACT_MATCH"}
    assert effect["provenance"]["attributable"]
    assert effect["provenance"]["common_prefix"]


def test_unmatched_invalid_and_unattributed_effects_are_null():
    scenarios = (
        (summary(sim_time_s=2e-6), record(), "UNMATCHED_TIME_OR_STRAIN"),
        (summary(causal_interpretation_valid=False), record(),
         "INVALID_CAUSAL_ROUTING"),
        (summary(), {}, "PROVENANCE_UNATTRIBUTABLE"),
        (summary(), record(prefix="d"*64), "COMMON_PREFIX_MISMATCH"),
    )
    for subject, subject_record, expected in scenarios:
        effect = matched_causal_effect(subject, summary(), subject_record, record())
        assert effect["status"] == expected
        assert all(value is None for value in effect["numeric_effects"].values())
        assert effect["interpolation"]["used"] is False
