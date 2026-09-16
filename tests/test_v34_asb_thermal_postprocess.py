from full_model.analysis.postprocess_v34_asb_thermal import (
    actual_semantics,
    process_records,
    read_json_if_present,
)


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
