from full_model.analysis.postprocess_v34_asb_thermal import actual_semantics


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
