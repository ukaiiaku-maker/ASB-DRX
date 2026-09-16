from full_model.hpc3.run_v34_asb_thermal_case import CASES, case_definition


def test_v34_matrix_has_five_distinct_declared_operators():
    cases = [case_definition(index) for index in range(len(CASES))]
    assert len(cases) == 5
    assert len({case["case_name"] for case in cases}) == 5
    assert {case["causal_temperature_ablation"] for case in cases} == {
        "none", "freeze_flow", "freeze_recovery"}
    assert sum(case["thermal_control_semantics"] ==
               "exact_prescribed_temperature" for case in cases) == 1
    assert sum(case["conductivity_W_m_K"] > 0.0 for case in cases) == 1
    assert all(case["bath_coupling_W_m3_K"] == 0.0 for case in cases)


def test_exact_thermostat_and_conduction_are_not_mislabeled_finite_bath():
    thermostat = case_definition(3)
    conduction = case_definition(4)
    assert thermostat["thermal_control_semantics"] == "exact_prescribed_temperature"
    assert thermostat["conductivity_W_m_K"] == 0.0
    assert conduction["thermal_control_semantics"] == "auto"
    assert conduction["conductivity_W_m_K"] == 0.15
