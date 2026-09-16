from full_model.hpc3.run_v33_asb_causal_ablation import CASES, case_definition


def test_causal_matrix_includes_broad_and_localized_negative_evidence():
    cases = [case_definition(index) for index in range(len(CASES))]
    assert {case["temperature_K"] for case in cases} == {900.0, 1100.0}
    assert {case["causal_temperature_ablation"] for case in cases} == {
        "none", "freeze_flow", "freeze_recovery"}
    assert all(case["thermal_operator"] == "NO_CONDUCTION_LOCAL_ADIABATIC"
               for case in cases)


def test_only_broad_condition_gets_new_same_source_baseline():
    baseline = [case for case in (case_definition(i) for i in range(len(CASES)))
                if case["causal_temperature_ablation"] == "none"]
    assert len(baseline) == 1
    assert baseline[0]["temperature_K"] == 1100.0
