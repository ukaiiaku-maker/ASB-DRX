from full_model.analysis.analyze_v67_mobile_supply_collapse import classify


def _case(mobile, flow, taylor, speed, source):
    return {
        "extensive_cell_inventory_m2": {"mobile": mobile},
        "flow_exposure_fraction": flow,
        "support_family_weighted_means": {
            "taylor_Pa": taylor, "speed_abs_m_s": speed},
        "source_created_line_cell_sum_m2": source,
        "hard_audits_passed": True,
    }


def test_mobile_loss_with_rising_speed_and_fixed_taylor_classifies_starvation():
    result = classify(
        _case(1e16, .2, 2e8, 100.0, 1e12),
        _case(1e15, 1e-4, 2.05e8, 200.0, 2e12))
    assert result["decision"]["mobile_supply_collapse_not_taylor_locking"]


def test_rising_taylor_resistance_rejects_supply_only_diagnosis():
    result = classify(
        _case(1e16, .2, 2e8, 100.0, 1e12),
        _case(1e15, 1e-4, 4e8, 200.0, 2e12))
    assert not result["decision"]["mobile_supply_collapse_not_taylor_locking"]
