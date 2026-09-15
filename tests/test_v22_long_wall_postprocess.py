from full_model.analysis.postprocess_v22_long_wall import classify_campaign


def _case(name, q_mean, q_std, wall=False):
    return {
        "case": name,
        "complete_enough_to_classify": True,
        "wall_order": {
            "mean": q_mean,
            "maximum": q_mean,
            "standard_deviation": q_std,
        },
        "orientation": {"span_deg": 0.07},
        "independent_frank_bilby": {"candidate_wall_present": wall},
    }


def test_v22_uniform_order_decision_requires_disabled_control():
    cases = [
        _case("heterogeneous_on_64", 0.999, 1e-5),
        _case("order_disabled_64", 0.0, 0.0),
    ]
    result = classify_campaign(cases)
    assert result["classification"] == "WALL_ORDER_FUNCTIONAL_STILL_UNPHYSICAL"
    assert result["scientific_gate_passed"] is False
    assert result["decision_evidence"]["order_disabled_control_passed"] is True
    assert cases[0]["classification"] == "UNIFORM_ORDER_WITHOUT_LAGB"


def test_v22_wall_candidate_is_not_promoted_before_release_audit():
    cases = [
        _case("heterogeneous_on_64", 0.7, 0.1, wall=True),
        _case("order_disabled_64", 0.0, 0.0),
    ]
    result = classify_campaign(cases)
    assert result["classification"] == (
        "COMPATIBLE_LAGB_CANDIDATE_REQUIRES_RELEASE_AUDIT")
    assert result["scientific_gate_passed"] is False
