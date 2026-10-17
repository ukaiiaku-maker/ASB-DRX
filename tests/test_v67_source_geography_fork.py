from full_model.analysis.analyze_v67_source_geography_fork import classify


def _case(parent="same", flow=.001, mobile=1.0, source=.1):
    return {
        "parent_checkpoint_sha256": parent,
        "flow_exposure_fraction": flow,
        "extensive_cell_inventory_m2": {"mobile": mobile},
        "source_increment_cell_sum_m2": source,
        "hard_audits_passed": True,
    }


def test_matched_random_physical_flow_recovery_is_explicit():
    parent = {"checkpoint_sha256": "same", "hard_audits_passed": True}
    result = classify(
        parent, _case(), _case(flow=.02, mobile=2.0, source=.2))
    assert result["classification"] == "MATCHED_RANDOM_RESTORES_PHYSICAL_FLOW"
    assert result["same_parent_checkpoint"]


def test_different_parent_invalidates_geography_fork():
    parent = {"checkpoint_sha256": "same", "hard_audits_passed": True}
    result = classify(parent, _case(), _case(parent="other", flow=.02))
    assert result["classification"] == "SOURCE_GEOGRAPHY_FORK_INVALID"


def test_small_response_is_classified_insufficient():
    parent = {"checkpoint_sha256": "same", "hard_audits_passed": True}
    result = classify(
        parent, _case(), _case(flow=.0015, mobile=1.1, source=.1))
    assert result["classification"] == (
        "SOURCE_GEOGRAPHY_INSUFFICIENT_TO_PREVENT_FLOW_ARREST")
