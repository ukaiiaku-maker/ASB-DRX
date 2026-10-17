from full_model.analysis.analyze_v67_interface_emission_fork import classify


def _case(parent="same", flow=.001, mobile=1.0, emitted=.1):
    return {
        "parent_checkpoint_sha256": parent,
        "flow_exposure_fraction": flow,
        "extensive_cell_inventory_m2": {"mobile": mobile},
        "emitted_pair_increment_cell_sum_m2": emitted,
        "hard_audits_passed": True,
    }


def test_interface_emission_flow_recovery_is_explicit():
    parent = {"checkpoint_sha256": "same", "hard_audits_passed": True}
    result = classify(parent, _case(), _case(flow=.02, mobile=2.0))
    assert result["classification"] == (
        "INTERFACE_EMISSION_RESTORES_PHYSICAL_FLOW")


def test_interface_emission_requires_exact_parent():
    parent = {"checkpoint_sha256": "same", "hard_audits_passed": True}
    result = classify(parent, _case(), _case(parent="other", flow=.02))
    assert result["classification"] == "INTERFACE_EMISSION_FORK_INVALID"


def test_small_interface_emission_response_is_insufficient():
    parent = {"checkpoint_sha256": "same", "hard_audits_passed": True}
    result = classify(parent, _case(), _case(flow=.0015, mobile=1.1))
    assert result["classification"] == (
        "INTERFACE_EMISSION_INSUFFICIENT_TO_PREVENT_FLOW_ARREST")
