from full_model.analysis.analyze_v67_arrival_exchange_candidate import classify


def _payload(flow=.05, base=1e-4, dynamic=1e-4, static=1e-4,
             persistence=2, softening=0.0, hard=True):
    return {
        "high_rate": {
            "reaction_only_ad600": {"endpoint": {
                "hard_valid": hard,
                "plastic_to_imposed_power_fraction": base}},
            "arrival_limited_ad600": {"endpoint": {
                "hard_valid": hard,
                "plastic_to_imposed_power_fraction": flow,
                "terminal_band_persistence_intervals": persistence,
                "endpoint_post_peak_softening_fraction": softening}},
        },
        "lower_rate_dynamic": {"result": {
            "hard_valid": hard, "grain20_relative_area_change": dynamic}},
        "retained_static": {"result": {
            "hard_valid": hard, "grain20_relative_area_change": static}},
    }


def test_valid_candidate_separates_carrier_repair_from_asb_qualification():
    decision = classify(_payload())
    assert decision["prevents_high_rate_flow_collapse"]
    assert decision["preserves_lower_rate_dynamic_and_static_rx"]
    assert not decision["qualifies_high_rate_asb"]
    assert decision["classification"] == (
        "CARRIER_COLLAPSE_PREVENTED_RX_PRESERVED_ASB_NOT_YET_QUALIFIED")


def test_asb_requires_persistence_and_softening_and_hard_validity():
    assert classify(_payload(persistence=20, softening=.05))[
        "qualifies_high_rate_asb"]
    assert classify(_payload(hard=False))["classification"] == (
        "HARD_INVALID_ARRIVAL_EXCHANGE_CANDIDATE")
