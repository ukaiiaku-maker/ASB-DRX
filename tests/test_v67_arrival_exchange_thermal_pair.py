from full_model.analysis.analyze_v67_arrival_exchange_thermal_pair import classify


def _row(bands, persistence, softening, temp, growth=1e-4, hard=True):
    return {
        "hard_valid": hard,
        "active_band_samples": bands,
        "terminal_band_persistence_intervals": persistence,
        "endpoint_post_peak_softening_fraction": softening,
        "endpoint": {"temperature_range_K": temp},
        "grain20_relative_area_change": growth,
    }


def test_heat_biased_recurrence_does_not_qualify_asb_without_softening():
    ad = [_row(1, 0, 0, 40.0), _row(24, 2, 0, 45.0)]
    bath = [_row(1, 0, 0, .1), _row(0, 0, 0, .01)]
    decision = classify(ad, bath)
    assert decision["heat_biases_late_band_recurrence"]
    assert not decision["asb_qualified"]
    assert decision["classification"] == (
        "HEAT_BIASED_BAND_RECURRENCE_WITH_RX_BUT_NO_ASB_SOFTENING")


def test_asb_requires_persistence_softening_and_bath_separation():
    ad = [_row(20, 20, .05, 45.0)] * 2
    bath = [_row(0, 0, 0, .01)] * 2
    assert classify(ad, bath)["asb_qualified"]
    assert classify(ad, [_row(20, 20, .05, .01)] * 2)["asb_qualified"] is False
