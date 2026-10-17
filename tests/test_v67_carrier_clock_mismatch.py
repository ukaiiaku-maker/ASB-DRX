from full_model.analysis.analyze_v67_carrier_clock_mismatch import (
    classify_clock_audit,
)


def _metrics(**changes):
    values = {
        "capture_accepted": True,
        "hall_petch_emission_accepted": True,
        "existing_to_new_opportunity_rate_ratio": 9.0e7,
        "neutral_interface_pair_to_mobile_inventory_ratio": 40.0,
    }
    values.update(changes)
    return values


def test_clock_audit_requires_independent_energy_qualification():
    assert classify_clock_audit(_metrics(capture_accepted=False)) == (
        "CAPTURE_CLOCK_NOT_ENERGY_QUALIFIED")
    assert classify_clock_audit(
        _metrics(hall_petch_emission_accepted=False)) == (
            "HALL_PETCH_EMISSION_NOT_ENERGY_QUALIFIED")


def test_clock_audit_separates_site_measure_from_reservoir_capacity():
    assert classify_clock_audit(_metrics(
        existing_to_new_opportunity_rate_ratio=1.0e3)) == (
            "SOURCE_OPPORTUNITY_MEASURE_NOT_RATE_SEPARATING")
    assert classify_clock_audit(_metrics(
        neutral_interface_pair_to_mobile_inventory_ratio=.5)) == (
            "INTERFACE_RESERVOIR_NOT_CARRIER_DOMINANT")
    assert classify_clock_audit(_metrics()) == (
        "SITE_MULTIPLICITY_DOMINATES_CARRIER_RECYCLE_CLOCK")
