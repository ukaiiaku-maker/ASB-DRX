from full_model.analysis.analyze_v67_carrier_transparency_requirement import (
    classify_transparency,
)


def _rows(value=.985, hard=True):
    return {
        "adiabatic": {
            "hard_audits_passed": hard,
            "minimum_transparency_for_no_net_mobile_loss": value,
        },
        "strong_bath": {
            "hard_audits_passed": hard,
            "minimum_transparency_for_no_net_mobile_loss": value-.001,
        },
    }


def test_transparency_bound_requires_hard_audits():
    assert classify_transparency(_rows(hard=False)) == (
        "TRANSPARENCY_BOUND_NOT_AUDIT_QUALIFIED")


def test_near_complete_transparency_is_not_silently_promoted():
    assert classify_transparency(_rows()) == (
        "NEAR_COMPLETE_TRANSPARENCY_REQUIRED_NOT_PRODUCTION_IDENTIFIED")
    assert classify_transparency(_rows(.5)) == (
        "PARTIAL_TRANSPARENCY_COULD_CLOSE_MOBILE_BALANCE")
    assert classify_transparency(_rows(1.1)) == (
        "TRANSPARENCY_ALONE_CANNOT_CLOSE_MOBILE_BALANCE")
