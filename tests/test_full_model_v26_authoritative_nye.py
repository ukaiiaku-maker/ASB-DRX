import numpy as np

from full_model.production.reaction_cone import (
    audit_reaction_cone, circuit_event_from_line_change, swept_line_event,
)


def test_strict_cone_rejects_boolean_ownership_without_beta_increment():
    event = circuit_event_from_line_change(
        "flag_only", [2.48e-10, 0, 0], [0, 0, 1], [0, 1, 0],
        [0, 1, 0], 1e8, swept_area_declared=True)
    result = audit_reaction_cone(
        (event,), event.circuit_increment_per_extent_m*1e7,
        require_event_identity=True)
    assert result["inadmissible_events"] == ["flag_only"]
    assert not result["event_validation"][0][
        "authoritative_event_identity_valid"]
    assert result["classification"] == (
        "FB_TARGET_OUTSIDE_AUTHORITATIVE_REACTION_CONE")


def test_explicit_swept_area_event_closes_nye_beta_identity_and_is_reachable():
    event = swept_line_event(
        "swept", [2.48e-10, 0, 0], [0, 0, 1], [0, 1, 0],
        [0, 1, 0], 1e8, kinetic_exposure_extent_m1=1e8)
    result = audit_reaction_cone(
        (event,), event.circuit_increment_per_extent_m*1e7,
        require_event_identity=True)
    valid = result["event_validation"][0]
    assert valid["authoritative_event_identity_valid"]
    assert valid["nye_identity_absolute_residual_m"] == 0.0
    assert result["classification"] == "FB_TARGET_REACHABLE_WITH_CURRENT_CAPACITY"


def test_declared_source_tensor_must_match_total_nye_increment():
    event = circuit_event_from_line_change(
        "loop_source", [2.48e-10, 0, 0], [0, 0, 1], [0, 1, 0],
        [0, 1, 0], 1e8, explicit_source_or_sink=True,
        declared_nye_source_increment_per_extent_m=np.array([
            [0.0, 2.48e-10, -2.48e-10], [0, 0, 0], [0, 0, 0]]),
        event_class="loop_nucleation_or_escape")
    assert event.validate(require_event_identity=True)[
        "authoritative_event_identity_valid"]

