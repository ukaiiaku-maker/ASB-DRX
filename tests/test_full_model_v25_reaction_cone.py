import numpy as np

from full_model.production.reaction_cone import (
    ReactionEvent, audit_reaction_cone, circuit_event_from_line_change,
)


def event(name, direction, capacity=2.0, exposure=2.0):
    return ReactionEvent(
        name, np.asarray(direction, float), 0.0, np.zeros(3), np.zeros((3, 3)),
        np.zeros(3), 0.0, 0.0, 0.0, capacity, exposure)


def test_capacity_limited_cone_distinguishes_reachability_and_exposure():
    reached = audit_reaction_cone(
        (event("x", [1, 0, 0]), event("y", [0, 1, 0])), [1, 1, 0])
    assert reached["classification"] == "FB_TARGET_REACHABLE_WITH_SUFFICIENT_LOCAL_CAPACITY"
    slow = audit_reaction_cone(
        (event("x", [1, 0, 0], exposure=.1),
         event("y", [0, 1, 0], exposure=.1)), [1, 1, 0])
    assert slow["classification"] == "FB_TARGET_REACHABLE_BUT_KINETICALLY_UNEXPOSED"
    outside = audit_reaction_cone((event("x", [1, 0, 0]),), [0, 1, 0])
    assert outside["classification"] == "FB_TARGET_OUTSIDE_PHYSICAL_REACTION_CONE"


def test_local_line_rotation_without_swept_area_is_not_an_admissible_nye_source():
    rotated = circuit_event_from_line_change(
        "unowned_rotation", [1e-10, 0, 0], [0, 0, 1], [0, 1, 0],
        [0, 1, 0], 1e8)
    result = audit_reaction_cone((rotated,), [1e-2, 0, 0])
    assert result["inadmissible_events"] == ["unowned_rotation"]
    assert result["classification"] == "FB_TARGET_OUTSIDE_PHYSICAL_REACTION_CONE"
    swept = ReactionEvent(**{**rotated.__dict__, "name": "swept_rotation",
                             "swept_area_declared": True})
    assert audit_reaction_cone((swept,), [1e-2, 0, 0])["reachable"]
