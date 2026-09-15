import numpy as np

from full_model.production.extensive_wall import (
    orientation_gradient_frank_bilby_target_m1,
)
from full_model.production.wall_circuit_diagnostics import (
    classify_persistent_wall_history, local_integrated_wall_circuits,
)


def periodic_tilt_wall(n, length_m, angle_deg=2.0, axis=0,
                       wall_width_m=0.2e-6):
    dx = length_m/n
    coordinate = (np.arange(n)-n/2)*dx
    theta_1d = .5*np.deg2rad(angle_deg)*(
        np.tanh((coordinate+length_m/4)/wall_width_m)
        -np.tanh((coordinate-length_m/4)/wall_width_m))
    theta = (np.broadcast_to(theta_1d[:, None], (n, n)).copy() if axis == 0
             else np.broadcast_to(theta_1d[None, :], (n, n)).copy())
    alpha = orientation_gradient_frank_bilby_target_m1(theta, dx)
    return theta, alpha, dx


def test_local_integrated_circuits_close_and_converge_at_fixed_physical_width():
    errors = []
    for n in (32, 64, 128):
        theta, alpha, dx = periodic_tilt_wall(n, 6.4e-6)
        audit = local_integrated_wall_circuits(
            theta, alpha, alpha, dx, normal_window_m=1.4e-6,
            plateau_offset_m=.8e-6)
        assert len(audit["segments"]) == 2
        assert audit["ordered_line_overlap"] > .99
        for segment in audit["segments"]:
            assert np.rad2deg(segment.local_misorientation_rad) > 1.9
            assert abs(segment.ordered_supply_ratio-1.0) < .02
            assert segment.ordered_frank_bilby_residual < .02
        errors.append(max(s.ordered_frank_bilby_residual
                          for s in audit["segments"]))
    assert max(errors)-min(errors) < .01


def test_local_normal_handles_second_axis_and_global_span_cannot_qualify_elsewhere():
    theta, alpha, dx = periodic_tilt_wall(64, 6.4e-6, axis=1)
    # Put unrelated ordered content far inside a plateau. It should be counted
    # outside interface support even though the domain has a 2-degree span.
    false_alpha = alpha.copy()
    false_alpha[30:34, 30:34, 0, 2] += 2e5
    audit = local_integrated_wall_circuits(
        theta, false_alpha, alpha, dx, normal_window_m=1.0e-6,
        plateau_offset_m=.8e-6)
    assert len(audit["segments"]) == 2
    assert all(abs(abs(s.normal_xy[1])-1.0) < 1e-12 for s in audit["segments"])
    assert audit["ordered_line_outside_support"] > 0.0


def test_zero_orientation_gradient_produces_no_candidate_segments():
    theta = np.zeros((32, 32)); alpha = np.zeros((32, 32, 3, 3))
    audit = local_integrated_wall_circuits(
        theta, alpha, alpha, 1e-7, normal_window_m=1e-6)
    assert audit["segments"] == []
    assert audit["ordered_line_overlap"] == 0.0


def test_scientific_classification_requires_qualified_release_persistence():
    theta, alpha, dx = periodic_tilt_wall(64, 6.4e-6)
    snapshots = []
    for time_s, active in ((0.0, True), (1e-6, False), (2e-6, False)):
        snapshots.append({
            "time_s": time_s, "mechanical_loading_active": active,
            "orientation_rad": theta, "ordered_nye_m1": alpha,
            "total_nye_m1": alpha, "spacing_m": dx,
            "normal_window_m": 1.4e-6, "plateau_offset_m": .8e-6,
        })
    result = classify_persistent_wall_history(
        snapshots, required_release_persistence_s=1e-6)
    assert result["scientific_gate_passed"]
    assert result["grain_labels_allocated"] == 0
    # A loaded-only match remains a fixture and cannot pass persistence.
    result = classify_persistent_wall_history(snapshots[:1])
    assert result["fixture_passed"] and not result["scientific_gate_passed"]
