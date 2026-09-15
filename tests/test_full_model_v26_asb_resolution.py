from full_model.production.asb_grid_audit import (
    compare_kernel_audits, first_law_budget, kernel_moment_audit,
    manufactured_periodic_kernel,
)


def audit(n, minimum=0.0):
    kernel, rx, ry, _ = manufactured_periodic_kernel(
        n, 10e-6, .30e-6, minimum_sigma_pixels=minimum)
    return kernel_moment_audit(kernel, rx, ry, 10e-6/n)


def test_manufactured_kernel_preserves_energy_and_converges_when_resolved():
    a128, a192 = audit(128), audit(192)
    result = compare_kernel_audits(a128, a192)
    assert abs(a128["normalization"]-1.0) < 1e-14
    assert result["passed"]


def test_two_pixel_floor_is_detected_by_moments_not_nominal_width_alone():
    coarse, fine = audit(32, minimum=2.0), audit(192, minimum=2.0)
    result = compare_kernel_audits(coarse, fine)
    assert not result["passed"]
    assert result["second_moment_relative_difference"] > .05


def test_first_law_reports_both_plastic_partition_and_system_residual():
    result = first_law_budget(
        external_work_J_m3=10, elastic_change_J_m3=2,
        plastic_work_J_m3=8, defect_free_energy_change_J_m3=1,
        taylor_quinney_heat_J_m3=6, conducted_out_J_m3=2,
        thermal_energy_change_J_m3=4, other_dissipation_J_m3=1)
    assert result["plastic_partition_residual_J_m3"] == 0
    assert result["system_residual_J_m3"] == 0

