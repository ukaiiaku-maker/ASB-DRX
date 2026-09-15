import numpy as np

from full_model.production.asb_grid_audit import (
    categorical_restriction_mismatch, compare_physical_scales,
    integrate_work_heat, physical_scales,
)


def params(n):
    return dict(Nx=n, L_phys=10e-6, kappa_eta=5e-7, W_eta=5e6,
                heat_process_zone_sigma_um=.3,
                heat_process_zone_min_sigma_px=2., k_thermal=.015,
                cp_rho_vol=3.8e6, collective_activity_smooth_um=.1,
                nuc_barrier_thickness_b=2., b=2.48e-10)


def test_physical_length_audit_exposes_cell_minimum_but_not_cell_area():
    coarse, fine = physical_scales(params(64)), physical_scales(params(128))
    comparison = compare_physical_scales(coarse, fine)
    assert comparison["domain_size_m"]["passed"]
    assert comparison["heat_kernel_effective_m"]["passed"]
    assert coarse["cell_area_m2"] == 4*fine["cell_area_m2"]


def test_partition_restriction_is_label_permutation_invariant():
    coarse = np.array([[0, 0], [1, 1]])
    fine = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)+7
    result = categorical_restriction_mismatch(coarse, fine)
    assert result["restriction_consistent"]


def test_work_heat_integrals_use_physical_time_and_volume_density():
    history = dict(time_s=[0, 1, 2], stress_Pa=[2, 2, 2],
                   plastic_power_W_m3=[3, 3, 3],
                   deposited_heat_W_m3=[2.7, 2.7, 2.7],
                   mean_temperature_K=[300, 301, 302])
    result = integrate_work_heat(history, 4, 10, 300)
    assert result["external_work_J_m3"] == 16
    assert result["plastic_work_J_m3"] == 6
    assert result["deposited_heat_J_m3"] == 5.4
    assert result["stored_thermal_energy_change_J_m3"] == 20
