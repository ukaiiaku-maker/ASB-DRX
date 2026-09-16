from full_model.analysis.run_v34_mura_affinity_forks import energy


def test_v34_energy_reports_raw_average_and_unit_thickness_total():
    result = energy(80.0, 4, 0.5, 2.0)
    assert result["raw_sum_J_m3_cells"] == 80.0
    assert result["volume_average_J_m3"] == 20.0
    assert result["unit_thickness_total_J"] == 20.0
    assert result["volume_average_rate_W_m3"] == 10.0


def test_external_constraint_energy_can_be_reported_without_rate():
    result = energy(-12.0, 4, 0.5)
    assert result == {
        "raw_sum_J_m3_cells": -12.0,
        "volume_average_J_m3": -3.0,
        "unit_thickness_total_J": -3.0,
    }
