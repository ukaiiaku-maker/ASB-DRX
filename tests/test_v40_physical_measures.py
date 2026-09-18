import json
from pathlib import Path

import pytest

from full_model.analysis.postprocess_v40_common_measures import snapshot


def test_uniform_density_sum_quadrature_is_grid_independent():
    length = 3.2e-6
    density = 3.25e5
    values = []
    for grid in (64, 128, 192):
        raw = density * grid * grid
        values.append((raw / grid**2, raw * (length / grid)**2))
    assert [row[0] for row in values] == pytest.approx([density] * 3)
    assert [row[1] for row in values] == pytest.approx(
        [density * length**2] * 3)


def test_snapshot_separates_run_length_from_selected_time():
    result = {
        "grid": 2,
        "completed_intervals": 2,
        "records": [
            {
                "physical_time_end_s": 1.0,
                "front_contour_displacement_m": 2.0,
                "front_signed_sweep_m3": 3.0,
                "front_absolute_sweep_m3": 4.0,
                "front_complete_energy_delta_J": 5.0,
                "mura_plastic_work_increment_J_m3_cells": 8.0,
                "mura_deposited_heat_increment_J_m3_cells": 12.0,
                "mura_stored_line_energy_increment_J_m3_cells": 4.0,
                "state_metrics": {
                    "temperature_minimum_K": 1101.0,
                    "temperature_maximum_K": 1102.0,
                },
            },
            {
                "physical_time_end_s": 2.0,
                "front_contour_displacement_m": 20.0,
                "front_signed_sweep_m3": 30.0,
                "front_absolute_sweep_m3": 40.0,
                "front_complete_energy_delta_J": 50.0,
                "mura_plastic_work_increment_J_m3_cells": 80.0,
                "mura_deposited_heat_increment_J_m3_cells": 120.0,
                "mura_stored_line_energy_increment_J_m3_cells": 40.0,
                "state_metrics": {
                    "temperature_minimum_K": 1110.0,
                    "temperature_maximum_K": 1120.0,
                },
            },
        ],
    }
    selected = snapshot(result, 0)
    assert selected["simulation_interval_count"] == 2
    assert selected["selected_snapshot_index"] == 0
    assert selected["selected_snapshot_time_s"] == 1.0
    assert selected["energy_density_sum_quadrature"][
        "mura_plastic_work_increment_J_m3_cells"]["volume_mean_J_m3"] == 2.0
    assert selected["temperature"]["maximum_rise_K"] == 2.0
