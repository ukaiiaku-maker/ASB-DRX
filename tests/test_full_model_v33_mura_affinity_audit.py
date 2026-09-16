import pytest

from full_model.analysis.audit_v33_mura_affinity import energy_forms


def test_energy_forms_distinguish_raw_average_and_unit_thickness_total():
    result = energy_forms(80.0, 4, 0.5, 2.0)
    assert result["raw_sum_J_m3_cells"] == 80.0
    assert result["volume_average_J_m3"] == 20.0
    assert result["unit_thickness_total_J"] == 20.0
    assert result["raw_sum_rate_W_m3_cells"] == 40.0
    assert result["volume_average_rate_W_m3"] == 10.0
    assert result["unit_thickness_total_rate_W"] == 10.0


def test_energy_forms_preserve_absent_elastic_source():
    assert energy_forms(None, 4, 0.5, 2.0) is None


def test_energy_forms_rejects_zero_time_by_arithmetic():
    with pytest.raises(ZeroDivisionError):
        energy_forms(1.0, 4, 0.5, 0.0)
