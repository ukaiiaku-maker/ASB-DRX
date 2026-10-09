import numpy as np

from full_model.analysis.audit_v63_phase_sensitive_fields import (
    common_quadrature, comparison, normalized_coefficients,
)


def test_complex_comparison_detects_translation_and_sign_hidden_by_power():
    n = 48
    x, y = np.meshgrid(np.arange(n)/n, np.arange(n)/n, indexing="ij")
    field = np.sin(2*np.pi*(3*x+2*y))+0.4*np.cos(2*np.pi*(x-4*y))
    shifted = np.roll(field, 7, axis=0)
    translated = comparison(field, shifted, (7,), 96)
    reversed_sign = comparison(field, -field, (7,), 96)
    assert translated["radial_power_total_variation"] < 1e-14
    assert translated["bands"]["7"]["complex_coefficient_relative_l2"] > .5
    assert reversed_sign["radial_power_total_variation"] < 1e-14
    assert reversed_sign["bands"]["7"]["complex_coefficient_relative_l2"] > 1.9


def test_common_quadrature_preserves_mean_and_retained_mode():
    n = 48
    x, y = np.meshgrid(np.arange(n)/n, np.arange(n)/n, indexing="ij")
    field = 2.5+np.sin(2*np.pi*(5*x-3*y))
    coefficients = normalized_coefficients(field)
    band = coefficients[n//2-7:n//2+8, n//2-7:n//2+8]
    projected = common_quadrature(band, 96)
    assert abs(projected.mean()-field.mean()) < 1e-13
    assert abs(projected.max()-field.max()) < 2e-2


def test_identical_physical_modes_on_different_grids_compare_exactly():
    fields = []
    for n in (48, 64):
        x, y = np.meshgrid(np.arange(n)/n, np.arange(n)/n, indexing="ij")
        fields.append(np.stack((np.sin(2*np.pi*(2*x+y)),
                                np.cos(2*np.pi*(3*x-2*y))), axis=2))
    result = comparison(fields[0], fields[1], (7, 15, 23), 96)
    for row in result["bands"].values():
        assert row["complex_coefficient_relative_l2"] < 2e-14
        assert row["physical_projection_relative_l2"] < 2e-14
