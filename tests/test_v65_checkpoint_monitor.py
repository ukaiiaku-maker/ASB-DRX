from full_model.analysis.monitor_v65_checkpoints import nonnegative_softening


def test_softening_is_zero_at_roundoff_scale_new_peak():
    assert nonnegative_softening(100.0, 100.0 + 2e-14) == 0.0


def test_softening_reports_physical_stress_loss():
    assert nonnegative_softening(100.0, 75.0) == 0.25


def test_softening_is_defined_for_zero_peak():
    assert nonnegative_softening(0.0, 0.0) == 0.0
