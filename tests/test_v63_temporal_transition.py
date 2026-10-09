import pytest

from full_model.analysis.run_v58_three_grain_production import (
    _validate_v63_temporal_transition,
)


PARENT = "1c0cb4ca0e8a9a30b0c1a320817a0b35b4d14499"


def test_v63_temporal_transition_requires_exact_ratio_and_same_configuration():
    parent = {"n": 64, "dt_s": 6.25e-9, "case": "baseline"}
    half = {"n": 64, "dt_s": 3.125e-9, "case": "baseline"}
    dt, source = _validate_v63_temporal_transition(
        parent, half, {"source_commit": PARENT}, 2)
    assert dt == 6.25e-9
    assert source == PARENT

    eighth = {"n": 64, "dt_s": 0.78125e-9, "case": "baseline"}
    dt, source = _validate_v63_temporal_transition(
        parent, eighth, {"source_commit": PARENT}, 8)
    assert dt == 6.25e-9
    assert source == PARENT


def test_v63_temporal_transition_rejects_configuration_or_lineage_change():
    parent = {"n": 64, "dt_s": 6.25e-9, "case": "baseline"}
    with pytest.raises(ValueError):
        _validate_v63_temporal_transition(
            parent, {"n": 48, "dt_s": 3.125e-9, "case": "baseline"},
            {"source_commit": PARENT}, 2)
    with pytest.raises(ValueError):
        _validate_v63_temporal_transition(
            parent, {"n": 64, "dt_s": 3.125e-9, "case": "baseline"},
            {"source_commit": "different"}, 2)
