import numpy as np
import pytest

from full_model.analysis.postprocess_v32_asb_anchor import (
    effective_support,
    raw_conjunction,
    summarize_pair,
)
from full_model.production.asb_classifier import ASBSnapshot


def _snapshot(time_s, active, heating, softening, width=4.0e-7):
    return ASBSnapshot(
        time_s=time_s,
        active_fraction=active,
        temperature_excess_K=heating,
        softening_fraction=softening,
        effective_width_m=width,
    )


def test_effective_support_distinguishes_uniform_and_single_cell():
    uniform = effective_support(np.ones((4, 4)))
    localized = effective_support(np.eye(1, 16).reshape(4, 4))
    assert uniform["inverse_participation_fraction"] == 1.0
    assert uniform["entropy_effective_fraction"] == pytest.approx(1.0)
    assert localized["inverse_participation_fraction"] == pytest.approx(1.0/16.0)
    assert localized["entropy_effective_fraction"] == pytest.approx(1.0/16.0)


def test_raw_conjunction_requires_simultaneous_fixed_criteria():
    history = [
        _snapshot(0.0, 0.20, 60.0, 0.10),
        _snapshot(0.5e-6, 0.20, 60.0, 0.25),
        _snapshot(1.5e-6, 0.20, 60.0, 0.25),
    ]
    result = raw_conjunction(history, interface_width=1.0e-7)
    assert result["qualifying_snapshot_count"] == 2
    assert result["longest_conjunctive_persistence_s"] == pytest.approx(1.0e-6)


def test_incomplete_anchor_is_not_misclassified_as_mechanistic_negative():
    history = [
        _snapshot(0.0, 1.0, 0.0, 0.0),
        _snapshot(1.0e-6, 0.80, 100.0, 0.15),
    ]
    support = [effective_support(np.ones((2, 2))) for _ in history]
    result = summarize_pair(
        history, support, [0, 2700], 1.0e-7, target_steps=5000,
        strain_increment=1.0e-4,
    )
    assert result["complete"] is False
    assert result["diagnosis"] == "ANCHOR_INCOMPLETE_UNDEREXPOSURE_NOT_EXCLUDED"
    assert result["latest_nominal_strain"] == 0.2701
