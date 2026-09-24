import numpy as np

from full_model.analysis.postprocess_v53_asb_mechanism import (
    field_metrics, overlap, periodic_components, physical_parameter_audit,
)


def test_periodic_edge_band_is_one_component():
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    components = periodic_components(mask)
    assert len(components) == 1
    assert np.count_nonzero(components[0]) == 16


def test_uniform_field_has_explicitly_undefined_width():
    metrics, component = field_metrics(np.ones((8, 8)), 1.0)
    assert metrics["undefined_localized_width"] is True
    assert metrics["second_moment_widths"] is None
    assert not np.any(component)


def test_component_overlap_tracks_same_support():
    left = np.zeros((5, 5), dtype=bool); left[:, 0] = True
    right = left.copy()
    assert overlap(left, right) == 1.0


def test_physical_audit_allows_only_causal_flag_and_runtime_restart():
    left = {"T0": 900.0, "k_thermal": .15,
            "causal_temperature_ablation": "none", "restart_file": "a"}
    right = {"T0": 900.0, "k_thermal": .15,
             "causal_temperature_ablation": "freeze_flow", "restart_file": "b"}
    assert physical_parameter_audit(left, right)["only_declared_intervention_differs"]
    right["k_thermal"] = .2
    assert not physical_parameter_audit(left, right)["only_declared_intervention_differs"]
