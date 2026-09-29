from full_model.analysis.postprocess_v59_network import network_topology_metrics
from full_model.analysis.run_v58_three_grain_production import initialize_network_state


def test_network_topology_metrics_distinguish_resolved_four_grain_network():
    state = initialize_network_state(
        32, 900.0, 4, spacing_m=5e-6/32,
        interface_width_m=3.125e-7)
    metrics = network_topology_metrics(state)
    assert metrics["declared_grain_count"] == 4
    assert metrics["resolved_core_count_support_gt_0p8"] == 4
    assert metrics["dominant_adjacency_count"] >= 4
    assert metrics["partition_maximum_absolute_error"] < 1e-14
