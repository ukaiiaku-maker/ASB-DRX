import numpy as np

from full_model.analysis.analyze_v33_front_preterminal import (
    _field_metrics, _periodic_components)


def test_periodic_component_audit_merges_seam_filament_without_inventing_core():
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, [0, 1, 6, 7]] = True
    components = _periodic_components(mask)
    assert len(components) == 1
    assert components[0]["area_cells2"] == 4
    assert components[0]["inradius_cells"] == 1.0
    assert components[0]["touches_periodic_seam"]


def test_diagnostic_topology_queries_are_bitwise_read_only_and_energy_is_positive():
    n = 32
    x = np.broadcast_to(np.arange(n, dtype=float)[:, None], (n, n))
    phi = np.tanh((x-10.25)/2.0)
    eta = np.stack(((1.0-phi)/2.0, (1.0+phi)/2.0), axis=2)
    before = eta.copy()
    metrics = _field_metrics(
        eta, np.ones((n, n), dtype=bool), 1e-7, 5e-7, 5e6)
    assert np.array_equal(eta, before)
    assert metrics["topology_filter_audit"]["eta_bitwise_unchanged"]
    assert metrics["topology_filter_audit"][
        "cut_cell_fraction_bitwise_unchanged"]
    assert metrics["energy"]["gradient_J_per_m"] > 0.0
    assert metrics["energy"]["bulk_J_per_m"] > 0.0


def test_zero_amplitude_seam_filament_has_no_declared_pure_core():
    phi = -np.ones((16, 16), dtype=float)
    phi[7, :] = 1e-3
    components = _periodic_components(phi > 0.0)
    assert len(components) == 1
    component = components[0]
    assert component["area_cells2"] == 16
    assert component["inradius_cells"] == 1.0
    assert np.count_nonzero(component["mask"] & (phi >= .9)) == 0
