from dataclasses import replace

import numpy as np

from full_model.analysis.run_v66_structural_bicrystal_front import _state
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters,
    evaluate_lattice_interface_absorption,
    project_lattice_line_to_interfaces,
)
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
    evaluate_complete_multigrain_local_fields,
)
from full_model.production.multigrain_common_state import InterfaceComponentState
from full_model.production.multigrain_production import MultiGrainInterface
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, nye_from_plastic_distortion,
)


def _fixture(n=16):
    length = 4.8e-6
    state, _, _ = _state(n, length, 1000.0, 2e7)
    shape = state.owners[0].mobile_plus_m2.shape
    component = InterfaceComponentState(
        MultiGrainInterface("edge-10-20", 10, 20).component_id, 10, 20,
        np.zeros((n, n)), np.zeros((n, n)),
        np.zeros((n, n, 3, shape[-1])),
        np.zeros((n, n, 3, shape[-1])),
        np.zeros_like(state.owners[0].junction_m2),
        boundary_burgers_m1=np.zeros((n, n, 3, 3)))
    state = replace(state, interfaces=(component,))
    return state, bcc_four_family_systems(), length/n


def test_one_flux_absorption_closes_line_beta_nye_and_interface_burgers():
    state, systems, spacing = _fixture()
    result = project_lattice_line_to_interfaces(
        state, systems=systems, spacing_m=spacing, fraction=.02,
        parameters=LatticeInterfaceAbsorptionParameters(
            maximum_fraction_per_interval=.05,
            capture_distance_m=spacing))
    assert result.removed_plus_line_per_thickness > 0.0
    np.testing.assert_allclose(
        result.removed_plus_line_per_thickness,
        result.interface_plus_line_per_thickness,
        rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(
        result.removed_minus_line_per_thickness,
        result.interface_minus_line_per_thickness,
        rtol=2e-14, atol=0.0)
    assert result.maximum_mura_relative_residual < 2e-11
    assert result.maximum_incident_capacity <= .02+1e-15
    assert np.max(result.capture_activity_m2) > 0.0
    assert np.max(result.capture_activity_m2) > 5*np.min(
        result.capture_activity_m2)
    for before, after in zip(state.owners, result.candidate_state.owners):
        initial_offset = (np.sum(before.family_nye_m1, axis=2)
                          -nye_from_plastic_distortion(before.beta_p, spacing))
        final_offset = (np.sum(after.family_nye_m1, axis=2)
                        -nye_from_plastic_distortion(after.beta_p, spacing))
        np.testing.assert_allclose(final_offset, initial_offset,
                                   rtol=0.0, atol=2e-7)


def test_zero_extent_is_exact_identity():
    state, systems, spacing = _fixture(8)
    result = project_lattice_line_to_interfaces(
        state, systems=systems, spacing_m=spacing, fraction=0.0,
        parameters=LatticeInterfaceAbsorptionParameters(
            capture_distance_m=spacing))
    for before, after in zip(state.owners, result.candidate_state.owners):
        for name in before.__dict__:
            np.testing.assert_array_equal(getattr(after, name), getattr(before, name))


def test_spatial_fraction_field_is_local_and_preserves_balances():
    state, systems, spacing = _fixture(16)
    fraction = np.zeros((1, 16, 16))
    fraction[:, 8:, :] = .02
    result = project_lattice_line_to_interfaces(
        state, systems=systems, spacing_m=spacing, fraction=fraction,
        parameters=LatticeInterfaceAbsorptionParameters(
            capture_distance_m=spacing))
    assert np.max(result.capture_activity_m2_by_interface[:, :8]) == 0.0
    assert np.max(result.capture_activity_m2_by_interface[:, 8:]) > 0.0
    np.testing.assert_allclose(
        result.removed_plus_line_per_thickness,
        result.interface_plus_line_per_thickness, rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(
        result.removed_minus_line_per_thickness,
        result.interface_minus_line_per_thickness, rtol=2e-14, atol=0.0)


def test_local_energy_fields_integrate_to_complete_functional_components():
    state, systems, spacing = _fixture(8)
    projected = project_lattice_line_to_interfaces(
        state, systems=systems, spacing_m=spacing, fraction=.02,
        parameters=LatticeInterfaceAbsorptionParameters(
            capture_distance_m=spacing)).candidate_state
    wall = CommonWallParameters(spacing_m=spacing)
    thickness = 1.0e-6
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, mean_strain=np.zeros((2, 2)),
        topologies=(), systems=systems)
    complete = evaluate_complete_multigrain_energy(projected, **common)
    local = evaluate_complete_multigrain_local_fields(
        projected, spacing_m=spacing, wall_parameters=wall,
        mean_strain=np.zeros((2, 2)), topologies=(), systems=systems)
    cell_volume = spacing**2*thickness
    for field, total in (
            (local.defect_storage_J_m3, complete.defect_storage_J),
            (local.signed_junction_storage_J_m3,
             complete.signed_junction_storage_J),
            (local.boundary_excess_J_m3, complete.boundary_excess_J),
            (local.recoverable_elastic_J_m3,
             complete.recoverable_elastic_J)):
        np.testing.assert_allclose(
            np.sum(field, dtype=np.longdouble)*cell_volume, total,
            rtol=2e-13, atol=1e-28)


def test_local_kinetic_evaluator_returns_spatial_fields_under_global_gate():
    state, systems, spacing = _fixture(8)
    wall = CommonWallParameters(spacing_m=spacing)
    result = evaluate_lattice_interface_absorption(
        state, mean_strain=np.array([[0.0, .02], [.02, 0.0]]),
        interval_s=1e-8, spacing_m=spacing,
        represented_thickness_m=1e-6, systems=systems,
        wall_parameters=wall,
        energy_kwargs={"topologies": (), "systems": systems},
        parameters=LatticeInterfaceAbsorptionParameters(
            capture_distance_m=spacing))
    component_id = state.interfaces[0].component_id
    assert result.local_kinetics_mode.startswith("LOCAL_COMPLETE_STORAGE")
    assert result.edge_fraction_field_by_interface[component_id].shape == (8, 8)
    assert result.edge_affinity_field_J_m_by_interface[component_id].shape == (8, 8)
    assert np.all(np.isfinite(
        result.edge_affinity_field_J_m_by_interface[component_id]))
    if result.accepted:
        assert result.delta_helmholtz_J <= 0.0
