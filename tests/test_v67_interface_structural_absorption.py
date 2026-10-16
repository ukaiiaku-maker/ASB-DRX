from dataclasses import replace
import math

import numpy as np

from full_model.analysis.run_v66_structural_bicrystal_front import _state
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
    evaluate_interface_structural_absorption,
    project_interface_burgers_to_structural_content,
)
from full_model.production.multigrain_common_state import InterfaceComponentState
from full_model.production.multigrain_production import MultiGrainInterface
from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.structural_interface_network import (
    initialize_structural_interfaces, interface_lengths_m,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, spectral_derivatives,
)


def _fixture(vector_mode="normal"):
    n = 8
    length = 4.8e-6
    spacing = length/n
    thickness = 2*2.48e-10
    state, _, _ = _state(n, length, 1000.0, 2e7)
    angles = np.radians((0.0, 20.0))
    state = replace(state, owners=tuple(
        replace(owner, orientation_rad=np.full_like(owner.orientation_rad, angle))
        for owner, angle in zip(state.owners, angles)))
    edge = MultiGrainInterface("edge-10-20", 10, 20)
    lengths = interface_lengths_m(
        state.supports, (edge,), state.grain_ids, spacing)
    structural = initialize_structural_interfaces(
        state.grain_ids, angles, ((10, 20),), lengths, 2.48e-10)
    eta = state.supports[1]-state.supports[0]
    gx, gy = spectral_derivatives(eta, spacing)
    mag = np.sqrt(gx*gx+gy*gy)
    nx = np.divide(gx, mag, out=np.zeros_like(gx), where=mag > 0.0)
    ny = np.divide(gy, mag, out=np.zeros_like(gy), where=mag > 0.0)
    direction = (np.stack((nx, ny, np.zeros_like(nx)), axis=-1)
                 if vector_mode == "normal"
                 else np.stack((-ny, nx, np.zeros_like(nx)), axis=-1))
    burgers = np.zeros((n, n, 3, 3))
    burgers[..., 0, :] = 100.0*direction
    plus = np.full((n, n, 3, 4), 1e14)
    minus = np.full_like(plus, 1e14)
    component = InterfaceComponentState(
        edge.component_id, 10, 20, np.zeros((n, n)), np.zeros((n, n)),
        plus, minus, np.zeros_like(state.owners[0].junction_m2),
        boundary_burgers_m1=burgers)
    state = replace(state, interfaces=(component,))
    state.validate()
    wall = CommonWallParameters(spacing_m=spacing)
    energy = {
        "topologies": (), "systems": bcc_four_family_systems(),
        "phase_barrier_J_m3": 0.0, "phase_gradient_J_m": 0.0,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": 1000.0,
    }
    return state, structural, spacing, thickness, wall, energy


def test_projection_closes_signed_content_vector_and_line_debit():
    state, structural, spacing, _, _, _ = _fixture()
    result = project_interface_burgers_to_structural_content(
        state, structural, spacing_m=spacing, fraction=.25,
        burgers_m=2.48e-10)
    assert result.structural_content_change[0] > 0.0
    assert result.removed_line_length_m > 0.0
    assert result.signed_content_balance == 0.0
    assert result.maximum_vector_balance_m1 < 1e-13
    before = state.interfaces[0]
    after = result.candidate_state.interfaces[0]
    removed_line = np.sum(
        before.boundary_plus_m2+before.boundary_minus_m2
        -after.boundary_plus_m2-after.boundary_minus_m2)*spacing**2
    np.testing.assert_allclose(
        removed_line, result.removed_line_length_m, rtol=1e-12, atol=0.0)


def test_tangential_burgers_control_does_not_rotate_or_debit_line():
    state, structural, spacing, _, _, _ = _fixture("tangent")
    result = project_interface_burgers_to_structural_content(
        state, structural, spacing_m=spacing, fraction=1.0,
        burgers_m=2.48e-10)
    assert abs(result.structural_content_change[0]) < 1e-12
    assert result.removed_line_length_m < 1e-12


def test_capped_hagb_absorption_is_arrhenius_energy_complete_and_label_safe():
    state, structural, spacing, thickness, wall, energy = _fixture()
    decision = evaluate_interface_structural_absorption(
        state, structural, mean_strain=np.zeros((2, 2)), interval_s=1e-3,
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, energy_kwargs=energy,
        parameters=InterfaceStructuralAbsorptionParameters(
            StructuralBoundaryParameters(burgers_m=wall.burgers_m), .10))
    assert decision.accepted
    assert decision.classification == "ACCEPTED_COMMON_AXIS_INTERFACE_ABSORPTION"
    assert decision.delta_helmholtz_J < 0.0
    assert decision.generated_heat_J > 0.0
    assert decision.relative_first_law_residual < 1e-5
    assert decision.committed_state.grain_ids == state.grain_ids
    np.testing.assert_array_equal(decision.committed_state.supports, state.supports)
    assert decision.committed_interfaces[0].signed_integrated_content != (
        structural[0].signed_integrated_content)


def test_complete_clock_absorbs_full_arrhenius_exposure_not_outer_step_cap():
    state, structural, spacing, thickness, wall, energy = _fixture()
    exact = evaluate_interface_structural_absorption(
        state, structural, mean_strain=np.zeros((2, 2)), interval_s=1e-3,
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, energy_kwargs=energy,
        parameters=InterfaceStructuralAbsorptionParameters(
            boundary=StructuralBoundaryParameters(burgers_m=wall.burgers_m),
            maximum_fraction_per_interval=.10,
            complete_physical_clock=True))
    assert exact.accepted
    assert exact.kinetic_fraction > .999
    assert exact.projection.signed_content_balance == 0.0
    assert exact.projection.maximum_vector_balance_m1 < 1e-12
    assert exact.relative_first_law_residual < 1e-5
