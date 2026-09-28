import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters,
)
from full_model.production.multigrain_common_state import PhysicalTransferLaw
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainInterface,
    MultiGrainProductionRuntime, advance_multigrain_front,
    advance_multigrain_mechanics,
)
from full_model.production.tensorial_nye import bcc_four_family_systems
from tests.test_full_model_v58_complete_multigrain_energy import _state


def _front_parameters(spacing):
    wall = CommonWallParameters(spacing_m=spacing, elastic_iterations=1)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("multi-grain-front", 1e8, 0.0, 1e8),
        PhysicalTransferLaw(.5, .1, .05), .35*EV_J, 1e9, 2.0, 1.5,
        .1, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=1e-3, maximum_fraction_per_step=.02,
        closure_fraction=.05)
    return wall, kinetics


def test_zero_pressure_two_boundary_production_step_is_complete_and_joint():
    spacing = 2e-8
    state = _state(16)
    wall, kinetics = _front_parameters(spacing)
    runtime = MultiGrainProductionRuntime((
        MultiGrainInterface("10-20", 10, 20),
        MultiGrainInterface("10-30", 10, 30)))
    evolved, runtime, decision = advance_multigrain_front(
        state, runtime, kinetics=kinetics, dt_s=1e-7,
        spacing_m=spacing, represented_thickness_m=5e-10,
        wall_parameters=wall, energy_kwargs=dict(
            phase_barrier_J_m3=5e6, phase_gradient_J_m=5e-7,
            reference_temperature_K=900.0),
        systems=bcc_four_family_systems())
    assert decision.accepted
    assert decision.energy_decision.complete_functional
    assert decision.energy_decision.external_work_J == 0.0
    assert len(decision.accepted_fraction_by_interface) == 2
    assert runtime.ledger.accepted_events == 1
    assert runtime.ledger.generated_heat_J > 0.0
    assert runtime.ledger.maximum_relative_energy_closure < .05
    assert evolved.ledger.energy_accepted_transactions == 1


def test_mechanical_step_evolves_supported_owner_and_preserves_dormant_history():
    spacing = 2e-8
    state = _state(12)
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        mobile_correlation_diffusivity_m2_s=0.0,
        thermal_diffusivity_m2_s=1e-7)
    stress = np.full((12, 12, 4), 8e8)
    dormant_before = state.owners[2].slip[:6].copy()
    evolved, decision = advance_multigrain_mechanics(
        state, driving=CommonWallDriving(resolved_stress_Pa=stress),
        systems=systems, topologies=(), wall_parameters=wall,
        dt_s=1e-12, represented_thickness_m=5e-10)
    assert decision.accepted
    assert decision.minimum_step_scale > 0.0
    assert decision.external_plastic_work_J >= 0.0
    # Grain 30 has no support in the upper half and cannot age while absent.
    np.testing.assert_array_equal(evolved.owners[2].slip[:6], dormant_before)
    assert np.max(np.abs(evolved.owners[0].slip-state.owners[0].slip)) > 0.0
    for owner_state in evolved.owners[1:]:
        np.testing.assert_array_equal(
            owner_state.temperature_K, evolved.owners[0].temperature_K)
