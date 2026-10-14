from dataclasses import replace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import network_interfaces
from full_model.analysis.run_v66_multigrain_handoff import _initial
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.multigrain_common_state import PhysicalTransferLaw
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainInterface, MultiGrainProductionRuntime,
)
from full_model.production.structural_interface_network import (
    initialize_structural_interfaces, interface_lengths_m,
)
from full_model.production.structural_multigrain_runtime import (
    StructuralMultigrainRuntime, advance_structural_junction_interval,
    synchronize_structural_owner_orientations,
    StructuralFrontRuntime, advance_structural_front_interval,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def _fixture(scale):
    n = 8
    length = 4.8e-6
    spacing = length/n
    thickness = 2*2.48e-10
    state, _, _ = _initial(n, length, 1000.0, 2e7)
    angles = np.radians((0.0, 2.5, 5.0))
    state = replace(state, owners=tuple(
        replace(owner, orientation_rad=np.full_like(owner.orientation_rad, angle))
        for owner, angle in zip(state.owners, angles)))
    edges = tuple(network_interfaces(state))
    lengths = interface_lengths_m(
        state.supports, edges, state.grain_ids, spacing)
    interfaces = list(initialize_structural_interfaces(
        state.grain_ids, angles,
        tuple((edge.grain_a_id, edge.grain_b_id) for edge in edges),
        lengths, 2.48e-10))
    interfaces[0] = replace(
        interfaces[0], signed_integrated_content=(
            scale*interfaces[0].signed_integrated_content))
    wall = CommonWallParameters(spacing_m=spacing)
    energy = {
        "topologies": (), "systems": bcc_four_family_systems(),
        "reference_temperature_K": 1000.0,
        "mean_strain": np.zeros((2, 2)),
    }
    runtime = StructuralMultigrainRuntime(tuple(interfaces))
    state = synchronize_structural_owner_orientations(
        state, runtime, spacing_m=spacing)
    return state, runtime, spacing, thickness, wall, energy


def test_complete_runtime_accepts_downhill_reaction_and_deposits_exact_heat():
    state, runtime, spacing, thickness, wall, energy = _fixture(1.02)
    evolved, runtime, decision = advance_structural_junction_interval(
        state, runtime, dt_s=1e-3, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs=energy)
    assert decision.accepted
    assert decision.classification == "ACCEPTED_COMPLETE_JUNCTION_RELAXATION"
    assert decision.generated_heat_J > 0.0
    assert decision.relative_first_law_residual < 1e-6
    assert runtime.ledger.accepted_reactions == 1
    assert runtime.ledger.reacted_absolute_content > 0.0
    assert np.mean(evolved.owners[0].temperature_K) > 1000.0
    assert evolved.grain_ids == state.grain_ids


def test_complete_runtime_rolls_back_uphill_reaction():
    state, runtime, spacing, thickness, wall, energy = _fixture(.98)
    evolved, after, decision = advance_structural_junction_interval(
        state, runtime, dt_s=1e-3, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs=energy)
    assert not decision.accepted
    assert decision.classification == "REJECTED_COMPLETE_ENERGY_UPHILL"
    assert after.interfaces == runtime.interfaces
    assert after.ledger.rejected_reactions == 1
    assert decision.generated_heat_J == 0.0
    for old, new in zip(state.owners, evolved.owners):
        assert np.array_equal(old.temperature_K, new.temperature_K)


def test_uncapped_physical_interval_subdivision_agrees():
    state, runtime, spacing, thickness, wall, energy = _fixture(1.02)
    once_state, once_runtime, once = advance_structural_junction_interval(
        state, runtime, dt_s=1e-8, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs=energy)
    half_state, half_runtime, _ = advance_structural_junction_interval(
        state, runtime, dt_s=5e-9, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs=energy)
    twice_state, twice_runtime, twice = advance_structural_junction_interval(
        half_state, half_runtime, dt_s=5e-9, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs=energy)
    once_content = np.asarray([
        item.signed_integrated_content for item in once_runtime.interfaces])
    twice_content = np.asarray([
        item.signed_integrated_content for item in twice_runtime.interfaces])
    assert once.accepted and twice.accepted
    np.testing.assert_allclose(
        twice_content, once_content, rtol=3e-7, atol=1e-10)
    np.testing.assert_allclose(
        np.mean(twice_state.owners[0].temperature_K),
        np.mean(once_state.owners[0].temperature_K), rtol=3e-7)


def test_combined_bicrystal_front_derives_rotation_and_preserves_content():
    from full_model.analysis.run_v66_structural_bicrystal_front import _state

    n = 8
    length = 4.8e-6
    spacing = length/n
    thickness = 2*2.48e-10
    state, _, _ = _state(n, length, 1000.0, 2e7)
    wall = CommonWallParameters(spacing_m=spacing)
    edge = MultiGrainInterface("edge-10-20", 10, 20)
    lengths = interface_lengths_m(
        state.supports, (edge,), state.grain_ids, spacing)
    structural = initialize_structural_interfaces(
        state.grain_ids, np.radians((0.0, 2.5)), ((10, 20),),
        lengths, wall.burgers_m)
    runtime = StructuralFrontRuntime(
        MultiGrainProductionRuntime((edge,)),
        StructuralMultigrainRuntime(structural))
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("test structural front", 1e10, 0.0, 1e10),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4, maximum_fraction_per_step=.01,
        closure_fraction=.05, maximum_backtracks=14,
        temperature_resolution="global_mean")
    old_content = runtime.structural_runtime.interfaces[0].signed_integrated_content
    evolved, after, decision = advance_structural_front_interval(
        state, runtime, kinetics=kinetics, dt_s=2e-8,
        maximum_substep_s=2e-8, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        energy_kwargs={
            "topologies": (), "systems": bcc_four_family_systems(),
            "phase_barrier_J_m3": 0.0, "phase_gradient_J_m": 0.0,
            "boundary_line_energy_J_m": wall.line_energy_J_m,
            "boundary_junction_energy_J_m": wall.junction_energy_J_m,
            "reference_temperature_K": 1000.0,
            "mean_strain": np.zeros((2, 2)),
        }, systems=bcc_four_family_systems())
    assert decision.accepted
    assert abs(decision.orientation_sync_energy_change_J) < 1e-25
    assert (after.structural_runtime.interfaces[0].signed_integrated_content
            == old_content)
    new_length = interface_lengths_m(
        evolved.supports, (edge,), evolved.grain_ids, spacing)[0]
    theta = float(np.mean(evolved.owners[1].orientation_rad)
                  -np.mean(evolved.owners[0].orientation_rad))
    np.testing.assert_allclose(
        new_length*theta, wall.burgers_m*old_content, rtol=1e-13)
    assert decision.junction_decision.classification == "NO_CYCLE_REACTION"
