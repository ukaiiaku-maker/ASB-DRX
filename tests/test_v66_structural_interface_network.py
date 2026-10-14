import math

import numpy as np
import pytest

from full_model.production.structural_interface_network import (
    audit_network_compatibility,
    conservative_geometry_update,
    implied_misorientations_rad,
    initialize_structural_interfaces,
    pair_interface_length_m,
    interface_lengths_m,
    network_boundary_energy_J_per_m,
    network_junction_disclination_energy_J_per_m,
    JunctionDisclinationParameters,
    JunctionReactionParameters,
    relax_junction_cycle_interval,
    project_content_to_compatible_orientations,
)
from full_model.analysis.run_v66_multigrain_handoff import _initial
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


B = 2.48e-10


def test_two_owner_conservative_motion_preserves_content_and_L_theta():
    ids = (10, 20)
    old_length = np.array((8e-6,))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.0)), ((10, 20),), old_length, B)
    new_length = np.array((5e-6,))
    result = conservative_geometry_update(
        ids, interfaces, old_length, new_length, B)
    assert result.integrated_content_change[0] == 0.0
    assert (old_length[0]*result.old_misorientation_rad[0]
            == pytest.approx(new_length[0]*result.new_misorientation_rad[0]))
    assert result.new_misorientation_rad[0] == pytest.approx(
        math.radians(2.0)*8.0/5.0)
    assert result.compatibility.weighted_rms_compatibility_residual_rad < 1e-14
    assert result.energy_change_J_per_m < 0.0


def test_interface_sign_exchange_reverses_implied_misorientation():
    length = np.array((4e-6,))
    forward = initialize_structural_interfaces(
        (1, 2), (0.0, .03), ((1, 2),), length, B)
    reverse = initialize_structural_interfaces(
        (1, 2), (0.0, .03), ((2, 1),), length, B)
    assert implied_misorientations_rad(forward, length, B)[0] == pytest.approx(.03)
    assert implied_misorientations_rad(reverse, length, B)[0] == pytest.approx(-.03)


def test_common_geometry_scaling_preserves_cycle_compatibility():
    ids = (10, 20, 30)
    lengths = np.array((4e-6, 5e-6, 6e-6))
    edges = ((10, 20), (20, 30), (10, 30))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.0, 5.0)), edges, lengths, B)
    audit = audit_network_compatibility(ids, interfaces, .7*lengths, B)
    assert audit.weighted_rms_compatibility_residual_rad < 1e-14
    assert audit.maximum_cycle_residual_rad < 1e-14


def test_differential_geometry_exposes_frank_cycle_residual():
    ids = (10, 20, 30)
    lengths = np.array((4e-6, 5e-6, 6e-6))
    edges = ((10, 20), (20, 30), (10, 30))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.0, 5.0)), edges, lengths, B)
    changed = lengths.copy(); changed[0] *= .7
    audit = audit_network_compatibility(ids, interfaces, changed, B)
    assert audit.weighted_rms_compatibility_residual_rad > math.radians(.05)
    assert audit.maximum_cycle_residual_rad > math.radians(.05)


def test_cycle_projection_restores_compatible_owner_orientations_with_ledger():
    ids = (10, 20, 30)
    old_lengths = np.array((4e-6, 5e-6, 6e-6))
    edges = ((10, 20), (20, 30), (10, 30))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.0, 5.0)), edges, old_lengths, B)
    changed = old_lengths*np.array((.7, 1.1, .95))
    result = project_content_to_compatible_orientations(
        ids, interfaces, changed, B)
    assert result.before_compatibility.maximum_cycle_residual_rad > 0.0
    assert result.candidate_compatibility.maximum_cycle_residual_rad < 1e-14
    assert np.max(np.abs(result.content_balance_residual)) == 0.0
    assert np.max(np.abs(result.edge_content_change)) > 0.0
    assert result.delta_boundary_energy_J_per_m < 0.0


def test_cycle_projection_is_edge_direction_invariant():
    ids = (10, 20, 30)
    lengths = np.array((3e-6, 4e-6, 5e-6))
    edges = ((10, 20), (20, 30), (10, 30))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.0, 5.0)), edges, lengths, B)
    changed = lengths*np.array((1.2, .8, 1.1))
    forward = project_content_to_compatible_orientations(
        ids, interfaces, changed, B)
    reversed_interfaces = tuple(
        type(item)(item.interface_id, item.owner_b, item.owner_a,
                   -item.signed_integrated_content)
        for item in interfaces)
    reverse = project_content_to_compatible_orientations(
        ids, reversed_interfaces, changed, B)
    assert reverse.delta_boundary_energy_J_per_m == pytest.approx(
        forward.delta_boundary_energy_J_per_m, rel=1e-14)
    assert reverse.edge_content_change == pytest.approx(
        -forward.edge_content_change, rel=1e-13, abs=1e-12)
    assert reverse.candidate_compatibility.maximum_cycle_residual_rad < 1e-14


def test_junction_disclination_energy_is_zero_for_compatible_cycle_and_quadratic():
    ids = (10, 20, 30)
    lengths = np.array((4e-6, 5e-6, 6e-6))
    edges = ((10, 20), (20, 30), (10, 30))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.0, 5.0)), edges, lengths, B)
    p = JunctionDisclinationParameters(
        shear_modulus_Pa=70e9, poisson_ratio=.34, outer_cutoff_m=5e-7)
    assert network_junction_disclination_energy_J_per_m(
        ids, interfaces, lengths, B, p) == pytest.approx(0.0, abs=1e-32)
    changed = lengths.copy(); changed[0] *= .7
    e1 = network_junction_disclination_energy_J_per_m(
        ids, interfaces, changed, B, p)
    # Halving the closure defect by adjusting the redundant edge quarters E.
    audit = audit_network_compatibility(ids, interfaces, changed, B)
    closure = audit.fundamental_cycle_residual_rad[0]
    adjusted = list(interfaces)
    redundant = adjusted[2]
    theta = implied_misorientations_rad(adjusted, changed, B)[2]
    adjusted[2] = type(redundant)(
        redundant.interface_id, redundant.owner_a, redundant.owner_b,
        changed[2]*(theta+0.5*closure)/B)
    e2 = network_junction_disclination_energy_J_per_m(
        ids, tuple(adjusted), changed, B, p)
    assert e1 > 0.0
    assert e2 == pytest.approx(.25*e1, rel=1e-12)


def test_arrhenius_cycle_reaction_is_downhill_heat_complete_and_reduces_closure():
    ids = (10, 20, 30)
    old_lengths = np.array((4e-6, 5e-6, 6e-6))
    changed = old_lengths*np.array((.7, 1.1, .95))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.0, 5.0)),
        ((10, 20), (20, 30), (10, 30)), old_lengths, B)
    before = audit_network_compatibility(ids, interfaces, changed, B)
    ledger = relax_junction_cycle_interval(
        ids, interfaces, changed, B, temperature_K=1000.0, dt_s=1e-3,
        represented_thickness_m=2*B,
        reaction_parameters=JunctionReactionParameters(
            maximum_fraction_per_interval=.1))
    after = audit_network_compatibility(
        ids, ledger.committed_interfaces, changed, B)
    assert ledger.accepted
    assert ledger.reacted_fraction == pytest.approx(.1)
    assert after.maximum_cycle_residual_rad == pytest.approx(
        .9*before.maximum_cycle_residual_rad, rel=1e-12)
    assert ledger.generated_heat_J == pytest.approx(-ledger.delta_helmholtz_J)
    assert ledger.absolute_reacted_content > 0.0


def test_arrhenius_cycle_reaction_rolls_back_uphill_boundary_projection():
    # Actual V66 prepared-handoff lengths: its least-squares compatible
    # boundary state is more expensive than the tiny cycle energy.
    ids = (10, 20, 30)
    old_lengths = np.array((3.961135857179654e-6,
                            3.961135857179654e-6,
                            4.208419290715075e-6))
    changed = np.array((3.961194833151073e-6,
                        3.961020965551564e-6,
                        4.2084570361496004e-6))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.5, 5.0)),
        ((10, 20), (10, 30), (20, 30)), old_lengths, B)
    ledger = relax_junction_cycle_interval(
        ids, interfaces, changed, B, temperature_K=1000.0, dt_s=1e-3,
        represented_thickness_m=2*B)
    assert not ledger.accepted
    assert ledger.classification == "REJECTED_UPHILL_JUNCTION_RELAXATION"
    assert ledger.committed_interfaces == interfaces
    assert ledger.generated_heat_J == 0.0
    assert np.max(np.abs(ledger.signed_edge_content_change)) == 0.0


def test_compatible_cycle_has_no_reaction_or_owner_allocation():
    ids = (10, 20, 30)
    lengths = np.array((4e-6, 5e-6, 6e-6))
    interfaces = initialize_structural_interfaces(
        ids, np.radians((0.0, 2.0, 5.0)),
        ((10, 20), (20, 30), (10, 30)), lengths, B)
    ledger = relax_junction_cycle_interval(
        ids, interfaces, lengths, B, temperature_K=1000.0, dt_s=1.0,
        represented_thickness_m=2*B)
    assert not ledger.accepted
    assert ledger.classification == "NO_CYCLE_REACTION"
    assert ledger.committed_interfaces == interfaces


def test_diffuse_pair_length_is_owner_swap_and_translation_invariant():
    n = 32
    spacing = 4.8e-6/n
    x = np.arange(n)[:, None]
    eta = .5+.49*np.cos(2*np.pi*x/n)*np.ones((1, n))
    supports = np.stack((eta, 1.0-eta))
    length = pair_interface_length_m(supports, 0, 1, spacing)
    swapped = pair_interface_length_m(supports, 1, 0, spacing)
    translated = pair_interface_length_m(
        np.roll(supports, 7, axis=1), 0, 1, spacing)
    assert length > 0.0
    assert swapped == pytest.approx(length, rel=1e-14)
    assert translated == pytest.approx(length, rel=1e-14)


def test_structural_interface_energy_is_a_complete_multigrain_component():
    n = 8
    length = 4.8e-6
    spacing = length/n
    thickness = 4.96e-10
    state, _, _ = _initial(n, length, 1000.0, 2e7)
    edges = ((10, 20), (10, 30), (20, 30))
    records = tuple(type("Edge", (), {"owner_a": a, "owner_b": b})()
                    for a, b in edges)
    lengths = interface_lengths_m(
        state.supports, records, state.grain_ids, spacing)
    interfaces = initialize_structural_interfaces(
        state.grain_ids, np.radians((0.0, 2.5, 5.0)), edges, lengths, B)
    wall = CommonWallParameters(spacing_m=spacing)
    energy = evaluate_complete_multigrain_energy(
        state, spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall, mean_strain=np.zeros((2, 2)),
        topologies=(), systems=bcc_four_family_systems(),
        reference_temperature_K=1000.0,
        structural_interfaces=interfaces, structural_burgers_m=B)
    expected = network_boundary_energy_J_per_m(
        interfaces, lengths, B)*thickness
    assert energy.structural_boundary_J == pytest.approx(expected, rel=1e-13)
    assert energy.helmholtz_J >= energy.structural_boundary_J
