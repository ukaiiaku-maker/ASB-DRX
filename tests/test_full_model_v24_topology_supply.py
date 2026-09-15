from dataclasses import replace

import numpy as np
import pytest

from full_model.production.density_state_map import DensityInventory
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)
from full_model.production.wall_topology_supply import (
    accepted_junction_topology_step, accepted_topology_ordering,
    accepted_transport_capture,
    aligned_state_from_directions, alignment_checkpoint_arrays,
    alignment_from_checkpoint_arrays, conservative_transport_capture_step,
    maximum_ledger_residual,
    reservoir_nye_m1, zero_alignment_state,
)


def fixture(n=7):
    shape = (n, n, 4); z = np.zeros(shape)
    inventory = DensityInventory(
        np.full(shape, 4e13), np.full(shape, 3e13), z.copy(), z.copy(),
        np.full(shape, 2e13), np.full(shape, 1e13), z.copy(), z.copy(),
        np.zeros((n, n, 0)))
    systems = bcc_four_family_systems()
    directions = np.zeros(shape+(3,))
    directions[..., 2] = 1.0
    alignment = aligned_state_from_directions(inventory, directions)
    orientation = np.zeros((n, n))
    return inventory, alignment, systems, orientation


def test_capture_moves_signed_line_alignment_and_closes_scalar_burgers_and_nye():
    inventory, alignment, systems, orientation = fixture()
    requested_plus = np.full(inventory.mobile_plus_m2.shape, 1.5e13)
    requested_minus = np.full(inventory.mobile_minus_m2.shape, 8e13)
    updated, aligned, ledger = accepted_transport_capture(
        inventory, alignment, requested_plus, requested_minus,
        systems, orientation)
    np.testing.assert_allclose(updated.mobile_plus_m2, 2.5e13)
    np.testing.assert_allclose(updated.wall_tangle_plus_m2, 3.5e13)
    # Availability limiting is sign/family local.
    np.testing.assert_allclose(updated.mobile_minus_m2, 0.0)
    np.testing.assert_allclose(updated.wall_tangle_minus_m2, 4e13)
    residual = maximum_ledger_residual(ledger)
    assert residual["scalar_line_m2"] == 0.0
    assert residual["alignment_m2"] == 0.0
    assert residual["total_nye_m1"] < 2e-12
    before = reservoir_nye_m1(alignment, systems, orientation)["total"]
    after = reservoir_nye_m1(aligned, systems, orientation)["total"]
    np.testing.assert_allclose(after, before, rtol=4e-16, atol=2e-12)


def test_topology_ordering_cannot_create_alignment_from_unpolarized_tangle():
    inventory, _, systems, orientation = fixture()
    alignment = zero_alignment_state(inventory, len(systems))
    request = np.full(inventory.wall_tangle_plus_m2.shape, 1e13)
    updated, aligned, ledger = accepted_topology_ordering(
        inventory, alignment, request, request, systems, orientation)
    assert np.max(np.abs(reservoir_nye_m1(
        aligned, systems, orientation)["wall_ordered"])) == 0.0
    np.testing.assert_allclose(updated.wall_ordered_plus_m2, 1e13)
    assert max(maximum_ledger_residual(ledger).values()) == 0.0


def test_alignment_bound_rejects_scalar_only_or_overpolarized_state():
    inventory, alignment, systems, _ = fixture()
    invalid = replace(
        alignment,
        wall_tangle_plus_m2=2.0*inventory.wall_tangle_plus_m2[..., None]
                            *np.ones((1, 1, 1, 3)))
    with pytest.raises(ValueError, match="alignment magnitude exceeds"):
        invalid.validate(inventory, len(systems))


def test_alignment_restart_is_exact_and_partial_restart_is_rejected():
    inventory, alignment, systems, _ = fixture()
    payload = alignment_checkpoint_arrays(alignment)
    restored = alignment_from_checkpoint_arrays(
        payload, inventory, len(systems))
    for name in alignment.__dict__:
        np.testing.assert_array_equal(getattr(restored, name),
                                      getattr(alignment, name))
    payload.pop("v24_alignment__mobile_plus_m2")
    with pytest.raises(ValueError, match="incomplete V24"):
        alignment_from_checkpoint_arrays(payload, inventory, len(systems))


def test_packet_crossing_declared_trap_is_captured_with_its_line_direction():
    inventory, _, systems, orientation = fixture(n=8)
    z = np.zeros_like(inventory.mobile_plus_m2)
    mobile = z.copy(); mobile[2, 3, 0] = 5e13
    inventory = replace(
        inventory, mobile_plus_m2=mobile, mobile_minus_m2=z.copy(),
        wall_tangle_plus_m2=z.copy(), wall_tangle_minus_m2=z.copy())
    direction = np.zeros(mobile.shape+(3,)); direction[..., 1] = 1.0
    alignment = aligned_state_from_directions(inventory, direction)
    velocity_plus = np.zeros(mobile.shape+(2,)); velocity_plus[..., 0] = 2.0
    velocity_minus = np.zeros_like(velocity_plus)
    trap = np.zeros((8, 8), bool); trap[3, :] = True
    updated, aligned, ledger = conservative_transport_capture_step(
        inventory, alignment, velocity_plus, velocity_minus, trap,
        systems, orientation, spacing_m=1e-7, dt_s=5e-8)
    assert updated.mobile_plus_m2[2, 3, 0] == 0.0
    assert updated.wall_tangle_plus_m2[3, 3, 0] == 5e13
    np.testing.assert_array_equal(
        aligned.wall_tangle_plus_m2[3, 3, 0], [0.0, 5e13, 0.0])
    assert ledger["sign"]["plus"]["global_scalar_residual_line_per_thickness"] == 0.0
    assert np.max(np.abs(ledger["global_nye_integral_residual_m"])) < 1e-25


def test_transport_rejects_super_cfl_before_mutating_state():
    inventory, alignment, systems, orientation = fixture(n=4)
    velocity = np.zeros(inventory.mobile_plus_m2.shape+(2,))
    velocity[..., 0] = 2.1
    with pytest.raises(ValueError, match="CFL"):
        conservative_transport_capture_step(
            inventory, alignment, velocity, -velocity, np.zeros((4, 4), bool),
            systems, orientation, spacing_m=1e-7, dt_s=5e-8)


def test_explicit_junction_route_closes_frank_node_and_declared_nye_source():
    systems = bcc_four_family_systems()
    topology = make_junction_topology(
        systems, 0, 1, sign_a=1, sign_b=-1, line_tension_J_m=1e-9)
    n = 5; shape = (n, n, 4); z = np.zeros(shape)
    plus = z.copy(); minus = z.copy()
    plus[..., 0] = 3e13; minus[..., 1] = 4e13
    inventory = DensityInventory(
        z.copy(), z.copy(), z.copy(), z.copy(), plus, minus,
        z.copy(), z.copy(), np.zeros((n, n, 1)))
    directions = np.zeros(shape+(3,)); directions[..., 2] = 1.0
    directions[..., 0, :] = topology.parent_line_directions[0]
    directions[..., 1, :] = topology.parent_line_directions[1]
    alignment = aligned_state_from_directions(inventory, directions)
    request = np.full((n, n, 1), 2e13)
    updated, aligned, ledger = accepted_junction_topology_step(
        inventory, alignment, request, systems, (topology,),
        np.zeros((n, n)), dt_s=1e-6)
    np.testing.assert_allclose(updated.wall_tangle_plus_m2[..., 0], 1e13)
    np.testing.assert_allclose(updated.wall_tangle_minus_m2[..., 1], 2e13)
    np.testing.assert_allclose(updated.junction_m2[..., 0], 2e13)
    assert np.max(np.abs(ledger["scalar_line_balance_residual_m2"])) < .02
    assert np.max(np.abs(ledger["vector_burgers_balance_residual_m1"])) < 1e-12
    assert ledger["frank_and_node_closure"][0]["frank_rule_residual_m"] < 1e-25
    assert ledger["frank_and_node_closure"][0]["line_node_residual"] < 1e-14
    # Reorientation is allowed only because its tensorial source is explicit.
    assert np.max(np.abs(ledger["R_topology_m1_s"])) > 0.0
    assert aligned.junction_alignment_m2.shape == (n, n, 1, 3)
