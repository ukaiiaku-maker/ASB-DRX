from dataclasses import replace

import numpy as np

from full_model.production.common_tensorial_wall import CommonWallState
from full_model.production.density_state_map import (
    DensityInventory, checkpoint_arrays, derived_density_fields,
    from_checkpoint_arrays, from_v22_common_state,
    legacy_scalar_views, line_energy_density_J_m3,
    taylor_resistance_Pa,
)
from full_model.production.extensive_wall import (
    ExtensiveWallParameters, accepted_ordering_step,
    accepted_ordering_step_jvp,
    extensive_wall_chemical_potentials_J_m,
    extensive_wall_energy_components_J_m3, ordered_wall_nye_m1,
    manufacture_ordered_inventory, ordering_residual,
    planar_frank_bilby_target_m1, wall_diagnostics,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, integrated_nye_closure, make_junction_topology,
)


def fixture(n=8):
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    shape = (n, n, 4)
    z = np.zeros(shape)
    inventory = DensityInventory(
        np.full(shape, 3e13), np.full(shape, 2e13),
        np.full(shape, 4e13), np.full(shape, 3e13),
        np.full(shape, 2e13), np.full(shape, 1e13),
        z.copy(), z.copy(), np.full((n, n, 1), 5e12))
    orientation = np.zeros((n, n))
    target = np.zeros((n, n, 3, 3))
    parameters = ExtensiveWallParameters(spacing_m=2e-7)
    return systems, topologies, inventory, orientation, target, parameters


def test_density_map_counts_every_line_once_and_preserves_units():
    systems, topologies, inventory, _, _, _ = fixture()
    fields = derived_density_fields(inventory, topologies)
    multiplicity = topologies[0].product_line_multiplicity
    expected = (4 * (3e13 + 2e13 + 4e13 + 3e13 + 2e13 + 1e13)
                + 5e12 * multiplicity)
    np.testing.assert_allclose(fields["rho_total_m2"], expected)
    np.testing.assert_allclose(
        line_energy_density_J_m3(inventory, topologies, 1.7e-9),
        1.7e-9 * expected)
    # Area integration returns line length per out-of-plane thickness.
    dx = 2e-7
    integrated = np.sum(fields["rho_total_m2"]) * dx * dx
    assert np.isclose(integrated, expected * inventory.mobile_plus_m2.shape[0]
                      * inventory.mobile_plus_m2.shape[1] * dx * dx)


def test_v22_restart_conversion_is_exact_and_q_is_only_a_partition():
    systems, topologies, inventory, orientation, _, _ = fixture(n=4)
    shape = inventory.mobile_plus_m2.shape
    wall_plus = np.full(shape, 7e13); wall_minus = np.full(shape, 5e13)
    old = CommonWallState(
        inventory.mobile_plus_m2, inventory.mobile_minus_m2,
        inventory.forest_plus_m2, inventory.forest_minus_m2,
        wall_plus, wall_minus, inventory.junction_m2,
        np.full(shape[:2], .3), np.zeros(shape[:2]), np.zeros(shape),
        np.zeros(shape[:2] + (3, 3)), np.zeros(shape + (3,)),
        np.zeros(shape + (3, 3)), orientation, np.full(shape[:2], 1100.0))
    new = from_v22_common_state(old)
    np.testing.assert_array_equal(
        new.wall_tangle_plus_m2 + new.wall_ordered_plus_m2, wall_plus)
    np.testing.assert_array_equal(
        new.wall_tangle_minus_m2 + new.wall_ordered_minus_m2, wall_minus)
    views = legacy_scalar_views(new, topologies)
    np.testing.assert_allclose(views["q_wall_v19"], .3)
    np.testing.assert_allclose(views["rho_wall"], np.sum(
        wall_plus + wall_minus, axis=2))


def test_extensive_checkpoint_is_bitwise_and_rejects_partial_state():
    systems, topologies, inventory, _, _, _ = fixture()
    payload = checkpoint_arrays(inventory)
    restored = from_checkpoint_arrays(payload, len(systems), len(topologies))
    for name in inventory.__dict__:
        np.testing.assert_array_equal(getattr(restored, name), getattr(inventory, name))
    payload.pop("v23_density__wall_ordered_minus_m2")
    try:
        from_checkpoint_arrays(payload, len(systems), len(topologies))
    except ValueError as exc:
        assert "incomplete V23" in str(exc)
    else:
        raise AssertionError("partial extensive restart was accepted")


def test_line_energy_and_taylor_reconstruction_have_physical_magnitude():
    systems, topologies, inventory, _, _, _ = fixture()
    tau = taylor_resistance_Pa(
        inventory, topologies, 116.5e9, systems[0].burgers_m,
        alpha=.3, wall_weight=2.0, junction_weight=1.0)
    assert np.all((tau > 1e7) & (tau < 2e9))


def test_extensive_chemical_potentials_are_exact_energy_derivatives():
    systems, topologies, inventory, orientation, target, parameters = fixture(n=6)
    inventory = replace(
        inventory,
        wall_ordered_plus_m2=np.full_like(inventory.wall_ordered_plus_m2, 1e12),
        wall_ordered_minus_m2=np.full_like(inventory.wall_ordered_minus_m2, 1e12))
    parameters = replace(parameters, ordered_gradient_J_m3=0.0)
    mu = extensive_wall_chemical_potentials_J_m(
        inventory, systems, topologies, orientation, target, parameters)
    h = 1e7
    for state_name, key in (
        ("wall_tangle_plus_m2", "tangle_plus"),
        ("wall_tangle_minus_m2", "tangle_minus"),
        ("wall_ordered_plus_m2", "ordered_plus"),
        ("wall_ordered_minus_m2", "ordered_minus"),
    ):
        delta = np.zeros_like(getattr(inventory, state_name)); delta[..., 0] = h
        ep = extensive_wall_energy_components_J_m3(
            replace(inventory, **{state_name: getattr(inventory, state_name) + delta}),
            systems, topologies, orientation, target, parameters)["total"]
        em = extensive_wall_energy_components_J_m3(
            replace(inventory, **{state_name: getattr(inventory, state_name) - delta}),
            systems, topologies, orientation, target, parameters)["total"]
        np.testing.assert_allclose((ep-em)/(2*h), mu[key][..., 0], rtol=2e-6,
                                   atol=1e-16)


def test_ordering_is_detailed_balanced_and_conserves_line_and_burgers():
    systems, topologies, inventory, orientation, target, parameters = fixture()
    # Supply target content that makes ordered plus line locally favorable.
    trial = replace(inventory, wall_ordered_plus_m2=np.full_like(
        inventory.wall_ordered_plus_m2, 5e13))
    target, _ = ordered_wall_nye_m1(trial, systems, orientation)
    before = derived_density_fields(inventory, topologies)["rho_total_m2"]
    updated, ledger, scale = accepted_ordering_step(
        inventory, systems, topologies, orientation, target,
        np.zeros(inventory.mobile_plus_m2.shape), np.full(orientation.shape, 1100.0),
        parameters, 1e-6)
    after = derived_density_fields(updated, topologies)["rho_total_m2"]
    np.testing.assert_allclose(after, before, rtol=0, atol=.25)
    for sign in ("plus", "minus"):
        np.testing.assert_allclose(
            getattr(updated, f"wall_tangle_{sign}_m2")
            + getattr(updated, f"wall_ordered_{sign}_m2"),
            getattr(inventory, f"wall_tangle_{sign}_m2")
            + getattr(inventory, f"wall_ordered_{sign}_m2"), rtol=0, atol=.25)
    assert 0 < scale <= 1
    assert np.max(ledger["free_energy_rate_W_m3"]) <= 1e-7
    transfer, turnover, mu = ordering_residual(
        inventory, systems, topologies, orientation, target,
        np.zeros(inventory.mobile_plus_m2.shape), np.full(orientation.shape, 1100.0),
        parameters)
    fwd = .5 * (turnover["plus"] + transfer["plus"])
    rev = .5 * (turnover["plus"] - transfer["plus"])
    # Use an interior state for a nonsingular gross reverse rate.
    interior = replace(inventory, wall_ordered_plus_m2=np.full_like(
        inventory.wall_ordered_plus_m2, 1e12))
    transfer, turnover, mu = ordering_residual(
        interior, systems, topologies, orientation, target,
        np.zeros(interior.mobile_plus_m2.shape), np.full(orientation.shape, 1100.0),
        parameters)
    fwd = .5 * (turnover["plus"] + transfer["plus"])
    rev = .5 * (turnover["plus"] - transfer["plus"])
    measured = fwd/rev
    expected = np.exp(np.clip(
        -(mu["ordered_plus"]-mu["tangle_plus"])*parameters.event_length_m
        /(1.380649e-23*1100.0), -700, 700))
    np.testing.assert_allclose(measured, expected, rtol=2e-10)


def test_accepted_extensive_map_jvp_includes_ordered_reservoirs():
    systems, topologies, inventory, orientation, target, parameters = fixture()
    inventory = replace(
        inventory,
        wall_ordered_plus_m2=np.full_like(inventory.wall_ordered_plus_m2, 1e12),
        wall_ordered_minus_m2=np.full_like(inventory.wall_ordered_minus_m2, 1e12))
    direction = replace(inventory, **{
        name: np.zeros_like(value) for name, value in inventory.__dict__.items()})
    perturb = np.zeros_like(inventory.wall_ordered_plus_m2); perturb[..., 0] = 1.0
    direction = replace(direction, wall_ordered_plus_m2=perturb,
                        wall_tangle_plus_m2=-perturb)
    jvp = accepted_ordering_step_jvp(
        inventory, direction, systems, topologies, orientation, target,
        np.zeros(inventory.mobile_plus_m2.shape), np.full(orientation.shape, 1100.0),
        parameters, 1e-9)
    assert np.max(np.abs(jvp.wall_ordered_plus_m2)) > 0
    np.testing.assert_allclose(jvp.wall_ordered_plus_m2
                               + jvp.wall_tangle_plus_m2, 0.0, atol=1e-10)


def test_wall_classifier_rejects_density_and_topology_false_positives():
    systems, topologies, inventory, orientation, target, _ = fixture()
    # No ordered line, high balanced ordered line, and high polarized line with
    # no orientation jump are all non-walls.
    assert not np.any(wall_diagnostics(
        inventory, systems, topologies, orientation, target)["physical_wall_mask"])
    balanced = replace(
        inventory,
        wall_ordered_plus_m2=np.full_like(inventory.wall_ordered_plus_m2, 1e14),
        wall_ordered_minus_m2=np.full_like(inventory.wall_ordered_minus_m2, 1e14))
    assert not np.any(wall_diagnostics(
        balanced, systems, topologies, orientation, target)["physical_wall_mask"])
    polarized = replace(
        inventory,
        wall_ordered_plus_m2=np.full_like(inventory.wall_ordered_plus_m2, 1e14))
    alpha, _ = ordered_wall_nye_m1(polarized, systems, orientation)
    assert not np.any(wall_diagnostics(
        polarized, systems, topologies, orientation, alpha)["physical_wall_mask"])


def test_manufactured_tilt_wall_and_controls_require_matching_nye_and_jump():
    systems, topologies, inventory, orientation, _, _ = fixture(n=16)
    x = np.arange(16)[:, None]
    angle = np.where(np.broadcast_to(x < 8, (16, 16)), 0.0, np.deg2rad(2.0))
    target, fb_closure = planar_frank_bilby_target_m1(
        (16, 16), 2e-7, 0.0, np.deg2rad(2.0), width_cells=3)
    wall, projection_residual = manufacture_ordered_inventory(
        inventory, systems, angle, target)
    assert projection_residual < 2e-14
    alpha, _ = ordered_wall_nye_m1(wall, systems, angle)
    integrated = integrated_nye_closure(alpha, 0, 2e-7, line_axis=2)
    np.testing.assert_allclose(integrated, fb_closure, rtol=2e-14, atol=2e-16)
    positive = wall_diagnostics(wall, systems, topologies, angle, target)
    assert np.any(positive["physical_wall_mask"])
    # Reversing line sign against the same Frank--Bilby target fails.
    reverse = replace(wall,
                      wall_ordered_plus_m2=wall.wall_ordered_minus_m2,
                      wall_ordered_minus_m2=wall.wall_ordered_plus_m2)
    assert not np.any(wall_diagnostics(
        reverse, systems, topologies, angle, target)["physical_wall_mask"])
    # An incompatible target line column fails even at identical norm.
    incompatible = np.roll(target, 1, axis=-1)
    assert not np.any(wall_diagnostics(
        wall, systems, topologies, angle, incompatible)["physical_wall_mask"])
    # A jointly rotated copy remains classified (objective scalar tests).
    rotated_angle = angle + .47
    rotated_target, _ = ordered_wall_nye_m1(wall, systems, rotated_angle)
    rotated = wall_diagnostics(
        wall, systems, topologies, rotated_angle, rotated_target)
    np.testing.assert_array_equal(rotated["physical_wall_mask"],
                                  positive["physical_wall_mask"])
