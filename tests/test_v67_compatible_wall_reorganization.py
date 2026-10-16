from dataclasses import replace

import numpy as np

from full_model.production.compatible_wall_reorganization import (
    CompatibleWallReorganizationParameters,
    evaluate_compatible_wall_reorganization,
)
from full_model.production.extensive_wall import (
    orientation_gradient_frank_bilby_target_m1,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from full_model.production.tensorial_nye import rotated_system_fields
from tests.test_full_model_common_tensorial_wall import fixture


def test_compatible_wall_glide_is_downhill_and_conserves_every_sign_inventory():
    systems, topologies, parameters, owner, _ = fixture(8)
    spacing = parameters.spacing_m
    coordinate = np.arange(8)[:, None]
    orientation = (.02*np.sin(2*np.pi*coordinate/8)*np.ones((1, 8)))
    closure = orientation_gradient_frank_bilby_target_m1(
        orientation, spacing)[..., :, 2]
    burgers = rotated_system_fields(systems, orientation)[0]
    signed = np.einsum(
        "...ai,...i->...a",
        np.linalg.pinv(np.swapaxes(burgers, -2, -1)), closure)
    # Translate the signs away from their compatible positions while keeping
    # each family's global Burgers inventory exactly unchanged.
    signed = np.roll(signed, 2, axis=0)
    total = np.abs(signed)+2.0e14
    owner = replace(
        owner, wall_plus_m2=.5*(total+signed),
        wall_minus_m2=.5*(total-signed),
        wall_order=np.ones(owner.wall_order.shape),
        orientation_rad=orientation,
        temperature_K=np.full(owner.temperature_K.shape, 1000.0))
    state = MultiGrainCommonState(
        (1,), np.ones((1,)+orientation.shape), (owner,))
    parameters = replace(
        parameters,
        wall_nye_match_coefficient_J_m=parameters.c44_Pa*spacing**2)
    decision = evaluate_compatible_wall_reorganization(
        state, interval_s=2.0e-5, mean_strain=np.zeros((2, 2)),
        spacing_m=spacing, represented_thickness_m=5.0e-10,
        systems=systems, wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems})
    assert decision.accepted
    assert decision.owner_fraction_by_id[1] > .99
    assert decision.delta_helmholtz_J < 0.0
    assert decision.relative_first_law_residual < 1.0e-10
    inventory_scale = float(np.sum(np.abs(signed)))
    assert decision.signed_content_residual_m2/inventory_scale < 1.0e-14
    assert decision.line_balance_residual_m2 == 0.0
    np.testing.assert_array_equal(
        decision.committed_state.owners[0].orientation_rad,
        state.owners[0].orientation_rad)
    np.testing.assert_array_equal(
        decision.committed_state.owners[0].wall_order,
        state.owners[0].wall_order)


def test_reaction_cone_conserves_resultant_burgers_vector_and_wall_line():
    systems, topologies, parameters, owner, _ = fixture(8)
    spacing = parameters.spacing_m
    coordinate = np.arange(8)[:, None]
    orientation = (.025*np.sin(2*np.pi*coordinate/8)*np.ones((1, 8)))
    closure = orientation_gradient_frank_bilby_target_m1(
        orientation, spacing)[..., :, 2]
    burgers = rotated_system_fields(systems, orientation)[0]
    signed = np.einsum(
        "...ai,...i->...a",
        np.linalg.pinv(np.swapaxes(burgers, -2, -1)), closure)
    # A spatially shifted distribution has the same family inventories but is
    # incompatible with the current orientation.  The broader reaction cone
    # is allowed to exchange family labels while preserving their resultant.
    signed = np.roll(signed, 2, axis=0)
    total = np.abs(signed)+2.0e14
    owner = replace(
        owner, wall_plus_m2=.5*(total+signed),
        wall_minus_m2=.5*(total-signed),
        wall_order=np.ones(owner.wall_order.shape),
        orientation_rad=orientation,
        temperature_K=np.full(owner.temperature_K.shape, 1000.0))
    state = MultiGrainCommonState(
        (1,), np.ones((1,)+orientation.shape), (owner,))
    parameters = replace(
        parameters,
        wall_nye_match_coefficient_J_m=parameters.c44_Pa*spacing**2)
    decision = evaluate_compatible_wall_reorganization(
        state, interval_s=2.0e-5, mean_strain=np.zeros((2, 2)),
        spacing_m=spacing, represented_thickness_m=5.0e-10,
        systems=systems, wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems},
        parameters=CompatibleWallReorganizationParameters(
            conservation_mode="burgers_vector"))
    assert decision.accepted
    assert decision.delta_helmholtz_J < 0.0
    inventory_scale = float(np.sum(np.abs(signed)))
    assert decision.signed_content_residual_m2/inventory_scale < 1.0e-11
    assert decision.line_balance_residual_m2 == 0.0


def test_reaction_cone_conserves_support_weighted_multigrain_inventory():
    systems, topologies, parameters, owner, _ = fixture(8)
    spacing = parameters.spacing_m
    coordinate = np.arange(8)[:, None]
    orientation = .025*np.sin(2*np.pi*coordinate/8)*np.ones((1, 8))
    closure = orientation_gradient_frank_bilby_target_m1(
        orientation, spacing)[..., :, 2]
    burgers = rotated_system_fields(systems, orientation)[0]
    signed = np.einsum(
        "...ai,...i->...a",
        np.linalg.pinv(np.swapaxes(burgers, -2, -1)), closure)
    signed = np.roll(signed, 2, axis=0)
    total = np.abs(signed)+2.0e14
    owner = replace(
        owner, wall_plus_m2=.5*(total+signed),
        wall_minus_m2=.5*(total-signed),
        wall_order=np.ones(owner.wall_order.shape),
        orientation_rad=orientation,
        temperature_K=np.full(owner.temperature_K.shape, 1000.0))
    # Every cell is in the owner's pure-enough reaction domain, but its
    # mixture weight varies.  An unweighted projection would pass a
    # single-owner test while changing the physical eta*rho inventory here.
    support = (.76+.24*np.arange(8)[:, None]/7.0)*np.ones((1, 8))
    state = MultiGrainCommonState(
        (1, 2), np.stack((support, 1.0-support)), (owner, owner))
    parameters = replace(
        parameters,
        wall_nye_match_coefficient_J_m=parameters.c44_Pa*spacing**2)
    decision = evaluate_compatible_wall_reorganization(
        state, interval_s=2.0e-5, mean_strain=np.zeros((2, 2)),
        spacing_m=spacing, represented_thickness_m=5.0e-10,
        systems=systems, wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems},
        parameters=CompatibleWallReorganizationParameters(
            conservation_mode="burgers_vector"))
    assert decision.accepted
    assert decision.signed_content_residual_m2 < 1e-10*float(np.sum(total))
    assert abs(decision.line_balance_residual_m2) < 1e-12*float(np.sum(total))

    direction = burgers/np.linalg.norm(burgers, axis=-1, keepdims=True)
    before = np.sum(
        support[..., None, None]*direction*signed[..., None],
        axis=(0, 1, 2), dtype=np.longdouble)
    changed = decision.committed_state.owners[0]
    after_signed = changed.wall_plus_m2-changed.wall_minus_m2
    after = np.sum(
        support[..., None, None]*direction*after_signed[..., None],
        axis=(0, 1, 2), dtype=np.longdouble)
    np.testing.assert_allclose(after, before, rtol=2e-11, atol=32.0)
