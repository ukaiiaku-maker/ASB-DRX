import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.dislocation_free_energy import DislocationFreeEnergyParameters
from full_model.production.predictive_signed_wall import (
    SignedWallParameters, SignedWallState, admissible_basis,
    advance_local_reactions, conservation_vectors,
    conservative_stoichiometric_matrix, linearized_operator,
    projected_hessian, state_vector,
)
from full_model.production.wall_ordering_energy import WallOrderingParameters


def parameters():
    rho_scale = 5.0e14
    line = 0.5*45.0e9*(2.48e-10)**2
    process = lambda name: ActivatedProcess(name, 1.0e7, 0.0, 1.0e8)
    return SignedWallParameters(
        rho_scale,
        DislocationFreeEnergyParameters(
            line, .06*line, 1.0e14, 1.0, rho_scale, 1.3, .3, .25),
        WallOrderingParameters(
            1.2*line*rho_scale*.3, rho_scale, 1.3, .3,
            .08*line*rho_scale, .25*line, .8*rho_scale,
            gradient_coefficient_J_m=2.0e-7),
        process("lock"), process("trap"), process("release"),
        process("annihilation"), process("order"),
        .9*EV_J, .8*EV_J, 1.1*EV_J, 1.2*EV_J, .7*EV_J,
        1.5e9, exp_floor=.05, neutral_capture_area_m2=1e-16,
        order_energy_scale_J_m3=1e6,
        density_gradient_J_m3_m2=1e-7, order_gradient_J_m=2e-7,
        compatibility_J_m=1e-8, diffusivity_m2_s=1e-12,
        orientation_mobility_m3_J_s=1e-5, plastic_spin_rate_s=10.0)


def homogeneous_state(families=4):
    total = 5.0e14
    fractions = (.10, .10, .20, .20, .20, .20)
    arrays = [np.full(families, f*total/families) for f in fractions]
    return SignedWallState(*arrays, np.asarray(.35))


def test_reaction_tangent_conserves_line_and_each_family_burgers_content():
    families = 4
    B = conservative_stoichiometric_matrix(families)
    total, signed = conservation_vectors(families)
    np.testing.assert_allclose(total@B, 0.0, atol=1e-15)
    np.testing.assert_allclose(signed@B, 0.0, atol=1e-15)
    Q = admissible_basis(families)
    np.testing.assert_allclose(Q.T@Q, np.eye(Q.shape[1]), atol=1e-14)


def test_projected_modes_report_accessibility_and_conservation():
    state = homogeneous_state()
    _, projected, modes = projected_hessian(
        state_vector(state)/np.r_[np.full(24, parameters().rho_scale_m2), 1.0],
        parameters())
    assert projected.shape == (17, 17)
    assert len(modes) == 17
    for mode in modes:
        assert abs(mode["total_line_change_normalized"]) < 2e-12
        assert max(abs(x) for x in mode["signed_change_by_family_normalized"]) < 2e-12
        assert mode["nonnegativity_admissible"]


def test_local_channels_are_nonnegative_and_ledger_annihilation_separately():
    p = parameters()
    state = homogeneous_state()
    # Favor mobile -> forest and forest -> wall in both signs.
    mu = np.vstack((np.full((2, 4), 4e-9),
                    np.full((2, 4), 2e-9),
                    np.zeros((2, 4))))
    updated, ledger = advance_local_reactions(
        state, mu, 8e8, 1100.0, 1e-5, p)
    updated.validate()
    assert ledger.locked_m2 > 0.0
    assert ledger.trapped_m2 > 0.0
    assert ledger.junctioned_m2 > 0.0
    assert ledger.neutral_annihilated_m2 >= 0.0
    np.testing.assert_allclose(ledger.total_line_change_m2,
                               -ledger.neutral_annihilated_m2,
                               rtol=1e-13, atol=1.0)
    np.testing.assert_allclose(ledger.signed_change_by_family_m2, 0.0,
                               atol=1.0)


def test_full_linearized_operator_has_orientation_and_opposite_signed_advection():
    p = parameters()
    state = homogeneous_state()
    scale = np.r_[np.full(24, p.rho_scale_m2), 1.0]
    x = state_vector(state)/scale
    velocities = np.array([[10., 0.], [0., 10.], [-10., 0.], [0., -10.]])
    L = linearized_operator(x, [2e6, 1e6], velocities, 8e8, 1100., p)
    assert L.shape == (26, 26)
    assert np.all(np.isfinite(L))
    # Positive and negative carriers of family zero have opposite phase speeds.
    assert np.imag(L[0, 0]) == -np.imag(L[4, 4])
    assert np.any(np.abs(L[-1, :-1]) > 0.0)
