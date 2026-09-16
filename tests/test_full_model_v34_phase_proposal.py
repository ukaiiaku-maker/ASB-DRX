import numpy as np

from full_model.production.phase_proposal import (
    adaptive_phase_proposal, preserve_existing_pair_components,
    project_simplex, spectral_phase_energy)


def _fixture(n=32, delta=1e-6):
    dx = 10e-6/n
    x = np.arange(n)[:, None]*dx
    phi = np.tanh((x-0.35*10e-6)/(0.45e-6))
    phi = np.broadcast_to(phi, (n, n)).copy()+delta
    return np.stack(((1.0-phi)/2.0, (1.0+phi)/2.0), axis=2), dx


def _operators(dx, kappa=5e-7, barrier=5e6):
    n = int(round(10e-6/dx))
    kx = 2*np.pi*np.fft.fftfreq(n, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(n, d=dx)
    k2 = kx[:, None]**2+ky[None, :]**2

    def force(eta):
        total_sq = np.sum(eta**2, axis=2)
        result = np.empty_like(eta)
        for i in range(eta.shape[2]):
            lap = np.real(np.fft.ifft2(-k2*np.fft.fft2(eta[:, :, i])))
            result[:, :, i] = (-kappa*lap
                               +2*barrier*eta[:, :, i]
                               *(total_sq-eta[:, :, i]**2))
        return result

    def energy(eta):
        return spectral_phase_energy(
            eta, spacing_m=dx, kappa_J_m=kappa,
            bulk_barrier_J_m3=barrier)
    return force, energy


def test_simplex_projection_is_bounded_and_conservative():
    values = np.array([[[-0.2, 0.3, 1.4], [0.2, 0.2, 0.2]]])
    result = project_simplex(values)
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0
    np.testing.assert_allclose(np.sum(result, axis=2), 1.0, atol=2e-15)


def test_adaptive_phase_proposal_resolves_high_diffusion_number_downhill():
    eta, dx = _fixture()
    force, energy = _operators(dx)
    result, diagnostics = adaptive_phase_proposal(
        eta, dt_s=1e-5, spacing_m=dx, mobility=3e-3,
        kappa_J_m=5e-7, force=force, energy=energy)
    assert diagnostics.accepted_substeps > 1
    assert diagnostics.integrated_dt_s == 1e-5
    assert diagnostics.energy_change_J_m <= 0.0
    assert diagnostics.simplex_sum_error < 2e-14
    assert diagnostics.simplex_minimum >= 0.0
    assert diagnostics.simplex_maximum <= 1.0
    assert np.max(np.abs(result-project_simplex(eta))) <= (
        diagnostics.maximum_abs_increment+1e-15)


def test_label_exchange_and_signed_perturbation_symmetry():
    eta_plus, dx = _fixture(delta=1e-6)
    eta_minus = eta_plus[:, :, ::-1].copy()
    force, energy = _operators(dx)
    kwargs = dict(
        dt_s=2e-8, spacing_m=dx, mobility=3e-3,
        kappa_J_m=5e-7, force=force, energy=energy)
    plus, plus_diag = adaptive_phase_proposal(eta_plus, **kwargs)
    minus, minus_diag = adaptive_phase_proposal(eta_minus, **kwargs)
    np.testing.assert_allclose(plus, minus[:, :, ::-1], rtol=0.0, atol=2e-12)
    assert plus_diag.accepted_substeps == minus_diag.accepted_substeps
    assert plus_diag.energy_change_J_m == minus_diag.energy_change_J_m


def test_pair_component_projection_removes_birth_but_allows_attached_advance():
    before = np.zeros((12, 12, 2), dtype=float)
    before[:, :, 0] = 1.0
    before[:4, :, 0] = 0.0
    before[:4, :, 1] = 1.0
    candidate = before.copy()
    candidate[4, :, :] = (0.4, 0.6)  # attached front advance
    candidate[8, 3, :] = (0.4, 0.6)  # disconnected label birth
    repaired, reverted = preserve_existing_pair_components(
        before, candidate, parent_label=0, child_label=1)
    assert reverted == 1
    np.testing.assert_array_equal(repaired[4, :, :], candidate[4, :, :])
    np.testing.assert_array_equal(repaired[8, 3, :], before[8, 3, :])


def test_pair_component_projection_blocks_bridge_that_splits_opposite_phase():
    before = np.zeros((12, 12, 2), dtype=float)
    before[:, :, 0] = 1.0
    before[:3, :, :] = (0.0, 1.0)
    candidate = before.copy()
    # Connected to the child through the periodic seam, but spanning the
    # parent would split it into two components.
    candidate[:, (0, 6), :] = (0.4, 0.6)
    repaired, reverted = preserve_existing_pair_components(
        before, candidate, parent_label=0, child_label=1)
    assert reverted >= 18
    np.testing.assert_array_equal(repaired[3:, 0, :], before[3:, 0, :])
    np.testing.assert_array_equal(repaired[3:, 6, :], before[3:, 6, :])
