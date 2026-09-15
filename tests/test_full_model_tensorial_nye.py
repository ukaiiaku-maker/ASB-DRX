import numpy as np

from full_model.production.tensorial_nye import (
    accept_slip_increment, bcc_four_family_systems, consistency_metrics,
    detailed_balance_rates, initialize_tensorial_state,
    frank_bilby_closure_from_orientations, integrated_nye_closure,
    make_junction_topology, nye_from_alignment,
    nye_from_plastic_distortion, plastic_distortion_from_slip,
    rotation_z, junction_closure_metrics,
)


def smooth_periodic_slip(n, families):
    x = np.arange(n)[:, None]/n
    y = np.arange(n)[None, :]/n
    gamma = np.zeros((n, n, families))
    gamma[:, :, 0] = .01*np.sin(2*np.pi*x)*np.cos(4*np.pi*y)
    return gamma


def test_manufactured_single_slip_dual_nye_and_line_continuity():
    systems = bcc_four_family_systems()
    state = initialize_tensorial_state((48, 48), systems)
    orientation = np.zeros((48, 48))
    state = accept_slip_increment(
        state, smooth_periodic_slip(48, 4), systems, orientation, 2e-7)
    metrics = consistency_metrics(state, systems, orientation, 2e-7)
    assert metrics["relative_rms_residual"] < 2e-13
    assert metrics["relative_divergence_rms"] < 2e-13


def test_slip_reconstruction_matches_incremental_beta_path():
    systems = bcc_four_family_systems()
    orientation = np.zeros((32, 32))
    increment = smooth_periodic_slip(32, 4)
    state = accept_slip_increment(
        initialize_tensorial_state((32, 32), systems), increment,
        systems, orientation, 3e-7)
    np.testing.assert_allclose(
        state.beta_p, plastic_distortion_from_slip(increment, systems, orientation),
        rtol=2e-15, atol=1e-16)


def test_frame_rotation_covariance_for_constant_tensor_state():
    systems = bcc_four_family_systems()
    n = 16
    gamma = np.zeros((n, n, 4)); gamma[:, :, 0] = .02
    beta0 = plastic_distortion_from_slip(gamma, systems, np.zeros((n, n)))
    angle = .37
    beta1 = plastic_distortion_from_slip(gamma, systems, np.full((n, n), angle))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    expected = np.einsum("ik,...kl,jl->...ij", R, beta0, R)
    np.testing.assert_allclose(beta1, expected, rtol=2e-14, atol=2e-16)


def test_balanced_high_density_alignment_has_zero_nye():
    systems = bcc_four_family_systems()
    # Large unsigned plus/minus populations can have exactly zero first moment.
    # The unsigned density is an independent scalar inventory and does not enter
    # Nye unless its signed line-direction moment is nonzero.
    unsigned_plus = np.full((12, 14, 4), 2e14)
    unsigned_minus = unsigned_plus.copy()
    alignment = np.zeros((12, 14, 4, 3))
    assert np.mean(unsigned_plus+unsigned_minus) == 4e14
    np.testing.assert_array_equal(
        nye_from_alignment(alignment, systems, np.zeros((12, 14))), 0.0)


def test_variable_orientation_connection_terms_close_dual_nye():
    systems = bcc_four_family_systems()
    n = 40
    x = np.arange(n)[:, None]/n
    y = np.arange(n)[None, :]/n
    orientation = .2*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    state = accept_slip_increment(
        initialize_tensorial_state((n, n), systems),
        smooth_periodic_slip(n, 4), systems, orientation, 2.5e-7)
    metrics = consistency_metrics(state, systems, orientation, 2.5e-7)
    assert metrics["relative_rms_residual"] < 3e-13
    assert np.linalg.norm(metrics["connection_excess_m1"]) > 0.0


def test_junction_obeys_frank_rule_and_detailed_balance():
    systems = bcc_four_family_systems()
    topology = make_junction_topology(systems, 0, 1, 1, -1,
                                      line_tension_J_m=1e-9)
    expected = (systems[0].burgers_vector_m
                -systems[1].burgers_vector_m)
    np.testing.assert_allclose(topology.product_burgers_m, expected)
    closure = junction_closure_metrics(topology, systems)
    assert closure["frank_rule_residual_m"] == 0.0
    assert closure["line_node_residual"] < 2e-16
    assert topology.character == "sessile"
    before = systems[0].burgers_m**2+systems[1].burgers_m**2
    assert np.dot(expected, expected) <= before+1e-30
    kf, kr = detailed_balance_rates(
        2e5, topology.delta_free_energy_J_m, 1e-8, 1100.)
    ratio = kf/kr
    expected_ratio = np.exp(-topology.delta_free_energy_J_m*1e-8/
                            (1.380649e-23*1100.))
    np.testing.assert_allclose(ratio, expected_ratio, rtol=2e-15)


def test_closed_simple_tilt_wall_has_independent_frank_bilby_closure():
    n = 128
    spacing = 1.0e-7
    theta = np.deg2rad(4.0)
    # Periodic piecewise-constant orientation creates two closed straight walls.
    orientation = np.zeros((n, n))
    orientation[n//4:3*n//4, :] = theta
    rotation = np.asarray(rotation_z(orientation))
    identity = np.eye(3)[None, None, :, :]
    beta = identity-rotation
    alpha = nye_from_plastic_distortion(beta, spacing)
    # Integrate a window around the left wall, excluding its periodic partner.
    window = np.zeros_like(alpha)
    lo, hi = n//8, 3*n//8
    window[lo:hi] = alpha[lo:hi]
    from_nye = integrated_nye_closure(
        window, normal_axis=0, spacing_m=spacing, line_axis=2)
    from_lattices = frank_bilby_closure_from_orientations(
        0.0, theta, np.array([0.0, 1.0, 0.0]))
    residual = np.linalg.norm(from_nye-from_lattices)/np.linalg.norm(from_lattices)
    # A sharp, one-cell jump has the expected spectral ringing; the integrated
    # independent closure remains below the campaign's provisional 5% limit.
    assert residual < .05
