from dataclasses import replace

import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
    accepted_euler_step, accepted_step_jacobian_vector_product,
    balance_ledger, fourier_symbol, jacobian_vector_product,
    resolved_driving_components, taylor_resistance_Pa,
    wall_polarization_invariants,
    wall_free_energy_density_J_m3,
    wall_free_energy_derivatives, wall_residual,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, junction_closure_metrics, make_junction_topology,
)


def fixture(n=12):
    systems = bcc_four_family_systems()
    topology = make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0)
    shape = (n, n, 4)
    mp = np.full(shape, 8e13); mm = np.full(shape, 7e13)
    fp = np.full(shape, 5e13); fm = np.full(shape, 4e13)
    wp = np.full(shape, 2e13); wm = np.full(shape, 1.5e13)
    junction = np.full((n, n, 1), 1e12)
    q = np.full((n, n), .2)
    slip = np.zeros(shape); beta = np.zeros((n, n, 3, 3))
    alignment = np.zeros((n, n, 4, 3))
    nye = np.zeros((n, n, 4, 3, 3)); orientation = np.zeros((n, n))
    temperature = np.full((n, n), 1100.0)
    state = CommonWallState(
        mp, mm, fp, fm, wp, wm, junction, q, np.zeros((n, n)), slip, beta, alignment, nye,
        orientation, temperature)
    driving = CommonWallDriving(
        np.full(shape, 2e-4), np.full(shape, 5e8))
    return systems, (topology,), CommonWallParameters(spacing_m=2e-7), state, driving


def burgers_inventory(state, systems, topologies):
    signed = (state.mobile_plus_m2-state.mobile_minus_m2
              +state.forest_plus_m2-state.forest_minus_m2
              +state.wall_plus_m2-state.wall_minus_m2)
    result = np.einsum(
        "...a,ai->...i", signed,
        np.stack([system.burgers_vector_m for system in systems]))
    for index, topology in enumerate(topologies):
        result += state.junction_m2[..., index, None]*topology.product_burgers_m
    return result


def test_common_residual_reaction_burgers_and_line_node_closure():
    systems, topologies, p, state, driving = fixture()
    residual = wall_residual(state, driving, systems, topologies, p)
    rate = residual.state_rate
    # Homogeneous transport has zero divergence; reactions conserve vector
    # Burgers inventory when explicit product reservoirs are included.
    residual_burgers = burgers_inventory(rate, systems, topologies)
    scale = max(float(np.max(np.abs(
        rate.forest_plus_m2[..., 0, None]*systems[0].burgers_vector_m))), 1.0)
    assert float(np.max(np.abs(residual_burgers)))/scale < 2e-15
    closure = junction_closure_metrics(topologies[0], systems)
    assert closure["frank_rule_residual_m"] == 0.0
    assert closure["line_node_residual"] < 2e-16
    assert np.all(residual.channel_rates_m2_s["junction_line_sink"] >= 0.0)
    ledger = balance_ledger(state, residual, systems, topologies)
    assert ledger["relative_burgers_rate_residual"] < 2e-15
    line_scale = max(abs(ledger["line_rate_m2_s"]), 1.0)
    assert abs(ledger["line_balance_residual_m2_s"])/line_scale < 2e-14
    assert ledger["relative_energy_balance_residual"] < 2e-15


def test_wall_order_residual_is_derivative_of_same_free_energy():
    systems, topologies, p, state, driving = fixture()
    derivative = wall_free_energy_derivatives(state, p)[
        "wall_order_derivative_J_m3"]
    h = 1e-6
    plus = CommonWallState(**{
        **state.__dict__, "wall_order": state.wall_order+h})
    minus = CommonWallState(**{
        **state.__dict__, "wall_order": state.wall_order-h})
    numerical = (wall_free_energy_density_J_m3(plus, p)
                 -wall_free_energy_density_J_m3(minus, p))/(2*h)
    np.testing.assert_allclose(derivative, numerical, rtol=3e-9, atol=.2)
    residual = wall_residual(state, driving, systems, topologies, p)
    assert np.all(residual.state_rate.wall_order*derivative <= 1e-8)


def test_reported_defect_energy_rate_is_directional_derivative():
    systems, topologies, p, state, driving = fixture()
    residual = wall_residual(state, driving, systems, topologies, p)
    rate = residual.state_rate
    # Keep the centered perturbation small relative to every positive reservoir
    # and to the bounded order coordinate.
    h = 3e-9
    plus = CommonWallState(**{
        name: value+h*getattr(rate, name)
        for name, value in state.__dict__.items()})
    minus = CommonWallState(**{
        name: value-h*getattr(rate, name)
        for name, value in state.__dict__.items()})
    numerical = np.mean(
        wall_free_energy_density_J_m3(plus, p, topologies)
        -wall_free_energy_density_J_m3(minus, p, topologies))/(2*h)
    np.testing.assert_allclose(
        np.mean(residual.free_energy_rate_W_m3), numerical,
        rtol=1e-7, atol=2e-2)


def test_accepted_step_is_exactly_scaled_common_residual_and_nonnegative():
    systems, topologies, p, state, driving = fixture()
    dt = 1e-8
    updated, residual, scale = accepted_euler_step(
        state, driving, systems, topologies, p, dt)
    assert 0.0 < scale <= 1.0
    for name in state.__dict__:
        expected = (getattr(state, name)
                    +dt*scale*getattr(residual.state_rate, name))
        np.testing.assert_allclose(getattr(updated, name), expected)
    updated.validate(systems, topologies)


def test_jvp_uses_authoritative_residual():
    systems, topologies, p, state, driving = fixture()
    zero = {name: np.zeros_like(value) for name, value in state.__dict__.items()}
    zero["wall_order"][:] = 1.0
    direction = CommonWallState(**zero)
    jvp = jacobian_vector_product(
        state, direction, driving, systems, topologies, p, relative_step=1e-7)
    h = 1e-7
    plus = CommonWallState(**{
        name: value+h*getattr(direction, name)
        for name, value in state.__dict__.items()})
    minus = CommonWallState(**{
        name: value-h*getattr(direction, name)
        for name, value in state.__dict__.items()})
    rp = wall_residual(plus, driving, systems, topologies, p).state_rate
    rm = wall_residual(minus, driving, systems, topologies, p).state_rate
    for name in state.__dict__:
        reference = (getattr(rp, name)-getattr(rm, name))/(2*h)
        np.testing.assert_allclose(getattr(jvp, name), reference, rtol=2e-7, atol=2e-5)


def test_accepted_step_jvp_reduces_to_identity_plus_dt_residual_jvp():
    systems, topologies, p, state, driving = fixture(n=4)
    zero = {name: np.zeros_like(value) for name, value in state.__dict__.items()}
    zero["wall_order"][:] = 1.0
    direction = CommonWallState(**zero)
    dt = 1e-10
    assert accepted_euler_step(state, driving, systems, topologies, p, dt)[2] == 1.0
    map_jvp = accepted_step_jacobian_vector_product(
        state, direction, driving, systems, topologies, p, dt, 1e-7)
    residual_jvp = jacobian_vector_product(
        state, direction, driving, systems, topologies, p, 1e-7)
    for name in state.__dict__:
        expected = getattr(direction, name)+dt*getattr(residual_jvp, name)
        np.testing.assert_allclose(
            getattr(map_jvp, name), expected, rtol=2e-4, atol=2e-6)


def test_nonlocal_mechanical_and_thermal_tangents_enter_same_symbol():
    systems, topologies, p, state, _ = fixture(n=8)
    driving = CommonWallDriving(mean_strain=np.array([[.004, 0.0], [0.0, -.001]]))
    symbol = fourier_symbol(
        state, driving, systems, topologies, p, (1, 0), relative_step=2e-6)
    matrix = symbol["matrix_s_inv"]
    assert matrix.shape[0] == matrix.shape[1] == 38
    assert np.all(np.isfinite(matrix))
    layout = symbol["layout"]
    beta_columns = [i for i, item in enumerate(layout) if item[0] == "beta_p"]
    temperature_column = next(i for i, item in enumerate(layout)
                              if item[0] == "temperature_K")
    density_rows = [i for i, item in enumerate(layout)
                    if item[0].startswith("mobile_")]
    # Beta perturbations alter nonlocal stress/glide, and temperature alters
    # the exact Arrhenius residual. Neither tangent is appended afterwards.
    assert np.max(np.abs(matrix[np.ix_(density_rows, beta_columns)])) > 0.0
    assert np.max(np.abs(matrix[density_rows, temperature_column])) > 0.0


def test_polarization_gate_stabilizes_q_without_physical_signed_wall():
    systems, topologies, p, state, _ = fixture(n=4)
    for gate_form in ("joint_rational", "product_rational"):
        parameters = replace(p, wall_gate_form=gate_form)
        for wall_plus, wall_minus in ((0.0, 0.0), (5e13, 5e13)):
            candidate = replace(
                state, wall_plus_m2=np.full_like(state.wall_plus_m2, wall_plus),
                wall_minus_m2=np.full_like(state.wall_minus_m2, wall_minus),
                wall_order=np.zeros_like(state.wall_order))
            invariants = wall_polarization_invariants(candidate, systems, parameters)
            assert np.max(invariants["wall_gate"]) < 1e-24
            h = 1e-4
            energy = lambda q: np.mean(wall_free_energy_density_J_m3(
                replace(candidate, wall_order=np.full_like(candidate.wall_order, q)),
                parameters, topologies, systems))
            curvature = (energy(h)-2*energy(0.0)+energy(-h))/h**2
            assert curvature > 0.0


def test_only_polarized_wall_can_favor_order_and_gate_is_objective():
    systems, topologies, p, state, driving = fixture(n=4)
    polarized = replace(
        state, wall_plus_m2=np.full_like(state.wall_plus_m2, 1e14),
        wall_minus_m2=np.zeros_like(state.wall_minus_m2),
        wall_order=np.zeros_like(state.wall_order))
    for gate_form in ("joint_rational", "product_rational"):
        parameters = replace(p, wall_gate_form=gate_form)
        e0 = wall_free_energy_density_J_m3(
            polarized, parameters, topologies, systems)
        e1 = wall_free_energy_density_J_m3(
            replace(polarized, wall_order=np.ones_like(polarized.wall_order)),
            parameters, topologies, systems)
        assert np.mean(e1-e0) < 0.0
        assert np.min(wall_residual(
            polarized, driving, systems, topologies,
            parameters).state_rate.wall_order) > 0.0
        forward = wall_polarization_invariants(polarized, systems, parameters)
        reversed_state = replace(
            polarized, wall_plus_m2=polarized.wall_minus_m2,
            wall_minus_m2=polarized.wall_plus_m2)
        reversed_gate = wall_polarization_invariants(
            reversed_state, systems, parameters)["wall_gate"]
        rotated_gate = wall_polarization_invariants(
            replace(polarized, orientation_rad=np.full_like(
                polarized.orientation_rad, .731)), systems, parameters)["wall_gate"]
        np.testing.assert_allclose(reversed_gate, forward["wall_gate"], rtol=2e-15)
        np.testing.assert_allclose(rotated_gate, forward["wall_gate"], rtol=2e-15)


def test_signed_wall_chemical_potentials_are_energy_derivatives():
    systems, topologies, p, state, _ = fixture(n=4)
    chemical = wall_free_energy_derivatives(state, p, topologies, systems)
    perturbation = 1e7
    for name, key in (("wall_plus_m2", "wall_plus_mu_J_m"),
                      ("wall_minus_m2", "wall_minus_mu_J_m")):
        delta = np.zeros_like(getattr(state, name)); delta[..., 0] = perturbation
        plus = replace(state, **{name: getattr(state, name)+delta})
        minus = replace(state, **{name: getattr(state, name)-delta})
        numerical = (wall_free_energy_density_J_m3(plus, p, topologies, systems)
                     -wall_free_energy_density_J_m3(minus, p, topologies, systems)
                     )/(2*perturbation)
        np.testing.assert_allclose(
            numerical, chemical[key][..., 0], rtol=2e-7, atol=2e-15)


def test_taylor_resistance_and_glide_have_required_limits_and_dissipation():
    systems, topologies, p, state, _ = fixture(n=4)
    stresses = np.broadcast_to(
        np.array([-2e9, -1e6, 0.0, 2e9]), state.mobile_plus_m2.shape).copy()
    driving = CommonWallDriving(resolved_stress_Pa=stresses)
    fields = resolved_driving_components(state, driving, systems, topologies, p)
    assert np.all(fields["taylor_resistance_Pa"] >= 0.0)
    assert np.all(stresses*fields["speed_m_s"] >= 0.0)
    assert np.all(fields["effective_stress_Pa"]*fields["speed_m_s"] >= 0.0)
    denser = replace(state, forest_plus_m2=4*state.forest_plus_m2)
    assert np.all(taylor_resistance_Pa(denser, systems, topologies, p)
                  >=taylor_resistance_Pa(state, systems, topologies, p))
    zero_taylor = replace(p, taylor_alpha=0.0)
    baseline = resolved_driving_components(
        state, driving, systems, topologies, zero_taylor)
    np.testing.assert_array_equal(baseline["effective_stress_Pa"], stresses)
    assert np.all(baseline["speed_m_s"][stresses == 0.0] == 0.0)


def test_multi_hit_off_is_exact_and_on_state_remains_bounded():
    systems, topologies, p, state, driving = fixture(n=4)
    altered = replace(state, multi_hit_coordination=np.full_like(
        state.multi_hit_coordination, .73))
    off_zero = wall_residual(state, driving, systems, topologies, p)
    off_altered = wall_residual(altered, driving, systems, topologies, p)
    for name in state.__dict__:
        np.testing.assert_array_equal(
            getattr(off_zero.state_rate, name), getattr(off_altered.state_rate, name))
    enabled = replace(
        p, multi_hit_enabled=True, multi_hit_relaxation_s=1e-4,
        multi_hit_collision_scale=1e3)
    updated, residual, scale = accepted_euler_step(
        altered, driving, systems, topologies, enabled, 1e-5)
    assert 0.0 < scale <= 1.0
    assert np.min(updated.multi_hit_coordination) >= 0.0
    assert np.max(updated.multi_hit_coordination) <= 1.0
    assert np.all(residual.channel_rates_m2_s["collision_frequency_s"] >= 0.0)


def test_unloaded_hold_decreases_declared_defect_free_energy():
    systems, topologies, p, state, _ = fixture(n=4)
    zero = np.zeros_like(state.mobile_plus_m2)
    residual = wall_residual(
        state, CommonWallDriving(glide_speed_m_s=zero,
                                 resolved_stress_Pa=zero),
        systems, topologies, p)
    assert np.max(residual.plastic_power_W_m3) == 0.0
    assert np.max(residual.free_energy_rate_W_m3) <= 0.0
