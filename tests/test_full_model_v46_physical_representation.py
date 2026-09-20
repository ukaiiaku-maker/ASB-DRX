import numpy as np

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.lattice_line_geometry import (
    apply_physical_reconstruction,
    empty_lattice_geometry,
    propose_plaquette_sweep,
)
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, synchronize_common,
)


def test_fixed_length_reconstruction_is_positive_conservative_and_self_adjoint():
    rng = np.random.default_rng(46)
    field = np.zeros((32, 32, 2))
    field[3, 7, 0] = 2.0
    field[20, 17, 1] = 3.0
    mapped = apply_physical_reconstruction(field, 1e-7, 4e-7)
    assert np.min(mapped) >= -2e-15
    np.testing.assert_allclose(mapped.sum(axis=(0, 1)), field.sum(axis=(0, 1)),
                               rtol=2e-14, atol=2e-14)
    left = rng.normal(size=field.shape)
    right = rng.normal(size=field.shape)
    kl = apply_physical_reconstruction(left, 1e-7, 4e-7)
    kr = apply_physical_reconstruction(right, 1e-7, 4e-7)
    np.testing.assert_allclose(np.sum(kl*right), np.sum(left*kr),
                               rtol=3e-14, atol=3e-14)


def test_fixed_length_reconstruction_commutes_with_authoritative_curl():
    rng = np.random.default_rng(460)
    beta = rng.normal(size=(24, 24, 3, 3))
    spacing = 3.2e-6/24
    length = 4e-7
    mapped_beta = apply_physical_reconstruction(beta, spacing, length)
    mapped_nye = apply_physical_reconstruction(
        nye_from_plastic_distortion(beta, spacing), spacing, length)
    np.testing.assert_allclose(
        nye_from_plastic_distortion(mapped_beta, spacing), mapped_nye,
        rtol=3e-13, atol=3e-7)


def test_geometry_transaction_uses_fixed_thickness_and_one_map_for_all_fields():
    built = build_case(16, length_m=3.2e-6, periodic_nye_consistent=True)
    state, _, _, _, systems, topologies, _, _, _, spacing = built
    thickness = 3.2e-6
    geometry = empty_lattice_geometry(
        (16, 16), len(systems), spacing,
        section_thickness_m=thickness)
    state = V24MechanicalWallState(
        state.common, state.density, state.reservoir_alignment, geometry)
    candidate = propose_plaquette_sweep(
        geometry, state.density, state.reservoir_alignment, state.common,
        systems, state.common.orientation_rad, (7, 7), 0, 1, 1.0,
        continuum_representation_length_m=4e-7)
    new_geometry, density, alignment, common, ledger = candidate
    assert ledger["continuum_representation"] == (
        "fixed_physical_positive_periodic_kernel")
    assert ledger["represented_section_thickness_m"] == thickness
    assert np.min(density.wall_ordered_plus_m2) >= 0.0
    assert np.max(np.linalg.norm(alignment.wall_ordered_plus_m2, axis=-1)
                  -density.wall_ordered_plus_m2) <= 2e-12*max(
                      np.max(density.wall_ordered_plus_m2), 1.0)
    np.testing.assert_allclose(
        np.sum(ledger["plastic_distortion_increment"], axis=(0, 1)),
        np.sum(ledger["raw_plastic_distortion_increment"], axis=(0, 1)),
        rtol=2e-14, atol=2e-30)
    expected = nye_from_plastic_distortion(
        ledger["plastic_distortion_increment"], spacing)
    np.testing.assert_allclose(
        np.sum(ledger["family_nye_increment_m1"], axis=2), expected,
        rtol=2e-13, atol=2e-10)
    new_geometry.validate(len(systems))
    synchronize_common(
        V24MechanicalWallState(common, density, alignment, new_geometry),
        topologies).validate(systems, topologies)


def test_fractional_capture_support_has_fixed_width_and_mapped_capture_closes():
    for n in (64, 96):
        built = build_case(n, length_m=3.2e-6, periodic_nye_consistent=True)
        state, driving, _, support, systems, topologies, parameters, extensive, kinetics, dx = built
        np.testing.assert_allclose(np.sum(support[:, 0])*dx, 0.9e-6,
                                   rtol=2e-14, atol=2e-21)
        from full_model.production.v24_mechanical_wall import (
            accepted_v24_mechanical_step,
        )
        updated, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, parameters,
            extensive, kinetics, 1e-9, topology_route_enabled=False,
            mura_transport_operator="compatible_dealiased")
        capture = ledger["transport_capture"]
        assert capture["capture_deposition_length_m"] == 4e-7
        assert capture["capture_deposition_representation"] == (
            "fixed_physical_positive_periodic_kernel")
        for sign in ("plus", "minus"):
            row = capture["sign"][sign]
            assert 0.0 <= row["capture_capacity_scale"] <= 1.0
            assert abs(row["global_scalar_residual_line_per_thickness"]) < 1e-12
            assert np.max(np.abs(
                row["global_alignment_residual_line_per_thickness"])) < 1e-12
        updated.validate(systems, topologies)
