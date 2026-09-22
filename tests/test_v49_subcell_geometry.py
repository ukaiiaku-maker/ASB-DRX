import numpy as np

from full_model.production.lattice_line_geometry import (
    empty_lattice_geometry, propose_plaquette_sweep,
    subcell_geometry_to_continuum,
)
from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.analysis.run_v46_geometry_representation import (
    LENGTH_M, REPRESENTATION_LENGTH_M, THICKNESS_M,
    prepare_represented_block,
)
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, mechanical_checkpoint_arrays, synchronize_common,
)


def _field(n=33):
    value = np.zeros((n, n, 2))
    value[8:15, 10:19, 0] = 1.0
    value[19:23, 4:11, 1] = -0.4
    return value


def test_subcell_continuum_map_has_rigid_integer_translation_and_invariants():
    dx = 2e-8
    value = _field()
    base = subcell_geometry_to_continuum(value, dx, 8e-8)
    shifted = subcell_geometry_to_continuum(
        value, dx, 8e-8, (3*dx, -2*dx))
    np.testing.assert_allclose(
        shifted, np.roll(base, (3, -2), axis=(0, 1)),
        rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(np.sum(shifted, axis=(0, 1)),
                               np.sum(base, axis=(0, 1)), rtol=2e-14)
    np.testing.assert_allclose(np.sum(shifted**2), np.sum(base**2),
                               rtol=2e-14)


def test_subcell_continuum_displacement_derivative_matches_same_path_fd():
    dx = 2e-8
    displacement = np.asarray((0.37*dx, -0.21*dx))
    mapped, derivative = subcell_geometry_to_continuum(
        _field(), dx, 8e-8, displacement,
        return_displacement_derivative=True)
    assert mapped.shape == _field().shape
    h = 1e-5*dx
    for axis in range(2):
        offset = np.zeros(2); offset[axis] = h
        plus = subcell_geometry_to_continuum(
            _field(), dx, 8e-8, displacement+offset)
        minus = subcell_geometry_to_continuum(
            _field(), dx, 8e-8, displacement-offset)
        numerical = (plus-minus)/(2*h)
        np.testing.assert_allclose(
            derivative[axis], numerical, rtol=3e-8,
            atol=2e-6*np.max(np.abs(derivative[axis])))


def test_fractional_occupancy_mixture_is_not_declared_rigid_translation():
    dx = 2e-8
    value = _field()
    theta = 0.4
    mixture = (1.0-theta)*value+theta*np.roll(value, 1, axis=0)
    rigid = subcell_geometry_to_continuum(
        value, dx, 0.0, (theta*dx, 0.0))
    # The former V48 path is a convex ensemble of integer states.  The new
    # coordinate map is a translated continuum field; they are not aliases.
    assert np.linalg.norm(mixture-rigid) > 1e-3


def test_batch_closed_surface_initializer_matches_sequential_fixture():
    n = 8
    batch, _, start, width = prepare_represented_block(n)
    data = build_case(n, length_m=LENGTH_M, periodic_nye_consistent=True)
    base, _, _, _, systems, topologies, _, _, _, dx = data
    geometry = empty_lattice_geometry(
        (n, n), len(systems), dx, section_thickness_m=THICKNESS_M)
    sequential = V24MechanicalWallState(
        base.common, base.density, base.reservoir_alignment, geometry)
    for i in range(start, start+width):
        for j in range(start, start+width):
            candidate = propose_plaquette_sweep(
                sequential.geometry, sequential.density,
                sequential.reservoir_alignment, sequential.common, systems,
                sequential.common.orientation_rad, (i, j), 0, 1, 1.0,
                continuum_representation_length_m=REPRESENTATION_LENGTH_M)
            sequential = synchronize_common(V24MechanicalWallState(
                candidate[3], candidate[1], candidate[2], candidate[0]),
                topologies)
    for name, value in mechanical_checkpoint_arrays(batch).items():
        if name.endswith("accepted_event_count"):
            assert int(value) == 0
            assert int(mechanical_checkpoint_arrays(sequential)[name]) == width**2
        else:
            np.testing.assert_allclose(
                value, mechanical_checkpoint_arrays(sequential)[name],
                rtol=3e-13, atol=5e-6)
