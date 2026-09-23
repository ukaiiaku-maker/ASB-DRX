import numpy as np

from tests.test_v50_production_subcell_geometry import physical_rectangle
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.nonlocal_elasticity import (
    elastic_energy_density, solve_periodic_eigenstrain,
    solve_periodic_eigenstrain_3d_z_invariant,
    z_invariant_equilibrium_residual,
)
from full_model.production.subcell_segment_geometry import (
    propose_subcell_x_face_moves,
)
from full_model.production.v24_mechanical_wall import (
    _elastic_energy_sum_J_m3_cells,
)


CONSTANTS = (228e9, 132e9, 116.5e9)


def patterned(n):
    x = np.arange(n)[:, None]/n
    y = np.arange(n)[None, :]/n
    value = np.zeros((n, n, 3, 3))
    value[..., 0, 1] = value[..., 1, 0] = (
        .003*np.sin(2*np.pi*x)*np.cos(4*np.pi*y))
    return value


def test_full_tensor_solver_recovers_retained_in_plane_limit():
    eigen3 = patterned(20)
    mean3 = np.zeros((3, 3)); mean3[0, 1] = mean3[1, 0] = .001
    stress3, strain3 = solve_periodic_eigenstrain_3d_z_invariant(
        eigen3, mean3, 1.6e-7, *CONSTANTS)
    stress2, strain2 = solve_periodic_eigenstrain(
        eigen3[..., :2, :2], mean3[:2, :2], 1.6e-7, *CONSTANTS,
        iterations=3)
    np.testing.assert_allclose(strain3[..., :2, :2], strain2,
                               rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(stress3[..., :2, :2], stress2,
                               rtol=2e-13, atol=2e-3)
    np.testing.assert_allclose(stress3[..., :2, 2], 0.0, atol=1e-6)


def test_antiplane_and_volumetric_eigenstrains_are_visible_and_equilibrated():
    n = 24
    x = np.arange(n)[:, None]/n
    y = np.arange(n)[None, :]/n
    eigen = np.zeros((n, n, 3, 3))
    eigen[..., 0, 2] = eigen[..., 2, 0] = (
        .002*np.sin(2*np.pi*x)*np.cos(2*np.pi*y))
    eigen[..., 2, 2] = .001*np.cos(2*np.pi*x)*np.sin(4*np.pi*y)
    stress, strain = solve_periodic_eigenstrain_3d_z_invariant(
        eigen, np.zeros((3, 3)), 2e-7, *CONSTANTS)
    energy = float(np.sum(elastic_energy_density(stress, strain, eigen)))
    assert energy > 0.0
    assert np.max(np.abs(stress[..., 0, 2])) > 0.0
    assert np.max(np.abs(stress[..., 2, 2])) > 0.0
    residual = z_invariant_equilibrium_residual(stress, 2e-7)
    scale = np.linalg.norm(stress)/2e-7
    assert np.linalg.norm(residual)/scale < 2e-13


def test_full_tensor_energy_has_the_declared_work_conjugate_derivative():
    n = 18
    eigen = patterned(n)
    x = np.arange(n)[:, None]/n
    perturbation = np.zeros_like(eigen)
    perturbation[..., 1, 2] = perturbation[..., 2, 1] = (
        .0017*np.cos(2*np.pi*x))

    def energy(scale):
        current = eigen+scale*perturbation
        stress, strain = solve_periodic_eigenstrain_3d_z_invariant(
            current, np.zeros((3, 3)), 2.5e-7, *CONSTANTS)
        return float(np.sum(elastic_energy_density(stress, strain, current)))

    h = 1e-5
    finite_difference = (energy(h)-energy(-h))/(2*h)
    stress, _ = solve_periodic_eigenstrain_3d_z_invariant(
        eigen, np.zeros((3, 3)), 2.5e-7, *CONSTANTS)
    virtual_work = -float(np.sum(stress*perturbation))
    np.testing.assert_allclose(finite_difference, virtual_work,
                               rtol=3e-7, atol=5e-2)


def test_production_rectangle_has_nonzero_conjugate_elastic_shape_force():
    state, data = physical_rectangle(32)
    mean3 = np.zeros((3, 3))
    mean3[:2, :2] = data[1].mean_strain
    mean3[0, 2] = mean3[2, 0] = 7.5e-4
    mean3[1, 2] = mean3[2, 1] = -4.0e-4
    driving = CommonWallDriving(
        mean_strain=data[1].mean_strain,
        fixed_eigenstrain=data[1].fixed_eigenstrain,
        full_tensor_z_invariant_enabled=True,
        mean_strain_3d=mean3)
    h = 2e-10
    proposals = {}
    for label, displacement in (("plus", h), ("minus", -h)):
        proposals[label] = propose_subcell_x_face_moves(
            state.subcell_geometry, state.density, state.reservoir_alignment,
            state.common, data[4], lower_displacement_m=0.0,
            upper_displacement_m=displacement)[3]
    energy = lambda common: _elastic_energy_sum_J_m3_cells(
        common, common.beta_p, driving, data[6])
    finite_difference = (energy(proposals["plus"])
                         -energy(proposals["minus"]))/(2*h)
    beta = state.common.beta_p
    eigen = .5*(beta+np.swapaxes(beta, -1, -2))
    stress, _ = solve_periodic_eigenstrain_3d_z_invariant(
        eigen, mean3, data[6].spacing_m, data[6].c11_Pa,
        data[6].c12_Pa, data[6].c44_Pa)
    dbeta = (proposals["plus"].beta_p
             -proposals["minus"].beta_p)/(2*h)
    deigen = .5*(dbeta+np.swapaxes(dbeta, -1, -2))
    virtual_work = -float(np.sum(stress*deigen))
    assert finite_difference != 0.0
    np.testing.assert_allclose(finite_difference, virtual_work,
                               rtol=2e-4, atol=2e7)


def test_joint_two_face_energy_retains_elastic_interaction():
    state, data = physical_rectangle(24)
    mean3 = np.zeros((3, 3)); mean3[:2, :2] = data[1].mean_strain
    mean3[0, 2] = mean3[2, 0] = 5e-4
    driving = CommonWallDriving(
        mean_strain=data[1].mean_strain,
        fixed_eigenstrain=data[1].fixed_eigenstrain,
        full_tensor_z_invariant_enabled=True, mean_strain_3d=mean3)
    energy = lambda common: _elastic_energy_sum_J_m3_cells(
        common, common.beta_p, driving, data[6])
    base = energy(state.common); h = 3e-10
    lower = propose_subcell_x_face_moves(
        state.subcell_geometry, state.density, state.reservoir_alignment,
        state.common, data[4], lower_displacement_m=h,
        upper_displacement_m=0.0)[3]
    upper = propose_subcell_x_face_moves(
        state.subcell_geometry, state.density, state.reservoir_alignment,
        state.common, data[4], lower_displacement_m=0.0,
        upper_displacement_m=-.7*h)[3]
    joint = propose_subcell_x_face_moves(
        state.subcell_geometry, state.density, state.reservoir_alignment,
        state.common, data[4], lower_displacement_m=h,
        upper_displacement_m=-.7*h)[3]
    interaction = (energy(joint)-base
                   -(energy(lower)-base)-(energy(upper)-base))
    assert np.isfinite(interaction)
    assert abs(interaction) > 1e-12*max(abs(energy(joint)-base), 1.0)
