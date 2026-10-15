from dataclasses import replace

import numpy as np

from full_model.analysis.run_v66_structural_bicrystal_front import _state
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters,
    project_lattice_line_to_interfaces,
)
from full_model.production.multigrain_common_state import InterfaceComponentState
from full_model.production.multigrain_production import MultiGrainInterface
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, nye_from_plastic_distortion,
)


def _fixture(n=16):
    length = 4.8e-6
    state, _, _ = _state(n, length, 1000.0, 2e7)
    shape = state.owners[0].mobile_plus_m2.shape
    component = InterfaceComponentState(
        MultiGrainInterface("edge-10-20", 10, 20).component_id, 10, 20,
        np.zeros((n, n)), np.zeros((n, n)),
        np.zeros((n, n, 3, shape[-1])),
        np.zeros((n, n, 3, shape[-1])),
        np.zeros_like(state.owners[0].junction_m2),
        boundary_burgers_m1=np.zeros((n, n, 3, 3)))
    state = replace(state, interfaces=(component,))
    return state, bcc_four_family_systems(), length/n


def test_one_flux_absorption_closes_line_beta_nye_and_interface_burgers():
    state, systems, spacing = _fixture()
    result = project_lattice_line_to_interfaces(
        state, systems=systems, spacing_m=spacing, fraction=.02,
        parameters=LatticeInterfaceAbsorptionParameters(
            maximum_fraction_per_interval=.05,
            capture_distance_m=spacing))
    assert result.removed_plus_line_m > 0.0
    np.testing.assert_allclose(
        result.removed_plus_line_m, result.interface_plus_line_m,
        rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(
        result.removed_minus_line_m, result.interface_minus_line_m,
        rtol=2e-14, atol=0.0)
    assert result.maximum_mura_relative_residual < 2e-11
    assert result.maximum_incident_capacity <= .02+1e-15
    assert np.max(result.capture_activity_m2) > 0.0
    assert np.max(result.capture_activity_m2) > 5*np.min(
        result.capture_activity_m2)
    for before, after in zip(state.owners, result.candidate_state.owners):
        initial_offset = (np.sum(before.family_nye_m1, axis=2)
                          -nye_from_plastic_distortion(before.beta_p, spacing))
        final_offset = (np.sum(after.family_nye_m1, axis=2)
                        -nye_from_plastic_distortion(after.beta_p, spacing))
        np.testing.assert_allclose(final_offset, initial_offset,
                                   rtol=0.0, atol=2e-7)


def test_zero_extent_is_exact_identity():
    state, systems, spacing = _fixture(8)
    result = project_lattice_line_to_interfaces(
        state, systems=systems, spacing_m=spacing, fraction=0.0,
        parameters=LatticeInterfaceAbsorptionParameters(
            capture_distance_m=spacing))
    for before, after in zip(state.owners, result.candidate_state.owners):
        for name in before.__dict__:
            np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
