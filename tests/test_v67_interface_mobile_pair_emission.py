from dataclasses import replace

import numpy as np

from full_model.analysis.run_v66_structural_bicrystal_front import _state
from full_model.production.interface_mobile_pair_emission import (
    evaluate_interface_mobile_pair_emission,
    project_interface_mobile_pair_emission,
)
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.multigrain_common_state import InterfaceComponentState
from full_model.production.multigrain_production import MultiGrainInterface
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, nye_from_plastic_distortion,
)


def _fixture(n=8):
    length = 4.8e-6
    state, _, _ = _state(n, length, 1000.0, 2e7)
    families = state.owners[0].mobile_plus_m2.shape[-1]
    plus = np.zeros((n, n, 3, families))
    minus = np.zeros_like(plus)
    plus[..., 0, :] = 2.0e14
    minus[..., 0, :] = 1.5e14
    component = InterfaceComponentState(
        MultiGrainInterface("edge-10-20", 10, 20).component_id, 10, 20,
        np.zeros((n, n)), np.zeros((n, n)), plus, minus,
        np.zeros_like(state.owners[0].junction_m2),
        boundary_burgers_m1=np.full((n, n, 3, 3), 7.0))
    return replace(state, interfaces=(component,)), length/n


def test_neutral_interface_emission_closes_line_burgers_and_mura():
    state, spacing = _fixture()
    systems = bcc_four_family_systems()
    stresses = tuple(np.full_like(owner.mobile_plus_m2, 2.0e9)
                     for owner in state.owners)
    result = project_interface_mobile_pair_emission(
        state, fraction=.1, resolved_stress_by_owner=stresses,
        systems=systems, spacing_m=spacing, burgers_m=2.48e-10,
        source_grain_size_m=2.4e-6)
    assert result.removed_interface_pair_cell_sum_m2 > 0.0
    np.testing.assert_allclose(
        result.created_lattice_pair_cell_sum_m2,
        result.removed_interface_pair_cell_sum_m2, rtol=2e-15)
    assert abs(result.line_balance_residual_cell_sum_m2) < 1e-12*result.removed_interface_pair_cell_sum_m2
    assert abs(result.signed_content_change_cell_sum_m2) < 1e-12
    assert result.maximum_mura_relative_residual < 2e-11
    before_burgers = state.interfaces[0].boundary_burgers_m1
    after_burgers = result.candidate_state.interfaces[0].boundary_burgers_m1
    np.testing.assert_array_equal(after_burgers, before_burgers)
    for before, after in zip(state.owners, result.candidate_state.owners):
        initial_offset = (np.sum(before.family_nye_m1, axis=2)
                          -nye_from_plastic_distortion(before.beta_p, spacing))
        final_offset = (np.sum(after.family_nye_m1, axis=2)
                        -nye_from_plastic_distortion(after.beta_p, spacing))
        np.testing.assert_allclose(final_offset, initial_offset, atol=2e-7)


def test_zero_emission_is_bitwise_identity():
    state, spacing = _fixture()
    result = project_interface_mobile_pair_emission(
        state, fraction=0.0,
        resolved_stress_by_owner=tuple(
            np.zeros_like(owner.mobile_plus_m2) for owner in state.owners),
        systems=bcc_four_family_systems(), spacing_m=spacing,
        burgers_m=2.48e-10, source_grain_size_m=2.4e-6)
    assert result.candidate_state is state


def test_emission_is_sign_exchange_symmetric():
    state, spacing = _fixture()
    systems = bcc_four_family_systems()
    stress = tuple(np.full_like(owner.mobile_plus_m2, 2.0e9)
                   for owner in state.owners)
    positive = project_interface_mobile_pair_emission(
        state, fraction=.05, resolved_stress_by_owner=stress,
        systems=systems, spacing_m=spacing, burgers_m=2.48e-10,
        source_grain_size_m=2.4e-6).candidate_state
    negative = project_interface_mobile_pair_emission(
        state, fraction=.05,
        resolved_stress_by_owner=tuple(-value for value in stress),
        systems=systems, spacing_m=spacing, burgers_m=2.48e-10,
        source_grain_size_m=2.4e-6).candidate_state
    for initial, plus, minus in zip(
            state.owners, positive.owners, negative.owners):
        np.testing.assert_allclose(plus.mobile_plus_m2, minus.mobile_plus_m2)
        np.testing.assert_allclose(plus.slip-initial.slip,
                                   -(minus.slip-initial.slip))


def test_arrhenius_emission_is_complete_energy_accepted_at_finite_load():
    state, spacing = _fixture()
    systems = bcc_four_family_systems()
    result = evaluate_interface_mobile_pair_emission(
        state, mean_strain=np.array([[0.0, .025], [.025, 0.0]]),
        interval_s=2e-9, systems=systems,
        wall_parameters=CommonWallParameters(spacing_m=spacing),
        represented_thickness_m=2*2.48e-10,
        source_parameters=ArrheniusSourceParameters(grain_size_m=2.4e-6),
        energy_kwargs={"topologies": (), "systems": systems})
    assert result.accepted
    assert result.classification == "ACCEPTED_INTERFACE_MOBILE_PAIR_EMISSION"
    assert result.delta_helmholtz_before_heat_J < 0.0
    assert result.generated_heat_J > 0.0
    assert result.relative_first_law_residual < 5e-5
    assert result.committed_state.grain_ids == state.grain_ids
    np.testing.assert_array_equal(result.committed_state.supports, state.supports)


def test_arrhenius_emission_complete_clock_is_timestep_stable():
    state, spacing = _fixture()
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(spacing_m=spacing)
    kwargs = dict(
        mean_strain=np.array([[0.0, .025], [.025, 0.0]]), systems=systems,
        wall_parameters=wall, represented_thickness_m=2*2.48e-10,
        source_parameters=ArrheniusSourceParameters(grain_size_m=2.4e-6),
        energy_kwargs={"topologies": (), "systems": systems})
    whole = evaluate_interface_mobile_pair_emission(
        state, interval_s=2e-9, **kwargs)
    half1 = evaluate_interface_mobile_pair_emission(
        state, interval_s=1e-9, **kwargs)
    half2 = evaluate_interface_mobile_pair_emission(
        half1.committed_state, interval_s=1e-9, **kwargs)
    assert whole.accepted and half1.accepted and half2.accepted
    split_pairs = (
        half1.projection.created_lattice_pair_cell_sum_m2
        +half2.projection.created_lattice_pair_cell_sum_m2)
    np.testing.assert_allclose(
        split_pairs, whole.projection.created_lattice_pair_cell_sum_m2,
        rtol=2e-5)
    for combined, direct in zip(
            half2.committed_state.interfaces, whole.committed_state.interfaces):
        np.testing.assert_allclose(
            combined.boundary_plus_m2, direct.boundary_plus_m2, rtol=2e-5)
        np.testing.assert_allclose(
            combined.boundary_minus_m2, direct.boundary_minus_m2, rtol=2e-5)
