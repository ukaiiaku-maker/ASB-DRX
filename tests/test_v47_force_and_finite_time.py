from dataclasses import replace

import numpy as np

from full_model.analysis.run_v47_ordering_finite_time import (
    exact_scalar_counterexample,
)
from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.extensive_wall import (
    extensive_wall_energy_components_J_m3,
    ordered_gradient_increment_J_m3_cells,
)
from full_model.production.v24_mechanical_wall import (
    V43GeometryKinetics,
    accepted_geometry_plaquette_transaction,
)
from tests.test_full_model_v43_lattice_geometry import (
    prepared_loop, verification_event,
)


def test_exact_tanh_counterexample_separates_accessibility_from_finite_time():
    row = exact_scalar_counterexample()
    assert row["accessibility_passed"]
    assert row["equilibrium_kkt_residual"] == 0.0
    np.testing.assert_allclose(row["exact_finite_time_q"],
                               0.3473761134414577, rtol=0.0, atol=2e-16)
    assert row["equilibrium_absolute_error"] > 0.15


def test_direct_ordered_gradient_increment_matches_endpoint_difference():
    state, args, _ = prepared_loop()
    before = state.density
    perturbation = np.zeros_like(before.wall_ordered_plus_m2)
    perturbation[2, 3, 0] = 1.25e7
    perturbation[7, 8, 0] = 2.5e6
    after = replace(before,
        wall_ordered_plus_m2=before.wall_ordered_plus_m2+perturbation)
    zero = np.zeros(state.common.orientation_rad.shape+(3, 3))
    before_energy = extensive_wall_energy_components_J_m3(
        before, args[3], args[4], state.common.orientation_rad, zero, args[6])
    after_energy = extensive_wall_energy_components_J_m3(
        after, args[3], args[4], state.common.orientation_rad, zero, args[6])
    endpoint = np.sum(after_energy["ordered_gradient"]
                      -before_energy["ordered_gradient"], dtype=np.longdouble)
    direct = ordered_gradient_increment_J_m3_cells(before, after, args[6])
    # Endpoint subtraction loses digits against the prepared background; the
    # cancellation-aware identity remains consistent at the observed scale.
    np.testing.assert_allclose(direct, endpoint, rtol=5e-4, atol=1e-18)
    assert direct != float(endpoint)


def test_physical_chemical_work_is_tied_to_signed_exchange_not_extent():
    state, args, _ = prepared_loop()
    physical = V43GeometryKinetics(
        ActivatedProcess("v47-physical-reservoir", 1e12),
        enthalpy_J=0.0, critical_stress_Pa=1e9,
        material_exchange_model="equilibrated_point_defect_reservoir",
        chemical_species="vacancy",
        chemical_potential_J_per_defect=1.0e-19,
        atomic_volume_m3_per_atom=1.8e-29,
        exchange_stoichiometry_defects_per_atom=1.0)
    _, ledger = accepted_geometry_plaquette_transaction(
        state, verification_event(extent=.25), args[3], args[4], args[1],
        args[5], args[6], physical, 1e-9)
    expected_count = (ledger["signed_material_exchange_volume_m3"]
                      /physical.atomic_volume_m3_per_atom)
    np.testing.assert_allclose(ledger["signed_material_exchange_count"],
                               expected_count, rtol=2e-15, atol=1e-20)
    np.testing.assert_allclose(
        ledger["chemical_reservoir_work_J"],
        physical.chemical_potential_J_per_defect*expected_count,
        rtol=2e-15, atol=1e-30)
    assert ledger["chemical_work_mode"] == "physical_species_exchange"
    # The credit is reconstructed from the signed physical exchange, not from
    # grid-cell count or the absolute proposal extent.
    assert ledger["signed_material_exchange_volume_m3"] != 0.0
    assert np.sign(ledger["chemical_reservoir_work_J"]) == np.sign(
        ledger["signed_material_exchange_volume_m3"])


def test_legacy_and_physical_chemical_credits_cannot_be_combined():
    with np.testing.assert_raises(ValueError):
        V43GeometryKinetics(
            ActivatedProcess("invalid-double-credit", 1.0),
            enthalpy_J=0.0, critical_stress_Pa=1.0,
            chemical_work_J_m3_cells_per_extent=1.0,
            chemical_species="vacancy", chemical_potential_J_per_defect=1.0,
            atomic_volume_m3_per_atom=1.0,
            exchange_stoichiometry_defects_per_atom=1.0)
