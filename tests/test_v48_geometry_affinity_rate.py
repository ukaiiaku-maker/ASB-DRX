from dataclasses import replace

import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.v24_mechanical_wall import (
    V43GeometryKinetics, accepted_geometry_plaquette_transaction,
)
from tests.test_full_model_v43_lattice_geometry import prepared_loop


def _kinetics(mu):
    return V43GeometryKinetics(
        ActivatedProcess("v48-affinity-geometry", 1e9),
        enthalpy_J=.2*EV_J, critical_stress_Pa=1e9,
        chemical_species="vacancy", chemical_potential_J_per_defect=mu,
        atomic_volume_m3_per_atom=1.8e-29,
        exchange_stoichiometry_defects_per_atom=1.0)


def _event():
    return {"cell": (6, 5), "family": 0, "burgers_sign": 1,
            "proposed_extent": .1}


def _run(mu, temperature=1100.0):
    state, args, _ = prepared_loop()
    state = replace(state, common=replace(
        state.common,
        temperature_K=np.full_like(state.common.temperature_K, temperature)))
    return (state,)+accepted_geometry_plaquette_transaction(
        state, _event(), args[3], args[4], args[1], args[5], args[6],
        _kinetics(mu), 1e-9)


def test_nonzero_exp_floor_geometry_rate_responds_to_complete_affinity():
    _, _, blocked = _run(-2.30e-18)
    _, advanced, moderate = _run(-2.00e-18)
    _, _, saturated = _run(-1.90e-18)
    assert blocked["classification"] == "AFFINITY_BLOCKED_ZERO_EVENT"
    assert blocked["affinity_biased_rate_s"] == 0.0
    assert moderate["accepted"] and moderate["downhill_activity"] < 1.0
    assert moderate["activation_enthalpy_J"] > 0.0
    assert moderate["affinity_biased_rate_s"] < saturated[
        "affinity_biased_rate_s"]
    assert moderate["event_rate_s"] == moderate["affinity_biased_rate_s"]
    assert moderate["observed_accepted_velocity_m_s"] > 0.0
    assert advanced is not None


def test_affinity_geometry_rate_has_arrhenius_temperature_response():
    low = _run(-2.0e-18, 800.0)[2]
    high = _run(-2.0e-18, 1400.0)[2]
    assert low["activation_enthalpy_J"] > 0.0
    assert high["arrhenius_unbiased_rate_s"] > low["arrhenius_unbiased_rate_s"]
    assert high["affinity_biased_rate_s"] > low["affinity_biased_rate_s"]


def test_affinity_blocked_event_is_exact_atomic_rollback():
    state, candidate, ledger = _run(-2.30e-18)
    assert candidate is state
    assert ledger["consumed_duration_s"] == 0.0
    assert ledger["committed_swept_area_m2"] == 0.0
    assert np.array_equal(
        ledger["irreversible_heat_increment_J_m3"],
        np.zeros_like(state.common.temperature_K))
    assert ledger["physical_event_count"] > 0.0
    assert ledger["physical_event_count_source"] == (
        "absolute_signed_species_exchange_count")


def test_actual_reverse_edge_reverses_exchange_and_complete_affinity():
    state, args, _ = prepared_loop()
    event = _event(); event["proposed_extent"] = .05
    forward, first = accepted_geometry_plaquette_transaction(
        state, event, args[3], args[4], args[1], args[5], args[6],
        _kinetics(-2e-18), 1e-6)
    assert first["accepted"]
    reverse_event = dict(event); reverse_event["proposed_extent"] = -.05
    restored, reverse = accepted_geometry_plaquette_transaction(
        forward, reverse_event, args[3], args[4], args[1], args[5], args[6],
        _kinetics(-2e-18), 1e-6)
    assert not reverse["accepted"]
    assert restored is forward
    np.testing.assert_allclose(
        reverse["available_energy_per_event_J"],
        -first["available_energy_per_event_J"], rtol=2e-12, atol=1e-30)
    np.testing.assert_allclose(
        reverse["affinity_probe_signed_material_exchange_count"],
        -first["signed_material_exchange_count"], rtol=2e-12, atol=1e-20)
