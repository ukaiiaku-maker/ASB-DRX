from dataclasses import fields
from dataclasses import replace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import initialize_state
from full_model.analysis.run_v67_drx_asb_overlap import (
    _physical_band, _response_snapshot, _temperature_seed,
)
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.multigrain_common_state import MultiGrainLedger


def test_checkpoint_bridge_preserves_nonzero_bath_rate():
    args = _args_from_configuration({
        "n": 16, "length_m": 4.8e-6, "interface_width_m": 6e-7,
        "temperature_K": 1000.0, "dt_s": 2e-9,
        "maximum_mechanical_subdivisions": 10,
        "thermal_control": "strong_bath", "thermal_bath_rate_s": 5e9,
    })
    assert args.thermal_control == "strong_bath"
    assert args.bath_rate == 5e9


def test_polycrystal_temperature_seed_is_mean_zero_and_owner_independent():
    state = initialize_state(
        16, 1000.0, spacing_m=3e-7, interface_width_m=6e-7)
    state = replace(
        state, ledger=MultiGrainLedger(attempted_transactions=7))
    seeded = _temperature_seed(state, 20.0, 6e-7, 3e-7, "diagonal")
    deltas = [owner.temperature_K-before.temperature_K
              for owner, before in zip(seeded.owners, state.owners)]
    assert abs(float(np.mean(deltas[0]))) < 1e-12
    assert np.ptp(deltas[0]) > 20.0
    assert seeded.ledger == state.ledger
    assert seeded.interfaces is state.interfaces
    for delta in deltas[1:]:
        np.testing.assert_array_equal(delta, deltas[0])
    for current, before in zip(seeded.owners, state.owners):
        for item in fields(current):
            if item.name != "temperature_K":
                np.testing.assert_array_equal(
                    getattr(current, item.name), getattr(before, item.name))


def test_relative_band_requires_nontrivial_plastic_flow_exposure():
    metrics = {"band_like": True}
    assert not _physical_band(metrics, 1e-8)
    assert _physical_band(metrics, .01)
    assert not _physical_band({"band_like": False}, 1.0)


def test_response_snapshot_records_stress_heat_and_physical_flow_exposure():
    class Endpoint:
        load = .1
        observables = {
            "instantaneous_plastic_power_W_m3": np.full((4, 4), 4.0),
            "common_stress_tensor_Pa": np.broadcast_to(
                np.array([[0.0, 2.0], [2.0, 0.0]]), (4, 4, 2, 2)),
            "common_temperature_K": np.arange(16.0).reshape(4, 4) + 1000.0,
        }

    snapshot = _response_snapshot(Endpoint(), 10.0)
    assert snapshot["load"] == .1
    assert snapshot["mean_shear_stress_Pa"] == 2.0
    assert snapshot["temperature_range_K"] == 15.0
    assert snapshot["plastic_to_imposed_power_fraction"] == .2
    assert not snapshot["connected_physical_band"]


def test_zero_rate_hold_cannot_be_classified_as_active_flow():
    class Endpoint:
        load = .1
        observables = {
            "instantaneous_plastic_power_W_m3": np.full((4, 4), 4.0),
            "common_stress_tensor_Pa": np.broadcast_to(
                np.array([[0.0, 2.0], [2.0, 0.0]]), (4, 4, 2, 2)),
            "common_temperature_K": np.full((4, 4), 1000.0),
        }

    snapshot = _response_snapshot(Endpoint(), 0.0)
    assert snapshot["plastic_to_imposed_power_fraction"] == 0.0
    assert not snapshot["connected_physical_band"]
