import json

import numpy as np
import pytest

from full_model.production.asb_physical_ledger import (
    CHANNEL_NAMES, PhysicalEnergyState, accepted_channel,
    accepted_periodic_power_channel,
    build_production_step_ledger, unavailable_channel,
)


def _channels(value=1.0):
    return {
        name: accepted_channel(name, value, 1.0, source="manufactured accepted event")
        for name in CHANNEL_NAMES
    }


def test_all_seven_channels_are_typed_and_not_a_residual_bucket():
    assert CHANNEL_NAMES == (
        "plastic_drag", "mobile_forest_recovery",
        "neutral_pair_annihilation", "junction_relaxation",
        "boundary_recovery", "thermal_conduction", "declared_sinks")
    assert "other_dissipation" not in CHANNEL_NAMES
    channels = _channels()
    assert set(channels) == set(CHANNEL_NAMES)
    assert all(channel.available for channel in channels.values())


def test_spatial_affinity_extent_product_is_evaluated_before_averaging():
    affinity = np.array([1.0, 3.0])
    extent = np.array([4.0, 2.0])
    channel = accepted_channel(
        "plastic_drag", affinity, extent, source="accepted fields")
    assert channel.dissipation_W_m3 == pytest.approx(5.0)
    assert channel.dissipation_W_m3 != pytest.approx(
        affinity.mean()*extent.mean())


def test_slip_family_power_is_summed_not_family_averaged():
    affinity = np.ones((3, 4))*2.0
    extent = np.ones((3, 4))*5.0
    channel = accepted_channel(
        "plastic_drag", affinity, extent, source="accepted fields",
        component_axis=-1)
    assert channel.dissipation_W_m3 == pytest.approx(40.0)


def test_missing_physical_owner_fails_closed():
    channels = _channels(0.0)
    channels["junction_relaxation"] = unavailable_channel(
        "junction_relaxation", "no declared production affinity")
    zero = PhysicalEnergyState()
    ledger = build_production_step_ledger(
        channels=channels, energy_before=zero, energy_after=zero,
        external_work_J_m3=0.0, deposited_heat_J_m3=0.0,
        exported_heat_J_m3=0.0, dt_s=1.0)
    assert not ledger.all_channels_available
    assert ledger.dissipation_J_m3 == 0.0


def test_constraint_energy_cannot_change_physical_first_law():
    before = PhysicalEnergyState(
        local_line_correlation_J_m3=2.0,
        numerical_augmented_constraints_J_m3=-1e23)
    after = PhysicalEnergyState(
        local_line_correlation_J_m3=3.0,
        thermal_J_m3=4.0,
        numerical_augmented_constraints_J_m3=1e23)
    ledger = build_production_step_ledger(
        channels=_channels(0.0), energy_before=before, energy_after=after,
        external_work_J_m3=5.0, deposited_heat_J_m3=0.0,
        exported_heat_J_m3=0.0, dt_s=1.0)
    assert ledger.first_law_residual_J_m3 == 0.0
    assert ledger.numerical_constraint_change_J_m3 == 2e23


def test_heat_is_not_double_counted_as_both_dissipation_and_storage():
    channels = _channels(0.0)
    channels["plastic_drag"] = accepted_channel(
        "plastic_drag", 4.0, 1.0, source="accepted slip")
    before = PhysicalEnergyState()
    after = PhysicalEnergyState(thermal_J_m3=4.0)
    ledger = build_production_step_ledger(
        channels=channels, energy_before=before, energy_after=after,
        external_work_J_m3=4.0, deposited_heat_J_m3=4.0,
        exported_heat_J_m3=0.0, dt_s=1.0)
    assert ledger.first_law_residual_J_m3 == 0.0
    assert ledger.dissipation_heat_residual_J_m3 == 0.0


def test_conduction_and_bath_exergy_are_not_counted_as_deposited_heat():
    channels = _channels(0.0)
    channels["thermal_conduction"] = accepted_channel(
        "thermal_conduction", 3.0, 2.0, source="Fourier exergy")
    channels["declared_sinks"] = accepted_channel(
        "declared_sinks", 4.0, 2.0, source="bath exergy")
    zero = PhysicalEnergyState()
    ledger = build_production_step_ledger(
        channels=channels, energy_before=zero, energy_after=zero,
        external_work_J_m3=0.0, deposited_heat_J_m3=0.0,
        exported_heat_J_m3=0.0, dt_s=1.0)
    assert ledger.dissipation_J_m3 == 14.0
    assert ledger.dissipation_heat_residual_J_m3 == 0.0


def test_periodic_transport_power_may_be_local_signed_but_not_globally_negative():
    channel = accepted_periodic_power_channel(
        "plastic_drag", np.array([-2.0, 4.0, 7.0]),
        source="manufactured periodic chemical-energy transport")
    assert channel.dissipation_W_m3 == pytest.approx(3.0)
    with pytest.raises(ValueError):
        accepted_periodic_power_channel(
            "plastic_drag", np.array([-4.0, 1.0]),
            source="globally inadmissible periodic transport")


def test_homogeneous_zero_process_and_json_restart_payload_are_exact():
    state = PhysicalEnergyState(thermal_J_m3=9.0)
    ledger = build_production_step_ledger(
        channels=_channels(0.0), energy_before=state, energy_after=state,
        external_work_J_m3=0.0, deposited_heat_J_m3=0.0,
        exported_heat_J_m3=0.0, dt_s=2e-9)
    payload = ledger.to_dict()
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert ledger.first_law_residual_J_m3 == 0.0
    assert ledger.dissipation_J_m3 == 0.0


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_negative_or_nonfinite_channel_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        accepted_channel(
            "boundary_recovery", bad, 1.0, source="manufactured")
