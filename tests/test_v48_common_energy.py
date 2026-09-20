import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, _energy_options, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time
from full_model.production.complete_front_energy import evaluate_complete_front_energy


def _energy(context, state, driving):
    return evaluate_complete_front_energy(
        state.common_front, state.eta, spacing_m=context["spacing_m"],
        represented_thickness_m=context["represented_thickness_m"],
        **_energy_options(context, driving))


def test_current_source_mura_cycle_closes_one_common_physical_energy():
    context = resolved_bicrystal(
        grid=16, length_m=3.2e-6, interface_width_m=4e-7)
    initial = context["state"]
    driving = driving_at_time(16, .01, "hold", 0.0, 2.5e-7)
    _, audit = run_i3_cycle(
        context, initial, initial.eta.copy(), driving,
        I3Controls(
            mura_enabled=True, front_enabled=False,
            trial_dt_s=5e-7, front_dt_s=5e-7,
            mura_transport_operator="compatible_dealiased"))
    energy = audit["complete_energy"]
    physical_keys = (
        "defect_storage_J", "signed_junction_storage_J",
        "boundary_excess_J", "recoverable_elastic_J", "phase_local_J",
        "phase_gradient_J", "thermal_internal_J")
    scale = max(abs(sum(energy[side][key] for key in physical_keys))
                for side in ("before", "after"))
    assert abs(energy["first_law_residual_J"])/scale < 2e-11
    assert energy["external_work_J"] == 0.0
    # The formerly unowned locking release is now explicitly converted to
    # heat and the ordering channel has its own independent entry.
    mura = audit["mura"]
    assert mura["locking_energy_change_J_m3_cells"] < 0.0
    np.testing.assert_allclose(
        mura["locking_heat_increment_J_m3_cells"],
        -mura["locking_energy_change_J_m3_cells"], rtol=2e-12, atol=1e-9)
    np.testing.assert_allclose(
        mura["ordering_heat_increment_J_m3_cells"],
        -mura["ordering_energy_change_J_m3_cells"], rtol=2e-12, atol=1e-9)


def test_endpoint_loading_clock_closes_work_conjugate_split_path():
    context = resolved_bicrystal(
        grid=16, length_m=3.2e-6, interface_width_m=4e-7)
    initial = context["state"]
    dt = 5e-7
    d0 = driving_at_time(16, .01, "continued_deformation", 100.0, 0.0)
    dm = driving_at_time(16, .01, "continued_deformation", 100.0, .5*dt)
    d1 = driving_at_time(16, .01, "continued_deformation", 100.0, dt)
    e0 = _energy(context, initial, d0)
    em0 = _energy(context, initial, dm)
    result, audit = run_i3_cycle(
        context, initial, initial.eta.copy(), dm,
        I3Controls(
            mura_enabled=True, front_enabled=False,
            trial_dt_s=dt, front_dt_s=dt,
            mura_transport_operator="compatible_dealiased"))
    em1 = _energy(context, result, dm)
    e1 = _energy(context, result, d1)
    external_work = ((em0.recoverable_elastic_J-e0.recoverable_elastic_J)
                     +(e1.recoverable_elastic_J-em1.recoverable_elastic_J))
    residual = (e1.internal_J-e0.internal_J)-external_work
    scale = max(abs(e0.internal_J), abs(e1.internal_J), abs(external_work))
    assert abs(residual)/scale < 2e-11
    np.testing.assert_allclose(
        d1.mean_strain[0, 1]-d0.mean_strain[0, 1], 100.0*dt,
        rtol=0.0, atol=2e-18)
    assert audit["mura"]["accepted_dt_s"] == dt
