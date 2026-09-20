import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import driving_at_time


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
