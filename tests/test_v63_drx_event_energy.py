from full_model.analysis.audit_v63_drx_event_energy import (
    energy_components, energy_delta,
)
from full_model.production.complete_front_energy import CompleteFrontEnergy


def test_energy_component_delta_includes_declared_physical_terms():
    before = CompleteFrontEnergy(1, 2, 3, 4, 5, 6, 7, 0)
    after = CompleteFrontEnergy(2, 4, 6, 8, 10, 12, 14, 0)
    values = energy_delta(before, after)
    assert values["defect_storage_J"] == 1
    assert values["recoverable_elastic_J"] == 4
    assert values["phase_local_J"] == 5
    assert values["phase_gradient_J"] == 6
    assert values["thermal_internal_J"] == 7
    assert values["helmholtz_J"] == 21
    assert values["internal_J"] == 28
    assert energy_components(after)["internal_J"] == 56
