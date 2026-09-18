import json
from pathlib import Path


def test_v41_selected_control_holds_geometry_and_freezes_only_flow_feedback():
    path = Path("full_model/hpc3/v41_thermal_selected_case.json")
    cases = json.loads(path.read_text())
    assert len(cases) == 1
    case = cases[0]
    assert case["id"] == "heterogeneity_long_freeze_flow"
    assert case["T0_K"] == 900.0
    assert case["strain_rate_s"] == 30000.0
    assert case["particle_radius_um"] == 1.5
    assert case["conductivity_W_m_K"] == 0.15
    assert case["causal_temperature_ablation"] == "freeze_flow"
