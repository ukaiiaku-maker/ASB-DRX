import json
from pathlib import Path


def test_v36_front_channel_decision_artifact_closes_all_discriminators():
    path = (Path(__file__).parents[1]/"full_model"/"verification"
            /"v36_front_channel_kinetics.json")
    result = json.loads(path.read_text())
    assert result["schema"] == "asb-drx/v36-front-channel-kinetics/v1"
    assert all(result["hard_checks"].values())
    assert not result["actual_reverse_edge"]
    assert result["frozen_v35_comparator"]["net_velocity_m_s"] == 0.0
    corrected = result["corrected_rate_channels"]
    assert corrected["a_to_b"]["acceptance_probability"] == 1.0
    assert corrected["b_to_a"]["acceptance_probability"] == 1.0
    assert (corrected["a_to_b"]["transition_state_rate_s"]
            !=corrected["b_to_a"]["transition_state_rate_s"])
    assert corrected["net_velocity_m_s"] > 0.0
    assert result["proposal_probe_control"]["states_bitwise_identical"]
    assert result["publication_accounting"]["external_work_J"] == 0.0
    assert result["publication_accounting"]["generated_heat_J"] > 0.0
