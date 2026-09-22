import json

from full_model.analysis.run_v50_completion_manager import (
    postprocess_pair, valid_full_duration_prefix,
)


def record(interval, stress):
    return {
        "interval": interval, "load_elapsed_time_s": interval*1e-6,
        "segments": [{"requested_duration_s": 1e-6,
                      "accepted_duration_s": 1e-6}],
        "endpoint_observables": {
            "mean_shear_stress_sigma_12_Pa": stress,
            "engineering_plastic_shear_gamma_p": interval*1e-5,
        },
        "cumulative_first_law_residual_J": interval*1e-30,
    }


def test_valid_prefix_stops_before_shortened_interval():
    manifest = {"records": [record(1, 2.0), record(2, 3.0)]}
    manifest["records"][1]["segments"][0]["accepted_duration_s"] = .5e-6
    assert valid_full_duration_prefix(manifest) == 1


def test_common_horizon_postprocessor_uses_shared_initial_state(tmp_path):
    base = {
        "completed_intervals": 2, "source_checkpoint_sha256": "shared",
        "source_sha": "source", "records": [record(1, 2.0), record(2, 4.0)],
    }
    loading = dict(base)
    hold = {**base, "records": [record(1, 1.0), record(2, 1.5)]}
    output = tmp_path/"pair.json"
    result = postprocess_pair(loading, hold, output)
    assert result["loading_minus_hold"][
        "mean_shear_stress_sigma_12_Pa"] == 2.5
    assert result["common_physical_time_s"] == 2e-6
    assert json.loads(output.read_text())["classification"] == (
        "COMMON_HORIZON_LOADING_HOLD_PAIR_COMPLETE")
