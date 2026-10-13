import json

from full_model.analysis.summarize_v65_live_trends import load_rows, summarize


def row(index):
    return {
        "adaptive_interval": index,
        "physical_time_s": float(index),
        "applied_shear_strain": .1*index,
        "shear_stress_Pa": 10.0+index,
        "temperature_mean_K": 900.0+2.0*index,
        "temperature_contrast_K": 3.0*index,
        "grain_volume_m3": [2.0, 10.0-index],
        "accepted_duration_s": 1.0,
        "wall_time_s": 4.0,
        "maximum_tolerance_ratio": .8,
        "accuracy_decision": {"attempts": [{"metric_errors": {
            "power_change": {"normalized_error": .016,
                             "relative_tolerance": .02}}}]},
    }


def test_live_trend_regime_and_eta(tmp_path):
    path = tmp_path/"telemetry.jsonl"
    path.write_text("\n".join(json.dumps(row(i)) for i in range(1, 5))+"\n")
    rows, sources = load_rows([path])
    result = summarize(rows, sources, window=3, target_time_s=6.0,
                       grain_index=1)
    assert result["regime"] == {
        "stress_hardening": True, "mean_heating": True,
        "contrast_increasing": True, "target_grain_receding": True}
    assert result["throughput"]["estimated_remaining_wall_time_s"] == 8.0
    assert result["accuracy"]["limiting_metric_counts"] == {
        "power_change": 3}
