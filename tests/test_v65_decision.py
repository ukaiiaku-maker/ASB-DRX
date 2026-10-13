from full_model.analysis.write_v65_decision import markdown


def test_decision_markdown_preserves_claim_boundaries():
    payload = {
        "provenance": {
            "final_time_s": 1.25e-6,
            "production_source_commit": "abc123",
        },
        "numerical_qualification": {
            "reference_over_candidate_speedup": 1.25,
            "maximum_candidate_reference_difference_fraction": 0.001,
        },
        "physical_claims": {
            "strict_asb": False,
            "final_temperature_peak_minus_mean_K": 55.0,
            "final_power_participation": 0.7,
            "final_power_aspect_ratio": 1.2,
            "final_heat_power_overlap": 0.24,
            "final_post_peak_softening_fraction": 0.0,
            "grain_20_maximum_growth_fraction": 0.08,
            "grain_20_maximum_growth_interval": 552,
            "grain_20_final_growth_fraction": 0.04,
            "grain_20_final_displacement_cells": 0.3,
        },
        "remaining_work": ["matched control"],
    }
    text = markdown(payload)
    assert "Strict ASB is `false`" in text
    assert "not spontaneous intragranular grain birth" in text
    assert "not labeled a complete whole-history record" in text
    assert "- matched control" in text
