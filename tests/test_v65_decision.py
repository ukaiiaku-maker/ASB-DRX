import json

from full_model.analysis.write_v65_decision import markdown, reduce


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


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


def test_final_decision_reduces_complete_evidence_schema(tmp_path):
    campaign = tmp_path
    analysis = campaign/"analysis"
    run = campaign/"runs/production_1p035_to1p25us_6debbe0"
    checkpoint = run/"checkpoint_adaptive_003000.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"terminal-state")
    write(campaign/"finalizer_status.json", {
        "status": "COMPLETE",
        "all_required_artifacts_present": True,
        "step_exit_codes": {"summary": 0, "canonical_tests": 0},
    })
    write(run/"result.json", {
        "execution_status": "TARGET_REACHED",
        "checkpoint": str(checkpoint),
        "source_commit": "production",
        "final_time_s": 1.25e-6,
        "target_time_s": 1.25e-6,
        "final_load": .062,
    })
    write(analysis/"production_to1p25us_summary.json", {"runs": [{
        "maximum_accepted_tolerance_ratio": .85,
        "accepted_intervals": 400,
        "runner_wall_time_s": 1000.0,
        "runner_wall_seconds_per_physical_ns": 140.0,
    }]})
    write(analysis/"final_asb_through1p25us.json", {
        "rows": [{
            "temperature_max_minus_mean_K": 60.0,
            "plastic_power": {"inverse_participation_fraction": .7},
            "power_aspect_ratio": 1.2,
            "heat_power_component_overlap": .24,
        }],
        "physical_episode": {
            "inherited_strict_asb": False,
            "episodes": [],
            "maximum_episode_duration_s": 0.0,
            "softening_summary": {
                "final_qualifying_operator_segment_fraction": 0.0},
            "missing_or_failed_requirements": ["matched control"],
        },
        "scalar_history": {
            "history_complete": False,
            "gaps": [{"after_interval": 2184, "before_interval": 2457}],
            "semantics": "available accepted-time history with declared gap",
        },
    })
    write(analysis/"final_grain20_selection.json", {
        "rows": [{
            "volume_growth_fraction": .04,
            "equivalent_contour_displacement_cells": .3,
        }],
        "volume_history_extrema": {"maximum_attained": {
            "volume_growth_fraction": .08,
            "adaptive_interval": 552,
        }},
    })
    write(analysis/"work_precision_1p030_to1p035us.json", {
        "candidate_is_faster": True,
        "candidate_agrees_with_reference_at_working_tolerance": True,
        "wall_time_speedup_reference_over_candidate": 1.29,
        "endpoint_metric_differences": {
            "supports": {"relative_difference": 1e-4},
            "power": {"relative_difference": .0019},
        },
    })
    write(analysis/"power_component_transition_fixture.json", {
        "decision": {
            "classification": "THRESHOLD_COMPONENT_SPLIT_WITHOUT_GLOBAL_LOCALIZATION",
            "strict_asb_claim": False,
        },
    })
    write(analysis/"live_monitor/stress_curvature_latest.json", {
        "decision": {
            "classification": "CONSISTENT_QUADRATIC_PEAK_PROJECTION_NOT_OBSERVED",
            "stress_peak_observed": False,
        },
        "fits": [
            {"projected_peak_time_s": 1.1466e-6},
            {"projected_peak_time_s": 1.1468e-6},
        ],
    })
    payload = reduce(campaign)
    assert payload["scoped_experiment_complete"] is True
    assert payload["full_scientific_objective_complete"] is False
    assert payload["localization_identity_audits"][0]["classification"] == (
        "THRESHOLD_COMPONENT_SPLIT_WITHOUT_GLOBAL_LOCALIZATION"
    )
    assert payload["preterminal_stress_peak_projection"]["decision"][
        "stress_peak_observed"] is False
    assert payload["numerical_qualification"][
        "production_candidate_qualified"] is True
    assert payload["physical_claims"] == {
        "prepared_boundary_transient_growth_then_recession": True,
        "spontaneous_intragranular_grain_birth": False,
        "strict_asb": False,
        "asb_episode_count": 0,
        "maximum_asb_episode_duration_s": 0.0,
        "final_post_peak_softening_fraction": 0.0,
        "final_temperature_peak_minus_mean_K": 60.0,
        "final_power_participation": .7,
        "final_power_aspect_ratio": 1.2,
        "final_heat_power_overlap": .24,
        "grain_20_maximum_growth_fraction": .08,
        "grain_20_maximum_growth_interval": 552,
        "grain_20_final_growth_fraction": .04,
        "grain_20_final_displacement_cells": .3,
    }
