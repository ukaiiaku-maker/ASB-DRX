from full_model.analysis.audit_v65_completion import (
    EXPECTED_PRODUCTION_SOURCE,
    EXPECTED_V64_CHECKPOINT,
    evaluate_records,
)


def complete_records():
    return {
        "manifest": {"lineage": {
            "terminal_checkpoint_sha256": EXPECTED_V64_CHECKPOINT,
        }},
        "result": {
            "source_commit": EXPECTED_PRODUCTION_SOURCE,
            "execution_status": "TARGET_REACHED",
            "final_time_s": 1.25e-6,
            "target_time_s": 1.25e-6,
        },
        "finalizer": {
            "status": "COMPLETE",
            "all_required_artifacts_present": True,
            "step_exit_codes": {"summary": 0, "canonical_tests": 0},
        },
        "closure": {"status": "COMPLETE"},
        "decision": {
            "scoped_experiment_complete": True,
            "full_scientific_objective_complete": False,
        },
        "identity_monitor": {"status": "COMPLETE"},
    }


def test_completion_records_require_every_scoped_terminal_condition():
    checks = evaluate_records(complete_records())
    assert checks
    assert all(checks.values())


def test_running_campaign_is_not_misclassified_complete():
    records = complete_records()
    records["result"] = None
    records["finalizer"] = None
    records["closure"] = None
    records["decision"] = None
    records["identity_monitor"] = {"status": "RUNNING"}
    checks = evaluate_records(records)
    assert not checks["production_target_reached"]
    assert not checks["finalizer_complete"]
    assert not checks["closure_complete"]
    assert not checks["component_identity_monitor_complete"]
