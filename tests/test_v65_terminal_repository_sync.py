from pathlib import Path

from full_model.analysis.sync_v65_terminal_repository import synchronize


def test_terminal_sync_records_exact_checkpoint_and_claim_boundary(tmp_path):
    checkpoint = tmp_path/"checkpoint_adaptive_004000.npz"
    checkpoint.write_bytes(b"terminal")
    import hashlib
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "active_production": {},
        "live_checkpoint_monitor": {},
        "component_identity_monitor": {},
        "autonomous_finalizer": {},
        "autonomous_closure": {},
        "execution_status": "RUNNING",
        "scientific_status": "PENDING",
    }
    result = {
        "execution_status": "TARGET_REACHED",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "target_time_s": 1.25e-6,
        "final_time_s": 1.25e-6,
        "final_load": .062,
    }
    latest = {
        "adaptive_interval": 4000,
        "physical_time_s": 1.25e-6,
        "checkpoint_sha256": sha,
        "grain_20": {
            "volume_growth_fraction": .04,
            "equivalent_contour_displacement_cells": .3,
        },
        "localization": {
            "stress_Pa": 3.9e9,
            "temperature_peak_minus_mean_K": 60.0,
            "power_participation": .7,
            "heat_power_overlap": .2,
            "strict_snapshot": False,
        },
    }
    identity = {
        "status": "COMPLETE",
        "completed_pair_count": 10,
        "latest_completed_pair": {"from_interval": 3936, "to_interval": 4000},
    }

    updated, paragraph = synchronize(manifest, result, latest, identity)

    assert updated["active_production"]["terminal_checkpoint_sha256"] == sha
    assert updated["active_production"]["status"] == "TARGET_REACHED"
    assert updated["component_identity_monitor"]["status"] == "COMPLETE"
    assert updated["execution_status"] == "TARGET_REACHED"
    assert updated["scientific_status"] == "SEE_FINAL_DECISION_RECORD"
    assert updated["autonomous_finalizer"]["status"].endswith(
        "SEE_FINALIZER_STATUS"
    )
    assert "final cumulative classifier and tests remain post-processing work" in paragraph
