from full_model.analysis.audit_v67_poly_source_equivalence import physical_changes


def test_source_equivalence_ignores_analysis_and_evidence_only_changes():
    assert physical_changes([
        "campaign_status.md",
        "full_model/analysis/analyze_v67_poly_rx_asb_long_horizon.py",
        "full_model/verification/example/result.json",
        "tests/test_example.py",
    ]) == []


def test_source_equivalence_detects_production_and_driver_changes():
    assert physical_changes([
        "full_model/production/flow.py",
        "full_model/analysis/run_v67_drx_asb_overlap.py",
        "full_model/analysis/unrelated_plot.py",
    ]) == [
        "full_model/analysis/run_v67_drx_asb_overlap.py",
        "full_model/production/flow.py",
    ]
