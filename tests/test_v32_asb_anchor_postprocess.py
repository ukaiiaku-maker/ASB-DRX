import numpy as np
import pytest

from full_model.analysis.postprocess_v32_asb_anchor import (
    effective_support,
    raw_conjunction,
    run_terminal_status,
    summarize_pair,
    thermal_control_semantics,
)
from full_model.production.asb_classifier import ASBSnapshot


def _snapshot(time_s, active, heating, softening, width=4.0e-7):
    return ASBSnapshot(
        time_s=time_s,
        active_fraction=active,
        temperature_excess_K=heating,
        softening_fraction=softening,
        effective_width_m=width,
    )


def test_effective_support_distinguishes_uniform_and_single_cell():
    uniform = effective_support(np.ones((4, 4)))
    localized = effective_support(np.eye(1, 16).reshape(4, 4))
    assert uniform["inverse_participation_fraction"] == 1.0
    assert uniform["entropy_effective_fraction"] == pytest.approx(1.0)
    assert localized["inverse_participation_fraction"] == pytest.approx(1.0/16.0)
    assert localized["entropy_effective_fraction"] == pytest.approx(1.0/16.0)


def test_raw_conjunction_requires_simultaneous_fixed_criteria():
    history = [
        _snapshot(0.0, 0.20, 60.0, 0.10),
        _snapshot(0.5e-6, 0.20, 60.0, 0.25),
        _snapshot(1.5e-6, 0.20, 60.0, 0.25),
    ]
    result = raw_conjunction(history, interface_width=1.0e-7)
    assert result["qualifying_snapshot_count"] == 2
    assert result["longest_conjunctive_persistence_s"] == pytest.approx(1.0e-6)


def test_incomplete_anchor_is_not_misclassified_as_mechanistic_negative():
    history = [
        _snapshot(0.0, 1.0, 0.0, 0.0),
        _snapshot(1.0e-6, 0.80, 100.0, 0.15),
    ]
    support = [effective_support(np.ones((2, 2))) for _ in history]
    result = summarize_pair(
        history, support, [0, 2700], 1.0e-7, target_steps=5000,
        strain_increment=1.0e-4,
    )
    assert result["complete"] is False
    assert result["diagnosis"] == "ANCHOR_INCOMPLETE_UNDEREXPOSURE_NOT_EXCLUDED"
    assert result["latest_nominal_strain"] == 0.2701


def test_terminal_status_recognizes_physical_validity_stop(tmp_path):
    (tmp_path/"v31_anchor_run_record.json").write_text('{"exit_code": 0}\n')
    (tmp_path/"run-from-000000.log").write_text(
        "THERMAL VALIDITY STOP at step 3123: Tmax beyond declared domain\n")
    status = run_terminal_status(tmp_path)
    assert status["terminal"] is True
    assert status["successful"] is True
    assert status["reason"] == "THERMAL_MODEL_VALIDITY_BOUNDARY"


def test_terminal_status_accepts_adaptive_runner_record(tmp_path):
    (tmp_path/"v32_adaptive_run_record.json").write_text('{"exit_code": 0}\n')
    status = run_terminal_status(tmp_path)
    assert status["terminal"] is True
    assert status["successful"] is True
    assert status["reason"] == "REQUESTED_HORIZON_OR_CLEAN_DRIVER_STOP"


def test_thermal_semantics_do_not_equate_finite_bath_with_isothermal():
    assert thermal_control_semantics({
        "k_thermal": 0.0, "T_bath_coupling": 0.0,
    })["classification"] == "NO_CONDUCTION_LOCAL_ADIABATIC"
    assert thermal_control_semantics({
        "k_thermal": 0.15, "T_bath_coupling": 0.0,
    })["classification"] == "FINITE_CONDUCTIVITY_PERIODIC_INSULATED"
    assert thermal_control_semantics({
        "k_thermal": 0.15, "T_bath_coupling": 2.0e10,
    })["classification"] == "FINITE_BATH"
    assert thermal_control_semantics({
        "heat_update_mode": "exact_prescribed_temperature",
        "k_thermal": 0.0, "T_bath_coupling": 0.0,
    })["classification"] == "EXACT_PRESCRIBED_TEMPERATURE"
