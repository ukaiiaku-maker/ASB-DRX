import numpy as np

from full_model.analysis.run_v36_recurrent_physical_response import (
    geometric_envelope, run_response,
)


def test_geometric_envelope_preserves_simplex_and_has_declared_direction():
    child = np.zeros((8, 8))
    child[2:6] = 1.0
    eta = np.stack((1.0-child, child), axis=2)
    expanded = geometric_envelope(eta, fraction=0.25, direction=1)
    contracted = geometric_envelope(eta, fraction=0.25, direction=-1)
    np.testing.assert_allclose(np.sum(expanded, axis=2), 1.0)
    np.testing.assert_allclose(np.sum(contracted, axis=2), 1.0)
    assert np.all(expanded >= 0.0) and np.all(expanded <= 1.0)
    assert np.all(contracted >= 0.0) and np.all(contracted <= 1.0)
    assert np.sum(expanded[:, :, 1]) > np.sum(child)
    assert np.sum(contracted[:, :, 1]) < np.sum(child)


def test_response_checkpoint_restart_reproduces_second_interval(tmp_path):
    continuous_dir = tmp_path/"continuous"
    split_dir = tmp_path/"split"
    continuous = run_response(
        output_dir=continuous_dir, protocol="hold", intervals=2,
        checkpoint_every=1)
    run_response(output_dir=split_dir, protocol="hold", intervals=1,
                 checkpoint_every=1)
    restarted = run_response(
        output_dir=split_dir, protocol="hold", intervals=2,
        checkpoint_every=1, resume=split_dir/"checkpoint_000001.npz")
    assert restarted["records"] == continuous["records"]
    assert restarted["physical_time_s"] == continuous["physical_time_s"]
    assert restarted["cumulative_signed_sweep_m3"] == continuous[
        "cumulative_signed_sweep_m3"]


def test_response_records_compatible_transport_and_misoriented_boundary(tmp_path):
    result = run_response(
        output_dir=tmp_path/"misoriented", protocol="continued_deformation",
        intervals=1, checkpoint_every=1, misorientation_deg=20.0,
        mura_transport_operator="compatible_dealiased",
        qualified_midpoint_loading=True)
    assert result["configuration"]["mura_transport_operator"] == (
        "compatible_dealiased")
    assert result["boundary_initialization"]["kind"] == "misoriented_bicrystal"
    row = result["records"][0]
    assert row["mura"]["transport_operator"] == "compatible_dealiased"
    assert row["cumulative_newly_swept_volume_m3"] >= 0.0
    assert row["loading_energy_audit"]["first_law_passed"]


def test_declared_dt_transition_preserves_clock_and_is_audited(tmp_path):
    output = tmp_path/"transition"
    first = run_response(
        output_dir=output, protocol="hold", intervals=1, dt_s=2.0e-9,
        checkpoint_every=1)
    continued = run_response(
        output_dir=output, protocol="hold", intervals=2, dt_s=4.0e-9,
        checkpoint_every=1, resume=output/"checkpoint_000001.npz",
        allow_dt_transition=True)
    assert continued["physical_time_s"] == first["physical_time_s"]+4.0e-9
    assert continued["numerical_method_transitions"] == [{
        "kind": "declared_common-clock_dt_transition",
        "completed_intervals_at_transition": 1,
        "physical_time_at_transition_s": first["physical_time_s"],
        "old_dt_s": 2.0e-9,
        "new_dt_s": 4.0e-9,
        "state_reset": False,
        "loading_origin_reset": False,
        "cumulative_work_reset": False,
    }]


def test_declared_loading_to_hold_transition_preserves_endpoint_load(tmp_path):
    output = tmp_path/"loading_transition"
    first = run_response(
        output_dir=output, protocol="continued_deformation", intervals=1,
        dt_s=2.0e-9, initial_shear=0.01, strain_rate_s=1.0e3,
        checkpoint_every=1, qualified_midpoint_loading=True)
    held_shear = 0.01+1.0e3*first["physical_time_s"]
    continued = run_response(
        output_dir=output, protocol="hold", intervals=2, dt_s=2.0e-9,
        initial_shear=held_shear, strain_rate_s=1.0e3,
        checkpoint_every=1, qualified_midpoint_loading=True,
        resume=output/"checkpoint_000001.npz",
        allow_loading_transition=True)
    transition = continued["numerical_method_transitions"][-1]
    assert transition["new_protocol"] == "hold"
    assert transition["continuous_from_prior_endpoint"]
    assert continued["records"][-1]["mean_shear_strain"] == held_shear
