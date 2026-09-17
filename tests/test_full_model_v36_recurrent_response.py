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
