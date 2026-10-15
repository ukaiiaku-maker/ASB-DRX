import numpy as np

from full_model.production.orientation_wall_compatibility import (
    desired_orientation_gradient_from_nye,
    reconstruct_orientation_from_ordered_nye,
)


def test_compatible_gradient_reconstruction_reduces_manufactured_mismatch():
    n = 32
    spacing = 1.0e-7
    x = np.arange(n)[:, None]*spacing
    y = np.arange(n)[None, :]*spacing
    theta = 0.03*np.sin(2*np.pi*x/(n*spacing))*np.sin(
        2*np.pi*y/(n*spacing))
    gx = (np.roll(theta, -1, 0)-np.roll(theta, 1, 0))/(2*spacing)
    gy = (np.roll(theta, -1, 1)-np.roll(theta, 1, 1))/(2*spacing)
    c, s = np.cos(theta), np.sin(theta)
    dR_ey = np.stack((-c, -s, np.zeros_like(theta)), axis=-1)
    dR_ex = np.stack((-s, c, np.zeros_like(theta)), axis=-1)
    alpha = np.zeros(theta.shape+(3, 3))
    alpha[..., :, 2] = gx[..., None]*dR_ey-gy[..., None]*dR_ex
    desired = desired_orientation_gradient_from_nye(alpha, theta)
    np.testing.assert_allclose(desired[..., 0], gx, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(desired[..., 1], gy, rtol=0.0, atol=1e-9)
    initial = np.zeros_like(theta)
    reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        alpha, initial, np.ones_like(theta, dtype=bool), spacing)
    assert np.std(reconstructed) > 0.01
    correlation = np.corrcoef(theta.ravel(), reconstructed.ravel())[0, 1]
    assert correlation > 0.98


def test_reconstruction_preserves_each_connected_rigid_rotation():
    theta = np.zeros((12, 12))
    domain = np.zeros_like(theta, dtype=bool)
    domain[1:5, 1:5] = True
    domain[7:11, 7:11] = True
    theta[1:5, 1:5] = .2
    theta[7:11, 7:11] = -.3
    alpha = np.zeros(theta.shape+(3, 3))
    reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        alpha, theta, domain, 1e-7)
    np.testing.assert_allclose(np.mean(reconstructed[1:5, 1:5]), .2)
    np.testing.assert_allclose(np.mean(reconstructed[7:11, 7:11]), -.3)
    np.testing.assert_array_equal(reconstructed[~domain], theta[~domain])
