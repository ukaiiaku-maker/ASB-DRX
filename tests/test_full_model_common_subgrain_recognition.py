import numpy as np
from scipy import ndimage

from full_model.production.common_subgrain_recognition import (
    neutral_common_subgrain_handoff, recognize_common_orientation_plateau,
)
from full_model.production.common_tensorial_wall import (
    CommonWallState,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, nye_from_plastic_distortion, rotation_z,
)


def manufactured_plateau(n=96, spacing=1.0e-7):
    theta = np.zeros((n, n))
    theta[n//3:2*n//3, n//3:2*n//3] = np.deg2rad(4.0)
    beta = np.eye(3)[None, None]-rotation_z(theta)
    alpha = nye_from_plastic_distortion(beta, spacing)
    family = np.zeros((n, n, 4, 3, 3)); family[:, :, 0] = alpha
    component = theta != 0.0
    shell = np.logical_xor(
        ndimage.binary_dilation(component, iterations=2),
        ndimage.binary_erosion(component, iterations=2))
    order = np.zeros((n, n)); order[shell] = 1.0
    density = np.full((n, n), 9.0e14); density[component] = 3.0e14
    return theta, family, order, density


def zero_common(n):
    family = np.zeros((n, n, 4))
    return CommonWallState(
        *(family.copy() for _ in range(6)), np.zeros((n, n, 0)),
        np.zeros((n, n)), np.zeros((n, n)), family.copy(),
        np.zeros((n, n, 3, 3)), np.zeros((n, n, 4, 3)),
        np.zeros((n, n, 4, 3, 3)), np.zeros((n, n)),
        np.full((n, n), 1100.0))


def test_common_recognition_is_read_only_and_frank_bilby_qualified():
    theta, family, order, density = manufactured_plateau()
    snapshots = [a.copy() for a in (theta, family, order, density)]
    result = recognize_common_orientation_plateau(
        theta, family, order, density, 1.0e-7)
    assert result["qualified"]
    assert result["frank_bilby_median_projected_relative_residual"] < 0.20
    for actual, expected in zip((theta, family, order, density), snapshots):
        np.testing.assert_array_equal(actual, expected)


def test_neutral_handoff_does_not_change_common_state():
    systems = bcc_four_family_systems()
    common = zero_common(32)
    mask = np.zeros((32, 32), dtype=bool); mask[10:22, 10:22] = True
    recognition = {"qualified": True, "component_mask": mask}
    handed, audit = neutral_common_subgrain_handoff(
        common, recognition, 1.0e-7)
    assert handed.front.ledger.parent_line_processed_m == 0.0
    assert audit["physical_sweep_m3"] == 0.0
    assert audit["generated_heat_J"] == 0.0
    assert audit["maximum_relative_state_change"] <= 128*np.finfo(float).eps


def test_unqualified_state_cannot_allocate_phase_support():
    systems = bcc_four_family_systems()
    common = zero_common(16)
    try:
        neutral_common_subgrain_handoff(
            common, {"qualified": False}, 1.0e-7)
    except ValueError as error:
        assert "qualified recognition" in str(error)
    else:
        raise AssertionError("unqualified state allocated phase support")
