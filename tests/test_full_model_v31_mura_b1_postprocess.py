from pathlib import Path

from full_model.analysis.postprocess_v31_mura_b1 import (
    CHECKPOINT, relative_pair_difference,
)


def test_v31_checkpoint_identity_is_exact_step_and_strain():
    match = CHECKPOINT.match(
        Path("checkpoint_step_000001000_strain_0.03000000.npz").name)
    assert match is not None
    assert int(match.group("step")) == 1000
    assert float(match.group("strain")) == 0.03
    assert CHECKPOINT.match(".checkpoint_step_000001000_strain_0.03000000.npz.tmp") is None


def test_v31_relative_pair_difference_handles_exact_zero_control():
    assert relative_pair_difference(0.0, 0.0) == 0.0
    assert relative_pair_difference(95.0, 100.0) == 0.05
