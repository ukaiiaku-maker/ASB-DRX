from pathlib import Path

from full_model.analysis.monitor_v65_component_identity import pending_pairs


def checkpoint(directory: Path, interval: int) -> Path:
    path = directory/f"checkpoint_adaptive_{interval:06d}.npz"
    path.touch()
    return path


def test_pending_pairs_respect_start_and_existing_atomic_output(tmp_path):
    run = tmp_path/"run"
    output = tmp_path/"analysis"
    run.mkdir(); output.mkdir()
    checkpoints = [checkpoint(run, value) for value in (3092, 3156, 3220, 3284)]
    (output/"power_component_transition_3156_3220.json").write_text("{}\n")

    pairs = pending_pairs(checkpoints, output, start_after_interval=3156)

    assert [
        (left.name, right.name, result.name) for left, right, result in pairs
    ] == [(
        "checkpoint_adaptive_003220.npz",
        "checkpoint_adaptive_003284.npz",
        "power_component_transition_3220_3284.json",
    )]
