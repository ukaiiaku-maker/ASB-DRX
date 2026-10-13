import numpy as np

from full_model.analysis.analyze_v65_power_component_transition import (
    transition_metrics,
)


def test_split_fragments_recover_parent_identity():
    parent = np.zeros((8, 8), dtype=bool)
    parent[2:6, 2:6] = True
    unchanged = np.zeros_like(parent)
    unchanged[0, 0] = True
    left = np.zeros_like(parent)
    left[2:6, 2:4] = True
    right = np.zeros_like(parent)
    right[2:6, 4:6] = True

    result = transition_metrics(
        [parent, unchanged], [left, right, unchanged], spacing_m=1.0
    )

    assert result["current_component_count_change"] == 1
    assert result["associated_current_component_indices"] == [0, 1]
    assert result["old_largest_to_associated_fragment_union"]["overlap"] == 1.0
    assert result["associated_fragment_union_cell_count"] == int(parent.sum())
