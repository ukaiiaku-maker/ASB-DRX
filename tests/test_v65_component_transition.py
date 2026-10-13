import numpy as np

from full_model.analysis.analyze_v65_power_component_transition import (
    classify_transition,
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


def test_rank_swap_does_not_splice_largest_component_identity():
    former_largest = np.zeros((12, 12), dtype=bool)
    former_largest[2:6, 2:6] = True
    former_second = np.zeros_like(former_largest)
    former_second[8:11, 8:11] = True
    retained = former_largest.copy()
    retained[5, 5] = False
    new_largest = np.zeros_like(former_largest)
    new_largest[7:11, 7:11] = True
    new_largest[6, 7] = True

    transition = transition_metrics(
        [former_largest, former_second], [new_largest, retained], spacing_m=1.0
    )
    decision = classify_transition(transition, 0.70, 0.71)

    assert transition["associated_current_component_indices"] == [1]
    assert decision["largest_component_rank_swap"]
    assert decision["classification"] == (
        "LARGEST_COMPONENT_RANK_SWAP_WITHOUT_GLOBAL_LOCALIZATION"
    )


def test_stable_component_identity_is_distinguished_from_transition():
    prior = np.zeros((10, 10), dtype=bool)
    prior[2:6, 2:6] = True
    current = prior.copy()
    current[5, 5] = False
    transition = transition_metrics([prior], [current], spacing_m=1.0)

    decision = classify_transition(transition, 0.72, 0.73)

    assert decision["stable_largest_component_identity"]
    assert decision["classification"] == (
        "STABLE_COMPONENT_IDENTITY_WITHOUT_GLOBAL_LOCALIZATION"
    )
