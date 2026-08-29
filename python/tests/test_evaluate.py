import numpy as np
import pytest

pytest.importorskip("torch")

from python.evaluate import named_action_counts, selected_action_kinds


def test_selected_action_kinds_resolves_local_ragged_indices() -> None:
    observation = {
        "action_offsets": np.array([0, 2, 5], dtype=np.uint64),
        "action_kinds": np.array([0, 1, 0, 2, 3], dtype=np.uint8),
    }

    kinds = selected_action_kinds(
        observation,
        np.array([1, 2], dtype=np.uint64),
    )

    assert kinds.tolist() == [1, 3]


def test_named_action_counts_preserves_zero_categories() -> None:
    counts = named_action_counts(np.array([7, 5, 3, 2, 1, 0], dtype=np.int64))

    assert counts == {
        "end_turn": 7,
        "move": 5,
        "recruit": 3,
        "build": 2,
        "plant_tree": 1,
        "diplomacy": 0,
    }
