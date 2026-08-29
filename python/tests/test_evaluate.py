import numpy as np
import pytest

pytest.importorskip("torch")

from python.evaluate import named_action_counts, paired_elo, selected_action_kinds
from python.evaluate_suite import aggregate_results


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


def test_suite_aggregate_counts_draws_and_truncations() -> None:
    aggregate = aggregate_results(
        [
            {
                "games": 4,
                "wins": 3,
                "draws": 1,
                "losses": 0,
                "truncations": 1,
                "terminal_draws": 0,
            },
            {
                "games": 4,
                "wins": 1,
                "draws": 0,
                "losses": 3,
                "truncations": 0,
                "terminal_draws": 0,
            },
        ]
    )

    assert aggregate["games"] == 8
    assert aggregate["score"] == 0.5625
    assert aggregate["truncations"] == 1
    assert aggregate["relative_elo"] == paired_elo(0.5625, 8)
