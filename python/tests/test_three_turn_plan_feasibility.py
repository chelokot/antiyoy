import numpy as np
import pytest

from antiyoy_rl import VectorEnv
from antiyoy_rl.slate_dataset import ReplayedSlatePosition
from python.audit_three_turn_plan_feasibility import (
    candidate_decisions,
    mean_plan_log_probabilities,
    selected_plan,
)


def test_candidate_decisions_replays_complete_turn_without_mutating_root() -> None:
    root = VectorEnv(1, width=7, height=5, seed=47)
    before = root.observe()
    end_turn = int(np.flatnonzero(before["action_kinds"] == 0)[0])
    branch = root.fork(np.asarray([0], dtype=np.uint64))
    branch.step(np.asarray([end_turn], dtype=np.uint64))
    position = ReplayedSlatePosition(
        seed=47,
        record={"candidate_action_indices": [[end_turn]]},
        root=root,
        post_turn=branch.observe(),
    )

    observations, labels, offsets = candidate_decisions(position)

    np.testing.assert_array_equal(labels, [end_turn])
    np.testing.assert_array_equal(offsets, [0, 1])
    np.testing.assert_array_equal(observations["action_kinds"], before["action_kinds"])
    np.testing.assert_array_equal(root.observe()["rounds"], before["rounds"])


def test_candidate_decisions_rejects_illegal_plan_action() -> None:
    root = VectorEnv(1, width=7, height=5, seed=47)
    position = ReplayedSlatePosition(
        seed=47,
        record={"candidate_action_indices": [[100_000]]},
        root=root,
        post_turn=root.observe(),
    )

    with pytest.raises(ValueError, match="not locally legal"):
        candidate_decisions(position)


def test_selected_plan_uses_static_and_index_tie_breaks() -> None:
    assert selected_plan(np.asarray([1.0, 2.0, 2.0]), [9, 3, 4]) == 2
    assert selected_plan(np.asarray([1.0, 1.0]), [2, 2]) == 0


def test_plan_log_probabilities_are_averaged_within_each_plan() -> None:
    scores = mean_plan_log_probabilities(
        np.asarray([-0.1, -1.5, -0.1, -0.3]), np.asarray([0, 2, 4])
    )

    np.testing.assert_allclose(scores, [-0.8, -0.2])
