from __future__ import annotations

import numpy as np
import pytest
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.puct import (
    PolicySearchConfig,
    policy_search_actions,
    root_perspective_leaf_values,
)


def uniform_evaluator(observation, _rules):
    actions = len(observation["action_kinds"])
    environments = len(observation["widths"])
    return torch.zeros(actions), torch.zeros(environments)


def test_root_perspective_values_use_root_experts_and_native_signs() -> None:
    evaluated_players: list[list[int]] = []

    def evaluator(observation, _rules):
        active_players = np.asarray(observation["active_players"])
        evaluated_players.append(active_players.tolist())
        return torch.zeros(3), torch.as_tensor(active_players + 1, dtype=torch.float32)

    observation = {
        "active_players": np.array([0, 1, 2], dtype=np.uint8),
        "action_kinds": np.array([0, 0, 0], dtype=np.uint8),
    }
    values = root_perspective_leaf_values(
        evaluator,
        observation,
        torch.zeros((3, 45)),
        torch.tensor([0.1, 0.2, 0.3]),
        np.array([2, 1, 0], dtype=np.uint8),
    )

    assert evaluated_players == [[2, 1, 0]]
    assert values.tolist() == [-3.0, 2.0, -1.0]


def test_root_perspective_values_reuse_active_evaluation_at_root() -> None:
    def unexpected_evaluator(_observation, _rules):
        raise AssertionError("identical perspectives must reuse the active evaluation")

    values = torch.tensor([0.25, -0.5])
    result = root_perspective_leaf_values(
        unexpected_evaluator,
        {"active_players": np.array([0, 1], dtype=np.uint8)},
        torch.zeros((2, 45)),
        values,
        np.array([0, 1], dtype=np.uint8),
    )

    assert result is values


def test_policy_search_is_deterministic_and_respects_active_mask() -> None:
    environment = VectorEnv(3, width=7, height=5, seed=311)
    rules = torch.zeros((3, 45))
    config = PolicySearchConfig(node_budget=32, leaf_batch_size=12)
    first_actions, first_metrics = policy_search_actions(
        environment,
        uniform_evaluator,
        rules,
        np.array([1, 0, 1], dtype=np.uint8),
        config,
        include_root_targets=True,
    )
    second_actions, second_metrics = policy_search_actions(
        environment,
        uniform_evaluator,
        rules,
        np.array([1, 0, 1], dtype=np.uint8),
        config,
        include_root_targets=True,
    )
    assert first_actions.tolist() == second_actions.tolist()
    assert first_actions[1] == 0
    assert first_metrics["nodes"].tolist() == [32, 0, 32]
    assert (
        first_metrics["root_visits"].tolist() == second_metrics["root_visits"].tolist()
    )
    assert first_metrics["evaluated_leaves"] == 64
    assert (
        first_metrics["root_action_offsets"][2]
        == first_metrics["root_action_offsets"][1]
    )
    for environment in (0, 2):
        start, end = first_metrics["root_action_offsets"][environment : environment + 2]
        assert first_metrics["root_probabilities"][int(start) : int(end)].sum() == (
            pytest.approx(1)
        )


def test_policy_search_rejects_malformed_active_mask() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=313)
    with pytest.raises(ValueError, match="one value per environment"):
        policy_search_actions(
            environment,
            uniform_evaluator,
            torch.zeros((2, 45)),
            np.array([1], dtype=np.uint8),
            PolicySearchConfig(node_budget=8),
        )
