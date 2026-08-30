from __future__ import annotations

import numpy as np
import pytest
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.puct import PolicySearchConfig, policy_search_actions


def uniform_evaluator(observation, _rules):
    actions = len(observation["action_kinds"])
    environments = len(observation["widths"])
    return torch.zeros(actions), torch.zeros(environments)


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
