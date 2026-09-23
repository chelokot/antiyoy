import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model_reply_search import ModelReplySearch


class EndTurnPolicy:
    def actions(
        self, observation: dict[str, np.ndarray], rule_features: torch.Tensor
    ) -> np.ndarray:
        return np.zeros(len(observation["widths"]), dtype=np.uint64)


def test_model_reply_search_caches_a_complete_legal_root_turn() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=29)
    agent = ModelReplySearch(node_budget=64, slate_size=4)
    policy = EndTurnPolicy()
    rules = torch.zeros((1, 45))
    root = int(environment.observe()["active_players"][0])

    while int(environment.observe()["active_players"][0]) == root:
        action = agent.actions(
            environment,
            policy,
            rules,
            np.asarray([True], dtype=np.bool_),
        )
        assert int(action[0]) < int(np.diff(environment.observe()["action_offsets"])[0])
        environment.step(action)
        if environment.done()[0]:
            break

    assert agent.root_turns == 1
    assert 1 <= agent.candidate_replies <= 4
    assert not agent.pending
    assert len(agent.decision_times_seconds) == 1
    assert agent.decision_times_seconds[0] > 0


def test_model_reply_search_skips_inactive_games() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=29)
    agent = ModelReplySearch(node_budget=64, slate_size=4)

    actions = agent.actions(
        environment,
        EndTurnPolicy(),
        torch.zeros((2, 45)),
        np.asarray([False, True], dtype=np.bool_),
    )

    assert actions[0] == 0
    assert agent.root_turns == 1
    assert len(agent.pending) <= 1
