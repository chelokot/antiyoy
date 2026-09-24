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


def test_model_reply_search_simulates_past_live_game_action_limit() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=29, action_limit=1)
    agent = ModelReplySearch(
        node_budget=64,
        slate_size=4,
        audit_native_replies=True,
        audit_round_modulus=1,
    )

    chosen = agent.actions(
        environment,
        EndTurnPolicy(),
        torch.zeros((1, 45)),
        np.asarray([True], dtype=np.bool_),
    )

    assert agent.candidate_replies > 0
    assert len(agent.native_reply_records) == 1
    assert environment.step(chosen)["truncated"].tolist() == [1]


def test_native_reply_audit_preserves_hybrid_turn() -> None:
    plain_environment = VectorEnv(1, width=7, height=5, seed=29)
    audited_environment = VectorEnv(1, width=7, height=5, seed=29)
    plain = ModelReplySearch(node_budget=64, slate_size=4)
    audited = ModelReplySearch(
        node_budget=64,
        slate_size=4,
        audit_native_replies=True,
        audit_reply_nodes=64,
        audit_round_modulus=1,
    )
    root = int(plain_environment.observe()["active_players"][0])
    active = np.asarray([True], dtype=np.bool_)
    rules = torch.zeros((1, 45))
    first_audited_action = None

    while int(plain_environment.observe()["active_players"][0]) == root:
        chosen = plain.actions(plain_environment, EndTurnPolicy(), rules, active)
        audited_chosen = audited.actions(
            audited_environment, EndTurnPolicy(), rules, active
        )
        if first_audited_action is None:
            first_audited_action = int(audited_chosen[0])
        np.testing.assert_array_equal(chosen, audited_chosen)
        plain_environment.step(chosen)
        audited_environment.step(audited_chosen)
        if plain_environment.done()[0]:
            break

    assert len(audited.native_reply_records) == 1
    record = audited.native_reply_records[0]
    assert len(record["candidate_plans"]) == len(record["native_reply_scores"])
    assert record["candidate_plans"][record["autonomous_selected_index"]][0] == (
        first_audited_action
    )
    assert record["autonomous_selected_index"] == max(
        range(len(record["autonomous_reply_scores"])),
        key=lambda index: (
            record["autonomous_reply_scores"][index],
            record["static_scores"][index],
            -index,
        ),
    )
    assert record["native_selected_index"] == max(
        range(len(record["native_reply_scores"])),
        key=lambda index: (
            record["native_reply_scores"][index],
            record["static_scores"][index],
            -index,
        ),
    )
    assert not plain.native_reply_records
