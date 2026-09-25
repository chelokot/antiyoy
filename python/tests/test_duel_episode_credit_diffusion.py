import pytest

from python.audit_duel_episode_credit_diffusion import common_prefix_actions


def test_common_prefix_uses_both_actor_and_action() -> None:
    source = {"actors": [0, 0, 1, 0], "actions": [1, 2, 0, 3]}
    teacher = {"actors": [0, 0, 0, 1], "actions": [1, 2, 0, 3]}

    assert common_prefix_actions(source, teacher) == 2


def test_outcome_discordant_pair_requires_a_real_action_divergence() -> None:
    source = {"actors": [0, 1], "actions": [1, 0]}
    teacher = {"actors": [0, 1], "actions": [1, 0]}

    with pytest.raises(ValueError, match="no action divergence"):
        common_prefix_actions(source, teacher)
