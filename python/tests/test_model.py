import numpy as np
import pytest

torch = pytest.importorskip("torch")

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import (
    UniversalPolicy,
    action_distribution,
    encode_rules,
    encode_rules_batch,
)


def test_policy_scores_exactly_the_legal_actions() -> None:
    environment = VectorEnv(4, width=7, height=5, seed=47)
    observation = environment.observe()
    policy = UniversalPolicy(hidden=32, layers=2)
    rules = encode_rules(environment.rules_json(), torch.device("cpu"))
    logits, values = policy(observation, rules)
    assert logits.shape == (int(observation["action_offsets"][-1]),)
    assert values.shape == (4,)
    distribution = action_distribution(logits, observation["action_offsets"])
    actions = distribution.sample()
    assert actions.shape == (4,)
    result = environment.step(actions.numpy().astype(np.uint64))
    assert result["actors"].shape == (4,)


def test_policy_conditions_each_environment_on_its_own_rules() -> None:
    environment = VectorEnv.mixed(
        [
            "classic_generic_2022",
            "classic_slay_2022",
            "online_duel_v1",
            "online_experimental_v2_260801",
        ],
        width=7,
        height=5,
        seed=71,
    )
    observation = environment.observe()
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    policy = UniversalPolicy(hidden=32, layers=2)

    logits, values = policy(observation, rules)

    assert logits.shape == (int(observation["action_offsets"][-1]),)
    assert values.shape == (4,)
    assert rules.shape == (4, 42)
