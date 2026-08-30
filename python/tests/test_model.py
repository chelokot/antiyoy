import numpy as np
import pytest

torch = pytest.importorskip("torch")

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import (
    UniversalPolicy,
    action_distribution,
    domain_key,
    encode_rules,
    encode_rules_batch,
    load_policy_state,
    rotate_observation_180,
)


def test_domain_key_is_order_independent_and_excludes_seeds() -> None:
    first = domain_key(
        "procedural_v1",
        {"width": 17, "land_density_per_million": 700_000},
    )
    second = domain_key(
        "procedural_v1",
        {"land_density_per_million": 700_000, "width": 17},
    )

    assert first == second
    assert len(first) == 64
    with pytest.raises(ValueError, match="seed"):
        domain_key("procedural_v1", {"seed": 1})


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
    assert rules.shape == (4, 45)


def test_policy_loads_the_published_observation_v6_weights() -> None:
    original = UniversalPolicy(hidden=16, layers=1)
    legacy = dict(original.state_dict())
    legacy["action_kind_embedding.weight"] = legacy["action_kind_embedding.weight"][:5]
    legacy["action_parameter_embedding.weight"] = legacy[
        "action_parameter_embedding.weight"
    ][:5]
    legacy["rule_projection.0.weight"] = legacy["rule_projection.0.weight"][:, :42]
    del legacy["relation_embedding.weight"]
    del legacy["proposal_embedding.weight"]
    restored = UniversalPolicy(hidden=16, layers=1)

    load_policy_state(restored, legacy)

    assert torch.equal(
        restored.action_kind_embedding.weight[:5],
        original.action_kind_embedding.weight[:5],
    )
    assert torch.equal(
        restored.rule_projection[0].weight[:, :42],
        original.rule_projection[0].weight[:, :42],
    )
    assert torch.count_nonzero(restored.rule_projection[0].weight[:, 42:]) == 0
    assert torch.count_nonzero(restored.action_kind_embedding.weight[5]) == 0


def test_policy_scores_player_targeted_diplomacy_actions() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=83, diplomacy=True)
    observation = environment.observe()
    rules = encode_rules(environment.rules_json(), torch.device("cpu"))

    logits, values = UniversalPolicy(hidden=16, layers=1)(observation, rules)

    assert 5 in observation["action_kinds"]
    assert logits.shape == (int(observation["action_offsets"][-1]),)
    assert values.shape == (1,)


def test_rotating_an_observation_twice_restores_every_tensor() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=97, diplomacy=True)
    observation = environment.observe()

    restored = rotate_observation_180(rotate_observation_180(observation))

    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(restored[key], value), key


def test_rotation_preserves_diplomacy_targets_and_remaps_cells() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=101, diplomacy=True)
    observation = environment.observe()

    rotated = rotate_observation_180(observation)

    diplomacy = observation["action_kinds"] == 5
    assert np.array_equal(rotated["action_targets"][diplomacy], observation["action_targets"][diplomacy])
    cell_targets = np.logical_and(observation["action_targets"] != 65535, ~diplomacy)
    assert np.array_equal(
        rotated["action_targets"][cell_targets],
        34 - observation["action_targets"][cell_targets],
    )
