import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl import ProceduralConfig, VectorEnv
from antiyoy_rl.model import (
    UniversalPolicy,
    action_distribution,
    concatenate_observations,
    domain_key,
    encode_rules,
    encode_rules_batch,
    load_policy_state,
    rotate_observation_180,
    select_environments,
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


def test_native_rule_encoder_preserves_the_complete_feature_order() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=47)
    raw = torch.tensor(
        [
            2,
            10,
            1,
            5,
            10,
            0,
            2,
            6,
            18,
            36,
            12,
            2,
            15,
            35,
            1,
            6,
            10,
            3,
            4,
            4,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            2,
            0.2,
            0.3,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
        ],
        dtype=torch.float32,
    )
    expected = torch.sign(raw) * torch.log1p(torch.abs(raw))

    actual = encode_rules(environment.rules_json(), torch.device("cpu"))

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


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


def test_policy_batches_heterogeneous_map_dimensions() -> None:
    profiles = [
        "classic_generic_2022",
        "online_default_v1",
        "classic_slay_2022",
        "online_duel_v1",
    ]
    generators = [
        ProceduralConfig(width=19, height=15, players=5, seed=101),
        ProceduralConfig(width=21, height=15, players=6, seed=102),
        ProceduralConfig(width=19, height=15, players=5, seed=103),
        ProceduralConfig(width=23, height=17, players=7, seed=104),
    ]
    environment = VectorEnv.procedural_domains(profiles, generators)
    observation = environment.observe()
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    policy = UniversalPolicy(hidden=16, layers=1)

    logits, values = policy(observation, rules)
    (logits.square().mean() + values.square().mean()).backward()
    distribution = action_distribution(logits.detach(), observation["action_offsets"])
    result = environment.step(distribution.sample().numpy().astype(np.uint64))

    assert logits.shape == (int(observation["action_offsets"][-1]),)
    assert values.shape == (4,)
    assert policy.action_head[-1].weight.grad is not None
    assert result["actors"].shape == (4,)


def test_selected_observations_recombine_without_changing_policy_outputs() -> None:
    environment = VectorEnv(3, width=7, height=5, seed=59)
    observation = environment.observe()
    selected = [select_environments(observation, [index]) for index in (2, 0, 1)]
    recombined = concatenate_observations(selected)
    policy = UniversalPolicy(hidden=16, layers=1)
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))

    logits, values = policy(recombined, rules[[2, 0, 1]])

    assert logits.shape == (int(recombined["action_offsets"][-1]),)
    assert values.shape == (3,)
    assert recombined["active_players"].tolist() == [0, 0, 0]


def test_policy_loads_the_published_observation_v6_weights() -> None:
    original = UniversalPolicy(hidden=16, layers=1)
    with torch.no_grad():
        original.action_kind_embedding.weight[5].zero_()
        original.action_parameter_embedding.weight[5].zero_()
        original.rule_projection[0].weight[:, 42:].zero_()
    legacy = dict(original.state_dict())
    legacy["action_kind_embedding.weight"] = legacy["action_kind_embedding.weight"][:5]
    legacy["action_parameter_embedding.weight"] = legacy[
        "action_parameter_embedding.weight"
    ][:5]
    legacy["rule_projection.0.weight"] = legacy["rule_projection.0.weight"][:, :42]
    del legacy["relation_embedding.weight"]
    del legacy["proposal_embedding.weight"]
    for key in (
        "turn_distance_embedding.weight",
        "cell_relation_embedding.weight",
        "player_count_embedding.weight",
        "round_projection.weight",
    ):
        del legacy[key]
    restored = UniversalPolicy(hidden=16, layers=1)

    load_policy_state(restored, legacy)
    environment = VectorEnv(2, width=7, height=5, seed=79)
    observation = environment.observe()
    rules = encode_rules(environment.rules_json(), torch.device("cpu"))
    original_logits, original_values = original(observation, rules)
    restored_logits, restored_values = restored(observation, rules)

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
    assert torch.count_nonzero(restored.turn_distance_embedding.weight) == 0
    assert torch.count_nonzero(restored.cell_relation_embedding.weight) == 0
    assert torch.count_nonzero(restored.player_count_embedding.weight) == 0
    assert torch.count_nonzero(restored.round_projection.weight) == 0
    assert torch.equal(restored_logits, original_logits)
    assert torch.equal(restored_values, original_values)


def test_policy_scores_player_targeted_diplomacy_actions() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=83, diplomacy=True)
    observation = environment.observe()
    rules = encode_rules(environment.rules_json(), torch.device("cpu"))

    policy = UniversalPolicy(hidden=16, layers=1)
    logits, values = policy(observation, rules)
    (logits.square().mean() + values.square().mean()).backward()

    assert 5 in observation["action_kinds"]
    assert logits.shape == (int(observation["action_offsets"][-1]),)
    assert values.shape == (1,)
    assert torch.count_nonzero(policy.turn_distance_embedding.weight.grad) > 0
    assert torch.count_nonzero(policy.cell_relation_embedding.weight.grad) > 0
    assert torch.count_nonzero(policy.player_count_embedding.weight.grad) > 0
    assert torch.count_nonzero(policy.round_projection.weight.grad) > 0


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
    assert np.array_equal(
        rotated["action_targets"][diplomacy], observation["action_targets"][diplomacy]
    )
    cell_targets = np.logical_and(observation["action_targets"] != 65535, ~diplomacy)
    assert np.array_equal(
        rotated["action_targets"][cell_targets],
        34 - observation["action_targets"][cell_targets],
    )
