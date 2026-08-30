import json

import numpy as np
import pytest

from antiyoy_rl import (
    GENERATOR_SCHEMA_VERSION,
    OBJECTIVE_SCHEMA_VERSION,
    OBSERVATION_VERSION,
    ProceduralConfig,
    ScenarioObjective,
    VectorEnv,
)


def test_observation_and_step_contract() -> None:
    environment = VectorEnv(4, width=7, height=5, seed=47)
    observation = environment.observe()
    assert observation["version"] == OBSERVATION_VERSION
    assert observation["cell_offsets"].tolist() == [0, 35, 70, 105, 140]
    assert observation["relation_offsets"].tolist() == [0, 0, 0, 0, 0]
    assert observation["player_counts"].tolist() == [2, 2, 2, 2]
    assert observation["relations"].shape == (0,)
    assert observation["proposals"].shape == (0,)
    assert observation["owners"].shape == (140,)
    assert observation["visible"].shape == (140,)
    assert set(np.unique(observation["visible"])).issubset({0, 1})
    actions = np.zeros(4, dtype=np.uint64)
    result = environment.step(actions)
    assert result["actors"].tolist() == [0, 0, 0, 0]
    assert result["terminal"].tolist() == [0, 0, 0, 0]
    assert result["adjudicated_winners"].tolist() == [255, 255, 255, 255]
    assert json.loads(environment.rules_json())["schema_version"] == 5


def test_truncation_reports_deterministic_adjudication() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=47, action_limit=1)

    result = environment.step(np.zeros(1, dtype=np.uint64))

    assert result["truncated"].tolist() == [1]
    assert result["winners"].tolist() == [255]
    assert result["adjudicated_winners"].shape == (1,)


def test_seeded_environments_are_equal_after_equal_actions() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=91)
    environment.reset(1, 91)
    for _ in range(20):
        observation = environment.observe()
        offsets = observation["action_offsets"]
        first_count = int(offsets[1] - offsets[0])
        second_count = int(offsets[2] - offsets[1])
        assert first_count == second_count
        selected = first_count // 2
        environment.step(np.array([selected, selected], dtype=np.uint64))
        observation = environment.observe()
        assert np.array_equal(observation["owners"][:35], observation["owners"][35:])
        assert np.array_equal(observation["objects"][:35], observation["objects"][35:])
        if any(environment.done()):
            break


def test_greedy_baseline_returns_legal_local_indices() -> None:
    environment = VectorEnv(4, width=7, height=5, seed=19)
    observation = environment.observe()
    actions = np.asarray(environment.greedy_actions(), dtype=np.uint64)
    counts = np.diff(observation["action_offsets"])
    assert np.all(actions < counts)
    environment.step(actions)


def test_search_teacher_is_deterministic_and_returns_legal_local_indices() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=29)
    environment.reset(1, 29)
    for _ in range(6):
        observation = environment.observe()
        actions = np.asarray(
            environment.search_actions(
                node_budget=256,
                beam_width=12,
                branch_width=20,
                maximum_actions_per_turn=12,
            ),
            dtype=np.uint64,
        )
        counts = np.diff(observation["action_offsets"])
        assert np.all(actions < counts)
        assert actions[0] == actions[1]
        environment.step(actions)


def test_search_teacher_rejects_invalid_configuration() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=31)
    with pytest.raises(ValueError, match="node budget"):
        environment.search_actions(node_budget=1)


def test_search_teacher_skips_inactive_environments() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=37)
    actions = np.asarray(
        environment.search_actions(
            node_budget=256,
            beam_width=12,
            branch_width=20,
            maximum_actions_per_turn=12,
            active_mask=np.array([0, 1], dtype=np.uint8),
        ),
        dtype=np.uint64,
    )
    assert actions[0] == 0
    counts = np.diff(environment.observe()["action_offsets"])
    assert actions[1] < counts[1]


def test_search_teacher_rejects_wrong_active_mask_length() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=41)
    with pytest.raises(ValueError, match="active mask has length 1, expected 2"):
        environment.search_actions(active_mask=np.array([1], dtype=np.uint8))


@pytest.mark.parametrize(
    ("profile", "expected_profile"),
    [
        ("classic_generic_2022", "ClassicGeneric"),
        ("classic_slay_2022", "ClassicSlay"),
        ("online_default_v1", "OnlineDefaultV1"),
        ("online_classic_v1", "OnlineClassicV1"),
        ("online_duel_v1", "OnlineDuelV1"),
        ("online_experimental_v1", "OnlineExperimentalV1"),
        ("online_experimental_v2_260801", "OnlineExperimentalV2_260801"),
    ],
)
def test_all_versioned_profiles_are_available(profile: str, expected_profile: str) -> None:
    environment = VectorEnv(1, width=7, height=5, profile=profile)
    assert json.loads(environment.rules_json())["profile"] == expected_profile


def test_mixed_batch_preserves_per_environment_rules() -> None:
    profiles = [
        "classic_generic_2022",
        "online_duel_v1",
        "online_experimental_v2_260801",
    ]
    environment = VectorEnv.mixed(profiles, width=7, height=5, seed=131)
    serialized = environment.rules_jsons()
    assert environment.environments == 3
    assert [json.loads(rules)["profile"] for rules in serialized] == [
        "ClassicGeneric",
        "OnlineDuelV1",
        "OnlineExperimentalV2_260801",
    ]
    environment.step(np.zeros(3, dtype=np.uint64))
    environment.reset(1, 999)
    assert json.loads(environment.rules_jsons()[1])["profile"] == "OnlineDuelV1"


def test_fog_mode_projects_active_player_visibility() -> None:
    full = VectorEnv(1, width=11, height=9, seed=211).observe()
    fog = VectorEnv(1, width=11, height=9, seed=211, fog=True).observe()
    assert np.all(full["visible"] == full["playable"])
    assert np.any(np.logical_and(fog["playable"] == 1, fog["visible"] == 0))


def test_diplomacy_configuration_reaches_state_and_action_masks() -> None:
    environment = VectorEnv(
        1,
        width=7,
        height=5,
        seed=223,
        diplomacy=True,
        initial_relation="neutral",
    )
    observation = environment.observe()
    rules = json.loads(environment.rules_json())
    assert rules["diplomacy"]["enabled"] is True
    assert rules["diplomacy"]["initial_relation"] == "Neutral"
    assert observation["relations"].tolist() == [3, 1, 1, 3]
    assert 5 in observation["action_kinds"]


def test_procedural_batch_regenerates_topology_from_seed() -> None:
    config = ProceduralConfig(
        width=17,
        height=13,
        players=4,
        seed=800,
        land_density_per_million=600_000,
        starting_province_size=7,
        starting_money=23,
    )
    serialized_config = json.loads(config.to_json())
    assert serialized_config["schema_version"] == GENERATOR_SCHEMA_VERSION

    environment = VectorEnv.procedural(2, config, profile="online_default_v1")
    observation = environment.observe()
    first_playable = observation["playable"][:221].copy()
    second_playable = observation["playable"][221:].copy()
    assert first_playable.sum() == 133
    assert second_playable.sum() == 133
    assert not np.array_equal(first_playable, second_playable)
    assert [
        json.loads(serialized)["seed"]
        for serialized in environment.generator_jsons()
        if serialized is not None
    ] == [800, 801]

    environment.reset(1, 800)
    reset = environment.observe()
    assert np.array_equal(reset["playable"][:221], reset["playable"][221:])
    assert np.array_equal(reset["owners"][:221], reset["owners"][221:])


def test_procedural_domains_preserve_each_worker_density() -> None:
    profiles = ["classic_generic_2022", "online_default_v1"]
    configs = [
        ProceduralConfig(
            width=17,
            height=13,
            players=4,
            seed=900,
            land_density_per_million=650_000,
        ),
        ProceduralConfig(
            width=17,
            height=13,
            players=4,
            seed=901,
            land_density_per_million=700_000,
        ),
    ]
    environment = VectorEnv.procedural_domains(profiles, configs)

    environment.reset(1, 999)
    generators = [json.loads(value) for value in environment.generator_jsons()]

    assert [value["land_density_per_million"] for value in generators] == [
        650_000,
        700_000,
    ]
    assert [value["seed"] for value in generators] == [900, 999]


def test_scenario_objective_controls_episode_termination() -> None:
    objective = ScenarioObjective.survive_through_round(0, 1)
    serialized = json.loads(objective.to_json())
    assert serialized["schema_version"] == OBJECTIVE_SCHEMA_VERSION
    environment = VectorEnv(1, width=7, height=5, objective=objective)
    assert json.loads(environment.objective_jsons()[0]) == serialized

    first = environment.step(np.zeros(1, dtype=np.uint64))
    assert first["terminal"].tolist() == [0]
    assert first["objective_satisfied"].tolist() == [0]
    second = environment.step(np.zeros(1, dtype=np.uint64))
    assert second["terminal"].tolist() == [1]
    assert second["objective_satisfied"].tolist() == [1]
    assert second["winners"].tolist() == [0]


def test_invalid_scenario_objective_is_rejected_at_construction() -> None:
    objective = ScenarioObjective.destroy_player(0, 2)
    with pytest.raises(RuntimeError, match="outside a game with 2 players"):
        VectorEnv(1, width=7, height=5, objective=objective)
