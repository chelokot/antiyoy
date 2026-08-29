import json

import numpy as np
import pytest

from antiyoy_rl import OBSERVATION_VERSION, VectorEnv


def test_observation_and_step_contract() -> None:
    environment = VectorEnv(4, width=7, height=5, seed=47)
    observation = environment.observe()
    assert observation["version"] == OBSERVATION_VERSION
    assert observation["cell_offsets"].tolist() == [0, 35, 70, 105, 140]
    assert observation["owners"].shape == (140,)
    assert observation["visible"].shape == (140,)
    assert set(np.unique(observation["visible"])).issubset({0, 1})
    actions = np.zeros(4, dtype=np.uint64)
    result = environment.step(actions)
    assert result["actors"].tolist() == [0, 0, 0, 0]
    assert result["terminal"].tolist() == [0, 0, 0, 0]
    assert json.loads(environment.rules_json())["schema_version"] == 4


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
