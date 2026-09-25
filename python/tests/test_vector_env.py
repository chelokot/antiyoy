import json

import numpy as np
import pytest

from antiyoy_rl import (
    GENERATOR_SCHEMA_VERSION,
    OBJECTIVE_SCHEMA_VERSION,
    OBSERVATION_VERSION,
    SCORE_COMPONENT_WEIGHTS,
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


def test_position_scores_use_requested_root_player_and_reject_fog() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=47)
    first = environment.position_scores(0)
    second = environment.position_scores(1)

    assert first.shape == (2,)
    np.testing.assert_array_equal(first, -second)
    components = np.asarray(environment.position_components(0), dtype=np.int64)
    assert components.shape == (2, 16)
    np.testing.assert_array_equal(
        components @ np.asarray(SCORE_COMPONENT_WEIGHTS, dtype=np.int64), first
    )
    np.testing.assert_array_equal(
        components, -np.asarray(environment.position_components(1), dtype=np.int64)
    )
    with pytest.raises(ValueError, match="outside the game"):
        environment.position_scores(2)
    with pytest.raises(ValueError, match="outside the game"):
        environment.position_components(2)
    with pytest.raises(ValueError, match="unavailable in fog"):
        VectorEnv(1, width=7, height=5, fog=True).position_scores(0)
    with pytest.raises(ValueError, match="unavailable in fog"):
        VectorEnv(1, width=7, height=5, fog=True).position_components(0)


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


def test_fork_branches_late_state_and_keeps_source_unchanged() -> None:
    config = ProceduralConfig(width=17, height=13, players=4, seed=481)
    source = VectorEnv.procedural(1, config, action_limit=30, fog=True)
    for _ in range(8):
        source.step(np.array([0], dtype=np.uint64))
    before = source.observe()
    branches = source.fork(np.array([0, 0], dtype=np.uint64))
    fork_observation = branches.observe()
    assert branches.environments == 2
    assert branches.rules_jsons() == [source.rules_json()] * 2
    assert branches.generator_jsons() == source.generator_jsons() * 2
    for name in ("owners", "objects", "visible", "active_players", "rounds"):
        np.testing.assert_array_equal(
            fork_observation[name][: len(before[name])], before[name]
        )
    action_count = int(fork_observation["action_offsets"][1])
    branches.step(np.array([0, action_count - 1], dtype=np.uint64))
    after = source.observe()
    for name in ("owners", "objects", "active_players", "rounds"):
        np.testing.assert_array_equal(after[name], before[name])
    branches.reset(1, 482)
    assert json.loads(branches.generator_jsons()[1])["seed"] == 482
    assert json.loads(source.generator_jsons()[0])["seed"] == 481


def test_fork_rejects_empty_and_invalid_indices() -> None:
    source = VectorEnv(1, width=7, height=5)
    with pytest.raises(RuntimeError, match="at least one scenario"):
        source.fork(np.array([], dtype=np.uint64))
    with pytest.raises(RuntimeError, match="outside a batch"):
        source.fork(np.array([1], dtype=np.uint64))


def test_fork_can_extend_simulation_action_limit_without_changing_source() -> None:
    source = VectorEnv(1, width=7, height=5, seed=47, action_limit=1)
    indices = np.array([0], dtype=np.uint64)
    inherited = source.fork(indices)
    extended = source.fork(indices, action_limit=2**32 - 1)

    assert inherited.step(indices)["truncated"].tolist() == [1]
    assert extended.step(indices)["truncated"].tolist() == [0]
    assert source.done() == [False]
    with pytest.raises(RuntimeError, match="greater than zero"):
        source.fork(indices, action_limit=0)


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


def test_search_teacher_can_replan_each_observed_state() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=29)
    first = np.asarray(environment.search_actions(node_budget=256), dtype=np.uint64)
    observation = environment.observe()
    selected_kind = observation["action_kinds"][int(first[0])]
    assert selected_kind != 0
    assert environment.search_counts().tolist() == [1]

    environment.step(first)
    environment.search_actions(node_budget=256)
    assert environment.search_counts().tolist() == [1]

    replanned = np.asarray(
        environment.search_actions_replanned(node_budget=256), dtype=np.uint64
    )
    assert environment.search_counts().tolist() == [2]

    fresh = VectorEnv(1, width=7, height=5, seed=29)
    fresh.step(first)
    fresh_action = np.asarray(fresh.search_actions(node_budget=256), dtype=np.uint64)
    np.testing.assert_array_equal(replanned, fresh_action)


def test_search_turn_plans_replay_to_exact_static_scores() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=29)
    plans, scores = environment.search_turn_plans(node_budget=64, slate_size=4)

    assert len(plans) == len(scores) == 1
    assert 1 <= len(plans[0]) <= 4
    assert environment.search_counts().tolist() == []
    root = int(environment.observe()["active_players"][0])
    first = int(environment.search_actions(node_budget=64)[0])
    assert plans[0][0][0] == first

    for plan, score in zip(plans[0], scores[0]):
        branch = environment.fork(np.asarray([0], dtype=np.uint64))
        for action in plan:
            offsets = branch.observe()["action_offsets"]
            assert 0 <= action < int(offsets[1] - offsets[0])
            branch.step(np.asarray([action], dtype=np.uint64))
        assert branch.done()[0] or int(branch.observe()["active_players"][0]) != root
        assert int(branch.position_scores(root)[0]) == score


def test_search_turn_plan_traces_match_independent_replay() -> None:
    from python.audit_duel_turn_trace_process_feasibility import candidate_trace

    environment = VectorEnv(1, width=7, height=5, seed=29)
    plans, scores, traces = environment.search_turn_plan_traces(
        node_budget=64, slate_size=4
    )
    indexed, static = environment.search_turn_plans(node_budget=64, slate_size=4)

    assert plans == indexed
    assert scores == static
    assert len(traces[0]) == len(plans[0])
    root = int(environment.observe()["active_players"][0])
    for plan, score, trace in zip(plans[0], scores[0], traces[0], strict=True):
        np.testing.assert_allclose(
            np.asarray(trace, dtype=np.float32),
            candidate_trace(environment, plan, score, root),
            rtol=0,
            atol=1e-6,
        )


def test_search_turn_process_components_match_independent_replay() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=29)
    plans, scores, traces, roots, posts = environment.search_turn_plan_process(
        node_budget=64, slate_size=4
    )
    assert (plans, scores, traces) == environment.search_turn_plan_traces(
        node_budget=64, slate_size=4
    )
    root = int(environment.observe()["active_players"][0])
    assert roots[0] == environment.position_components(root)[0]
    assert len(posts[0]) == len(plans[0])
    weights = np.asarray(SCORE_COMPONENT_WEIGHTS, dtype=np.int64)
    for plan, components in zip(plans[0], posts[0], strict=True):
        branch = environment.fork(np.asarray([0], dtype=np.uint64))
        for action in plan:
            branch.step(np.asarray([action], dtype=np.uint64))
        assert components == branch.position_components(root)[0]
        if not branch.done()[0]:
            assert int(np.asarray(components, dtype=np.int64) @ weights) == int(
                branch.position_scores(root)[0]
            )


def test_search_turn_plans_reject_invalid_configuration_and_fog() -> None:
    environment = VectorEnv(1, width=7, height=5)
    with pytest.raises(ValueError, match="node budget"):
        environment.search_turn_plans(node_budget=1)
    with pytest.raises(ValueError, match="slate size"):
        environment.search_turn_plans(slate_size=0)
    with pytest.raises(ValueError, match="unavailable in fog"):
        VectorEnv(1, width=7, height=5, fog=True).search_turn_plans()
    with pytest.raises(ValueError, match="unavailable in fog"):
        VectorEnv(1, width=7, height=5, fog=True).search_turn_plan_traces()
    with pytest.raises(ValueError, match="unavailable in fog"):
        VectorEnv(1, width=7, height=5, fog=True).search_turn_plan_process()


def test_search_turn_plans_skip_inactive_environments() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=29)
    plans, scores = environment.search_turn_plans(
        node_budget=64,
        slate_size=4,
        active_mask=np.asarray([0, 1], dtype=np.uint8),
    )

    assert plans[0] == scores[0] == []
    assert len(plans[1]) == len(scores[1]) > 0
    with pytest.raises(ValueError, match="active mask has length 1, expected 2"):
        environment.search_turn_plans(active_mask=np.asarray([1], dtype=np.uint8))


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


def test_reply_search_returns_deterministic_legal_actions_for_selected_games() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=43)
    environment.reset(1, 43)
    observation = environment.observe()
    first = np.asarray(
        environment.reply_search_actions(
            node_budget=64,
            reply_nodes=16,
            slate_size=4,
            beam_width=12,
            branch_width=20,
            maximum_actions_per_turn=12,
        ),
        dtype=np.uint64,
    )
    assert first[0] == first[1]
    assert np.all(first < np.diff(observation["action_offsets"]))
    assert environment.search_counts().tolist() == [1, 1]

    selected = environment.reply_search_actions(
        node_budget=64,
        reply_nodes=16,
        slate_size=4,
        active_mask=np.array([0, 1], dtype=np.uint8),
    )
    assert selected[0] == 0
    assert environment.search_counts().tolist() == [0, 1]


def test_reply_search_rejects_invalid_response_budget() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=45)
    with pytest.raises(ValueError, match="at least two reply nodes"):
        environment.reply_search_actions(reply_nodes=1)
    with pytest.raises(ValueError, match="positive slate size"):
        environment.reply_search_actions(slate_size=0)


def test_three_turn_reply_search_selects_deterministic_legal_action() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=46)
    duplicate = VectorEnv(1, width=7, height=5, seed=46)
    observation = environment.observe()
    first = environment.reply_search_actions(
        node_budget=64,
        reply_nodes=16,
        followup_nodes=8,
        slate_size=4,
        beam_width=12,
        branch_width=20,
        maximum_actions_per_turn=12,
    )
    assert 0 <= first[0] < np.diff(observation["action_offsets"])[0]
    assert (
        duplicate.reply_search_actions(
            node_budget=64,
            reply_nodes=16,
            followup_nodes=8,
            slate_size=4,
            beam_width=12,
            branch_width=20,
            maximum_actions_per_turn=12,
        )[0]
        == first[0]
    )
    assert environment.search_counts().tolist() == [1]
    assert duplicate.search_counts().tolist() == [1]


def test_three_turn_reply_search_can_replan_from_current_state() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=46)
    configuration = {
        "node_budget": 64,
        "reply_nodes": 16,
        "followup_nodes": 8,
        "slate_size": 4,
        "beam_width": 12,
        "branch_width": 20,
        "maximum_actions_per_turn": 12,
    }
    first = environment.reply_search_actions(**configuration)
    environment.step(first)
    environment.reply_search_actions(**configuration)
    assert environment.search_counts().tolist() == [1]

    fresh = environment.fork(np.array([0], dtype=np.uint64))
    replanned = environment.reply_search_actions(
        **configuration, replan_each_action=True
    )
    expected = fresh.reply_search_actions(**configuration)

    np.testing.assert_array_equal(replanned, expected)
    assert environment.search_counts().tolist() == [2]
    for _ in range(12):
        if environment.done()[0]:
            break
        environment.step(replanned)
        if environment.done()[0]:
            break
        fresh = environment.fork(np.array([0], dtype=np.uint64))
        replanned = environment.reply_search_actions(
            **configuration, replan_each_action=True
        )
        expected = fresh.reply_search_actions(**configuration)
        np.testing.assert_array_equal(replanned, expected)


def test_opt_in_exact_score_cache_preserves_replanned_game() -> None:
    plain = VectorEnv(1, width=7, height=5, seed=46)
    cached = VectorEnv(1, width=7, height=5, seed=46)
    configuration = {
        "node_budget": 64,
        "reply_nodes": 16,
        "followup_nodes": 8,
        "slate_size": 4,
        "beam_width": 12,
        "branch_width": 20,
        "maximum_actions_per_turn": 12,
        "replan_each_action": True,
    }
    for _ in range(32):
        if plain.done()[0]:
            break
        first = plain.reply_search_actions(**configuration)
        second = cached.reply_search_actions(**configuration, score_cache=True)
        np.testing.assert_array_equal(first, second)
        for key, values in plain.observe().items():
            np.testing.assert_array_equal(values, cached.observe()[key])
        plain.step(first)
        cached.step(second)
    assert plain.search_counts().tolist() == cached.search_counts().tolist()
    assert plain.search_cache_hits().tolist() == [0]
    assert cached.search_cache_hits()[0] > 0


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
def test_all_versioned_profiles_are_available(
    profile: str, expected_profile: str
) -> None:
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
