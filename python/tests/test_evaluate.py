import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

import python.evaluate as evaluate_module
from antiyoy_rl.model import UniversalPolicy
from antiyoy_rl.routed import RoutedPolicy
from antiyoy_rl.vector_value import (
    VECTOR_VALUE_ARTIFACT_KIND,
    VECTOR_VALUE_ARTIFACT_VERSION,
    RelativeValueHead,
    initialize_from_scalar_value_head,
)
from python.build_bundle import build_bundle, digest
from python.evaluate import (
    FIXED_SEAT_SCHEME,
    baseline_adjusted_elo_delta,
    evaluate,
    evaluation_schedule,
    named_action_counts,
    outcome_summary,
    paired_comparison_summary,
    paired_map_bootstrap_interval,
    paired_map_comparison,
    paired_method_comparison,
    paired_elo,
    paired_seeds,
    reference_adjusted_outcome,
    relative_skill_delta,
    seat_rotation_seeds,
    selected_action_kinds,
)
from python.tests.test_bundle import write_checkpoint
from python.evaluate_suite import (
    aggregate_outcomes,
    aggregate_results,
    minimum_profile_seat_slice,
    minimum_seat_slice,
)


def test_selected_action_kinds_resolves_local_ragged_indices() -> None:
    observation = {
        "action_offsets": np.array([0, 2, 5], dtype=np.uint64),
        "action_kinds": np.array([0, 1, 0, 2, 3], dtype=np.uint8),
    }

    kinds = selected_action_kinds(
        observation,
        np.array([1, 2], dtype=np.uint64),
    )

    assert kinds.tolist() == [1, 3]


def test_named_action_counts_preserves_zero_categories() -> None:
    counts = named_action_counts(np.array([7, 5, 3, 2, 1, 0], dtype=np.int64))

    assert counts == {
        "end_turn": 7,
        "move": 5,
        "recruit": 3,
        "build": 2,
        "plant_tree": 1,
        "diplomacy": 0,
    }


def test_policy_self_match_is_an_exact_zero_delta(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)

    result = evaluate(
        checkpoint,
        games=2,
        seed=91_000,
        device_name="cpu",
        baseline="policy",
        profile="classic_generic_2022",
        search_nodes=8,
        search_beam_width=4,
        search_branch_width=4,
        search_maximum_actions_per_turn=4,
        width=7,
        height=5,
        action_limit=12,
        model_agent="policy",
    )

    assert result["score_delta"] == pytest.approx(0.0)
    assert result["baseline_adjusted_elo_delta"] == pytest.approx(0.0)
    assert result["elo_delta"] == pytest.approx(0.0)
    assert result["game_seeds"] == [91_000, 91_000]
    assert result["model_seats"] == [0, 1]
    assert result["winners"] == result["baseline_self_play"]["winners"] * 2
    assert len(result["game_truncated"]) == 2
    assert sum(result["game_truncated"]) == result["truncations"]
    assert (
        sum(result["baseline_self_play"]["game_truncated"])
        == result["baseline_self_play"]["truncations"]
    )
    assert result["policy_search"]["decisions"] == 0
    assert result["paired_map_comparison"]["same"] == 1
    assert result["paired_map_bootstrap_95"]["score_delta"] == [0.0, 0.0]
    assert result["model_baseline_policy_actions"]["disagreements"] == 0
    assert [seat["paired_method_comparison"] for seat in result["seats"]] == [
        {
            "candidate_better": 0,
            "baseline_better": 0,
            "same": 1,
            "discordant": 0,
            "net_improvements": 0,
            "exact_two_sided_sign_test_p": 1.0,
        },
        {
            "candidate_better": 0,
            "baseline_better": 0,
            "same": 1,
            "discordant": 0,
            "net_improvements": 0,
            "exact_two_sided_sign_test_p": 1.0,
        },
    ]


def test_evaluation_identifies_each_action_limit_adjudication(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)

    result = evaluate(
        checkpoint,
        games=2,
        seed=91_003,
        device_name="cpu",
        baseline="policy",
        profile="classic_generic_2022",
        search_nodes=8,
        search_beam_width=4,
        search_branch_width=4,
        search_maximum_actions_per_turn=4,
        width=7,
        height=5,
        action_limit=1,
        model_agent="policy",
    )

    assert result["game_seeds"] == [91_003, 91_003]
    assert result["model_seats"] == [0, 1]
    assert result["game_truncated"] == [True, True]
    assert result["truncations"] == 2
    assert result["baseline_self_play"]["game_truncated"] == [True]


@pytest.mark.parametrize("followup_nodes", [0, 8])
def test_reply_search_baseline_runs_rotated_procedural_duel(
    tmp_path: Path, followup_nodes: int
) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)

    result = evaluate(
        checkpoint,
        games=2,
        seed=91_004,
        device_name="cpu",
        baseline="reply_search",
        profile="classic_generic_2022",
        search_nodes=32,
        search_beam_width=12,
        search_branch_width=20,
        search_maximum_actions_per_turn=12,
        reply_search_nodes=8,
        reply_slate_size=4,
        followup_search_nodes=followup_nodes,
        width=7,
        height=5,
        action_limit=40,
        procedural=True,
        generator_schema_version=2,
        players=2,
        model_agent="policy",
    )

    assert result["baseline"] == "reply_search"
    assert result["games"] == 2
    assert result["game_seeds"] == [91_004, 91_004]
    assert result["model_seats"] == [0, 1]
    assert result["search_nodes"] == 32
    assert result["search_maximum_actions_per_turn"] == 12
    assert result["reply_search_nodes"] == 8
    assert result["reply_slate_size"] == 4
    assert result["followup_search_nodes"] == followup_nodes


def test_model_reply_search_completes_matched_rotated_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)
    monkeypatch.setattr(
        RoutedPolicy,
        "actions",
        lambda self, observation, rules: np.zeros(
            len(observation["widths"]), dtype=np.uint64
        ),
    )

    result = evaluate(
        checkpoint,
        games=2,
        seed=91_005,
        device_name="cpu",
        baseline="search",
        profile="classic_generic_2022",
        search_nodes=32,
        search_beam_width=12,
        search_branch_width=20,
        search_maximum_actions_per_turn=24,
        reply_slate_size=4,
        width=7,
        height=5,
        action_limit=500,
        procedural=True,
        generator_schema_version=2,
        players=2,
        model_agent="model_reply_search",
    )

    assert result["game_seeds"] == [91_005, 91_005]
    assert result["model_seats"] == [0, 1]
    assert result["truncations"] == 0
    assert result["game_truncated"] == [False, False]
    assert result["model_reply_search"]["root_turns"] > 0
    assert result["model_reply_search"]["candidate_replies"] > 0
    assert result["model_reply_search"]["slate_size"] == 4


def test_reply_search_baseline_rejects_multiplayer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)
    with pytest.raises(ValueError, match="two-player games"):
        evaluate(
            checkpoint,
            games=3,
            seed=91_005,
            device_name="cpu",
            baseline="reply_search",
            profile="classic_generic_2022",
            search_nodes=32,
            search_beam_width=12,
            search_branch_width=20,
            search_maximum_actions_per_turn=12,
            width=7,
            height=5,
            action_limit=40,
            procedural=True,
            players=3,
        )


@pytest.mark.parametrize("followup_nodes", [0, 8])
def test_reply_teacher_audit_partitions_policy_decisions(
    tmp_path: Path, followup_nodes: int
) -> None:
    checkpoint = tmp_path / "policy.pt"
    source = tmp_path / "source.pt"
    write_checkpoint(checkpoint, 1.0)
    write_checkpoint(source, 1.5)

    result = evaluate(
        checkpoint,
        games=2,
        seed=91_006,
        device_name="cpu",
        baseline="policy",
        baseline_checkpoint_path=source,
        profile="classic_generic_2022",
        search_nodes=32,
        search_beam_width=12,
        search_branch_width=20,
        search_maximum_actions_per_turn=12,
        reply_search_nodes=8,
        reply_slate_size=4,
        followup_search_nodes=followup_nodes,
        width=7,
        height=5,
        action_limit=24,
        procedural=True,
        generator_schema_version=2,
        players=2,
        audit_reply_teacher=True,
    )

    counts = result["reply_teacher_agreement"]["by_seat"]
    assert result["followup_search_nodes"] == followup_nodes
    assert len(counts) == 2
    assert (
        sum(seat["decisions"] for seat in counts)
        == result["model_baseline_policy_actions"]["decisions"]
    )
    for seat in counts:
        assert seat["decisions"] == (
            seat["source_matches_teacher"] + seat["teacher_source_disagreements"]
        )
        assert seat["teacher_source_disagreements"] == (
            seat["student_matches_teacher_on_disagreements"]
            + seat["student_matches_source_on_disagreements"]
            + seat["student_matches_neither_on_disagreements"]
        )
        assert seat["student_matches_teacher"] == (
            seat["source_matches_teacher"]
            - seat["student_deviates_when_teacher_matches_source"]
            + seat["student_matches_teacher_on_disagreements"]
        )
    for seat in result["reply_teacher_agreement"]["turn_fidelity_by_seat"]:
        assert seat["whole_turn_matches"] <= seat["first_action_matches"]
        assert seat["first_action_matches"] <= seat["completed"]
        assert seat["completed"] <= seat["started"]


def test_reply_teacher_audit_requires_a_frozen_direct_baseline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)
    with pytest.raises(ValueError, match="frozen baseline checkpoint"):
        evaluate(
            checkpoint,
            games=2,
            seed=91_007,
            device_name="cpu",
            baseline="policy",
            profile="classic_generic_2022",
            search_nodes=32,
            search_beam_width=12,
            search_branch_width=20,
            search_maximum_actions_per_turn=12,
            width=7,
            height=5,
            action_limit=24,
            audit_reply_teacher=True,
        )


def test_rotated_procedural_evaluation_has_a_distinct_domain(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)
    arguments = {
        "games": 3,
        "seed": 91_001,
        "device_name": "cpu",
        "baseline": "policy",
        "profile": "classic_generic_2022",
        "search_nodes": 8,
        "search_beam_width": 4,
        "search_branch_width": 4,
        "search_maximum_actions_per_turn": 4,
        "width": 9,
        "height": 7,
        "action_limit": 12,
        "procedural": True,
        "players": 3,
        "starting_province_size": 3,
    }
    legacy = evaluate(checkpoint, **arguments)
    rotated = evaluate(checkpoint, **arguments, generator_schema_version=2)
    assert legacy["generator"] == "procedural_v1"
    assert rotated["generator"] == "procedural_v2"
    assert legacy["generator_config"]["schema_version"] == 1
    assert rotated["generator_config"]["schema_version"] == 2
    assert legacy["domain_descriptor"] == rotated["domain_descriptor"]
    assert legacy["domain"] != rotated["domain"]
    assert legacy["game_seeds"] == rotated["game_seeds"]
    assert legacy["score_delta"] == pytest.approx(0.0)
    assert rotated["score_delta"] == pytest.approx(0.0)


def test_rotated_generator_requires_a_procedural_map(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)
    with pytest.raises(ValueError, match="requires a procedural map"):
        evaluate(
            checkpoint,
            games=2,
            seed=91_002,
            device_name="cpu",
            baseline="policy",
            profile="classic_generic_2022",
            search_nodes=8,
            search_beam_width=4,
            search_branch_width=4,
            search_maximum_actions_per_turn=4,
            width=7,
            height=5,
            action_limit=12,
            generator_schema_version=2,
        )


def test_rotated_map_can_explicitly_reuse_the_legacy_policy_route(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.pt"
    specialist = tmp_path / "specialist.pt"
    bundle = tmp_path / "bundle.pt"
    write_checkpoint(primary, 1.0, profiles=["classic_generic_2022"])
    write_checkpoint(specialist, 7.0, profiles=["classic_generic_2022"])
    build_bundle(
        primary,
        {},
        bundle,
        {("classic_generic_2022", "procedural_v1", 3): specialist},
    )
    arguments = {
        "games": 3,
        "seed": 91_003,
        "device_name": "cpu",
        "baseline": "policy",
        "profile": "classic_generic_2022",
        "search_nodes": 8,
        "search_beam_width": 4,
        "search_branch_width": 4,
        "search_maximum_actions_per_turn": 4,
        "width": 9,
        "height": 7,
        "action_limit": 12,
        "procedural": True,
        "players": 3,
        "starting_province_size": 3,
        "generator_schema_version": 2,
    }
    native_route = evaluate(bundle, **arguments)
    reused_route = evaluate(bundle, **arguments, route_generator="procedural_v1")
    assert native_route["generator"] == reused_route["generator"] == "procedural_v2"
    assert native_route["domain"] == reused_route["domain"]
    assert native_route["route_generator"] == "procedural_v2"
    assert reused_route["route_generator"] == "procedural_v1"
    assert native_route["route_domain"] != reused_route["route_domain"]
    assert native_route["selected_experts"] == ["primary"] * 3
    assert all(
        expert.startswith("context:") for expert in reused_route["selected_experts"]
    )
    assert native_route["score_delta"] == pytest.approx(0.0)
    assert reused_route["score_delta"] == pytest.approx(0.0)


def test_policy_pair_uses_an_independent_baseline_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)

    result = evaluate(
        checkpoint,
        games=2,
        seed=91_050,
        device_name="cpu",
        baseline="policy",
        profile="classic_generic_2022",
        search_nodes=8,
        search_beam_width=4,
        search_branch_width=4,
        search_maximum_actions_per_turn=4,
        width=7,
        height=5,
        action_limit=12,
        baseline_checkpoint_path=checkpoint,
    )

    assert result["baseline_checkpoint"] == str(checkpoint)
    assert result["baseline_selected_experts"] == ["single", "single"]
    assert result["score_delta"] == pytest.approx(0.0)
    assert result["baseline_adjusted_elo_delta"] == pytest.approx(0.0)
    assert result["elo_delta"] == pytest.approx(0.0)


def test_baseline_checkpoint_requires_policy_baseline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)

    with pytest.raises(ValueError, match="requires the policy baseline"):
        evaluate(
            checkpoint,
            games=2,
            seed=91_075,
            device_name="cpu",
            baseline="greedy",
            profile="classic_generic_2022",
            search_nodes=8,
            search_beam_width=4,
            search_branch_width=4,
            search_maximum_actions_per_turn=4,
            width=7,
            height=5,
            action_limit=12,
            baseline_checkpoint_path=checkpoint,
        )


def test_policy_search_runs_the_native_tree_for_every_model_decision(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)

    result = evaluate(
        checkpoint,
        games=2,
        seed=91_100,
        device_name="cpu",
        baseline="policy",
        profile="classic_generic_2022",
        search_nodes=8,
        search_beam_width=4,
        search_branch_width=4,
        search_maximum_actions_per_turn=4,
        width=7,
        height=5,
        action_limit=12,
        model_agent="puct",
        puct_nodes=4,
        puct_root_value_weight=0.0,
        puct_leaf_batch_size=8,
    )

    search = result["policy_search"]
    assert search["decisions"] > 0
    assert search["evaluated_leaves"] > 0
    assert search["leaf_batches"] > 0
    assert search["total_nodes"] == search["decisions"] * search["node_budget"]
    assert search["root_visited_actions"] <= search["root_legal_actions"]
    assert search["roots_with_multiple_visited_actions"] <= search["decisions"]
    assert search["selected_unvisited_actions"] <= search["decisions"]
    assert search["total_root_visits"] > 0
    assert search["root_value_weight"] == 0.0
    assert search["value_perspective"] == "active"
    assert search["opponent_horizon"] == "search"
    assert search["objective"] == "scalar"
    assert result["score_delta"] == pytest.approx(0.0)


def test_single_disagreement_search_intervenes_at_most_once_per_game(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)

    result = evaluate(
        checkpoint,
        games=4,
        seed=91_110,
        device_name="cpu",
        baseline="policy",
        profile="classic_generic_2022",
        search_nodes=8,
        search_beam_width=4,
        search_branch_width=4,
        search_maximum_actions_per_turn=4,
        width=7,
        height=5,
        action_limit=24,
        model_agent="puct",
        puct_nodes=4,
        puct_root_value_weight=0.0,
        puct_leaf_batch_size=8,
        single_disagreement=True,
    )

    search = result["policy_search"]
    disagreements = result["model_baseline_policy_actions"]["disagreements"]
    assert search["single_disagreement_intervention"]
    assert search["intervened_games"] == disagreements
    assert sum(search["intervened_games_by_seat"]) == disagreements
    assert disagreements <= result["games"]
    assert search["decisions"] <= result["model_baseline_policy_actions"]["decisions"]
    assert search["root_visited_actions"] <= search["root_legal_actions"]
    if disagreements == 0:
        assert result["score_delta"] == pytest.approx(0.0)


def test_single_disagreement_stops_search_after_the_first_changed_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)
    native_search = evaluate_module.policy_search_actions

    def force_changed_action(environment, policy, rules, selected, config, **kwargs):
        actions, metrics = native_search(
            environment, policy, rules, selected, config, **kwargs
        )
        observation = environment.observe()
        direct = policy.actions(observation, rules)
        offsets = observation["action_offsets"]
        for index in np.flatnonzero(selected):
            legal_count = int(offsets[index + 1] - offsets[index])
            if legal_count > 1:
                actions[index] = (int(direct[index]) + 1) % legal_count
        return actions, metrics

    monkeypatch.setattr(evaluate_module, "policy_search_actions", force_changed_action)
    result = evaluate(
        checkpoint,
        games=2,
        seed=91_112,
        device_name="cpu",
        baseline="policy",
        profile="classic_generic_2022",
        search_nodes=8,
        search_beam_width=4,
        search_branch_width=4,
        search_maximum_actions_per_turn=4,
        width=7,
        height=5,
        action_limit=24,
        model_agent="puct",
        puct_nodes=4,
        puct_root_value_weight=0.0,
        puct_leaf_batch_size=8,
        single_disagreement=True,
    )

    assert result["policy_search"]["intervened_games"] == 2
    assert result["policy_search"]["intervened_games_by_seat"] == [1, 1]
    assert result["model_baseline_policy_actions"]["disagreements"] == 2


def test_single_disagreement_requires_an_identical_direct_reference(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)
    with pytest.raises(ValueError, match="requires PUCT against its own direct policy"):
        evaluate(
            checkpoint,
            games=2,
            seed=91_111,
            device_name="cpu",
            baseline="policy",
            profile="classic_generic_2022",
            search_nodes=8,
            search_beam_width=4,
            search_branch_width=4,
            search_maximum_actions_per_turn=4,
            width=7,
            height=5,
            action_limit=12,
            model_agent="policy",
            single_disagreement=True,
        )


def test_heuristic_puct_evaluation_records_its_value_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    write_checkpoint(checkpoint, 1.0)
    result = evaluate(
        checkpoint,
        games=2,
        seed=91_113,
        device_name="cpu",
        baseline="policy",
        profile="classic_generic_2022",
        search_nodes=8,
        search_beam_width=4,
        search_branch_width=4,
        search_maximum_actions_per_turn=4,
        width=7,
        height=5,
        action_limit=12,
        model_agent="puct",
        puct_nodes=4,
        puct_leaf_batch_size=8,
        puct_value_source="heuristic",
    )

    assert result["policy_search"]["value_source"] == "heuristic"
    assert result["policy_search"]["heuristic_scale"] == 2048.0
    assert result["policy_search"]["decisions"] > 0


def test_maxn_policy_search_loads_a_matching_one_pass_value_head(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    value_head_path = tmp_path / "vector-value.pt"
    write_checkpoint(checkpoint, 1.0)
    policy = UniversalPolicy(hidden=16, layers=1)
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(source["model"])
    value_head = RelativeValueHead(hidden=16)
    initialize_from_scalar_value_head(value_head, policy.value_head)
    torch.save(
        {
            "kind": VECTOR_VALUE_ARTIFACT_KIND,
            "artifact_version": VECTOR_VALUE_ARTIFACT_VERSION,
            "source": {"sha256": digest(checkpoint)},
            "architecture": {"hidden": 16},
            "model": value_head.state_dict(),
        },
        value_head_path,
    )

    result = evaluate(
        checkpoint,
        games=3,
        seed=91_125,
        device_name="cpu",
        baseline="policy",
        profile="classic_generic_2022",
        search_nodes=4,
        search_beam_width=2,
        search_branch_width=2,
        search_maximum_actions_per_turn=2,
        width=9,
        height=7,
        action_limit=10,
        procedural=True,
        players=3,
        starting_province_size=3,
        model_agent="puct",
        puct_nodes=2,
        puct_leaf_batch_size=8,
        puct_objective="maxn",
        maxn_value_head_path=value_head_path,
    )

    assert result["policy_search"]["objective"] == "maxn"
    assert result["policy_search"]["vector_value_head"] == str(value_head_path)
    assert result["policy_search"]["evaluated_leaves"] > 0


def test_paired_seeds_repeat_each_map_for_opposite_seats() -> None:
    assert paired_seeds(6, 100).tolist() == [100, 100, 101, 101, 102, 102]


def test_seat_rotation_seeds_repeat_each_map_for_every_player() -> None:
    assert seat_rotation_seeds(8, 100, 4).tolist() == [
        100,
        100,
        100,
        100,
        101,
        101,
        101,
        101,
    ]


def test_fixed_seat_schedule_uses_unique_seeds() -> None:
    seeds, seats = evaluation_schedule(4, 100, 6, 2)

    assert FIXED_SEAT_SCHEME == "unique_seed_fixed_seat_v1"
    assert seeds.tolist() == [100, 101, 102, 103]
    assert seats.tolist() == [2, 2, 2, 2]


@pytest.mark.parametrize(
    ("games", "players", "seat"), [(0, 6, 2), (4, 6, -1), (4, 6, 6)]
)
def test_fixed_seat_schedule_rejects_invalid_requests(
    games: int, players: int, seat: int
) -> None:
    with pytest.raises(ValueError):
        evaluation_schedule(games, 100, players, seat)


@pytest.mark.parametrize(("games", "players"), [(3, 4), (5, 4), (8, 1)])
def test_seat_rotation_seeds_reject_incomplete_rotations(
    games: int, players: int
) -> None:
    with pytest.raises(ValueError):
        seat_rotation_seeds(games, 100, players)


def test_multiplayer_skill_delta_is_zero_at_equal_opponent_win_rate() -> None:
    assert relative_skill_delta(0.25, 40, 4) == pytest.approx(0.0)


@pytest.mark.parametrize("games", [0, 1, 3])
def test_paired_seeds_reject_unpaired_game_counts(games: int) -> None:
    with pytest.raises(ValueError, match="positive even"):
        paired_seeds(games, 100)


def test_outcome_summary_reports_complete_seat_slice() -> None:
    assert outcome_summary(4, 2, 1, 1, 0, 1) == {
        "games": 4,
        "wins": 2,
        "draws": 1,
        "terminal_draws": 0,
        "truncations": 1,
        "adjudications": 0,
        "losses": 1,
        "score": 0.625,
        "elo_delta": paired_elo(0.625, 4),
    }


def test_reference_adjustment_calibrates_asymmetric_seat() -> None:
    outcome = outcome_summary(6, 1, 0, 5, 0, 0, players=3)

    adjusted = reference_adjusted_outcome(outcome, 6, 1, 0, 0)

    assert adjusted["baseline_score"] == pytest.approx(1 / 6)
    assert adjusted["score_delta"] == pytest.approx(0.0)
    assert adjusted["baseline_adjusted_elo_delta"] == pytest.approx(0.0)


def test_baseline_adjusted_elo_reports_method_uplift_at_a_fixed_seat() -> None:
    adjusted = reference_adjusted_outcome(
        outcome_summary(128, 17, 0, 111, 0, 0, players=5),
        128,
        15,
        0,
        0,
    )

    assert adjusted["elo_delta"] < 0
    assert adjusted["baseline_adjusted_elo_delta"] > 0
    assert adjusted["baseline_adjusted_elo_delta"] == pytest.approx(
        baseline_adjusted_elo_delta(17 / 128, 15 / 128, 128)
    )


def test_paired_method_comparison_counts_discordant_maps() -> None:
    comparison = paired_method_comparison(
        np.array([1.0, 1.0, 0.0, 0.5, 0.0]),
        np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
    )

    assert comparison == {
        "candidate_better": 2,
        "baseline_better": 1,
        "same": 2,
        "discordant": 3,
        "net_improvements": 1,
        "exact_two_sided_sign_test_p": 1.0,
    }


def test_paired_map_comparison_groups_both_seats_before_sign_test() -> None:
    candidate = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    baseline = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    assert paired_method_comparison(candidate, baseline)["candidate_better"] == 3
    assert paired_map_comparison(candidate, baseline, 2, None) == {
        "candidate_better": 1,
        "baseline_better": 1,
        "same": 1,
        "discordant": 2,
        "net_improvements": 0,
        "exact_two_sided_sign_test_p": 1.0,
    }
    assert paired_map_comparison(candidate, baseline, 2, 0) == (
        paired_method_comparison(candidate, baseline)
    )


def test_paired_map_bootstrap_preserves_seat_clusters_and_is_reproducible() -> None:
    candidate = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    baseline = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    interval = paired_map_bootstrap_interval(candidate, baseline, 2, None, 901, 128)

    assert interval == paired_map_bootstrap_interval(
        candidate, baseline, 2, None, 901, 128
    )
    assert interval["resamples"] == 128
    assert interval["score_delta"][0] < 0 < interval["score_delta"][1]
    assert interval["baseline_adjusted_elo_delta"][0] < 0
    assert interval["baseline_adjusted_elo_delta"][1] > 0


def test_suite_aggregate_counts_draws_and_truncations() -> None:
    aggregate = aggregate_results(
        [
            {
                "games": 4,
                "wins": 3,
                "draws": 1,
                "losses": 0,
                "truncations": 1,
                "terminal_draws": 0,
                "seats": [
                    {
                        "games": 2,
                        "wins": 2,
                        "draws": 0,
                        "losses": 0,
                        "truncations": 0,
                        "terminal_draws": 0,
                    },
                    {
                        "games": 2,
                        "wins": 1,
                        "draws": 1,
                        "losses": 0,
                        "truncations": 1,
                        "terminal_draws": 0,
                    },
                ],
            },
            {
                "games": 4,
                "wins": 1,
                "draws": 0,
                "losses": 3,
                "truncations": 0,
                "terminal_draws": 0,
                "seats": [
                    {
                        "games": 2,
                        "wins": 1,
                        "draws": 0,
                        "losses": 1,
                        "truncations": 0,
                        "terminal_draws": 0,
                    },
                    {
                        "games": 2,
                        "wins": 0,
                        "draws": 0,
                        "losses": 2,
                        "truncations": 0,
                        "terminal_draws": 0,
                    },
                ],
            },
        ]
    )

    assert aggregate["games"] == 8
    assert aggregate["score"] == 0.5625
    assert aggregate["truncations"] == 1
    assert aggregate["relative_elo"] == paired_elo(0.5625, 8)
    assert aggregate["seats"] == [
        {
            "seat": 0,
            "games": 4,
            "wins": 3,
            "draws": 0,
            "losses": 1,
            "truncations": 0,
            "adjudications": 0,
            "terminal_draws": 0,
            "score": 0.75,
            "relative_elo": paired_elo(0.75, 4),
        },
        {
            "seat": 1,
            "games": 4,
            "wins": 1,
            "draws": 1,
            "losses": 2,
            "truncations": 1,
            "adjudications": 0,
            "terminal_draws": 0,
            "score": 0.375,
            "relative_elo": paired_elo(0.375, 4),
        },
    ]


def test_suite_aggregate_supports_every_multiplayer_seat() -> None:
    result = {
        "players": 4,
        "games": 8,
        "wins": 2,
        "draws": 0,
        "losses": 6,
        "truncations": 0,
        "terminal_draws": 0,
        "seats": [
            {
                "games": 2,
                "wins": 1 if seat < 2 else 0,
                "draws": 0,
                "losses": 1 if seat < 2 else 2,
                "truncations": 0,
                "terminal_draws": 0,
            }
            for seat in range(4)
        ],
    }

    aggregate = aggregate_results([result])

    assert aggregate["score"] == 0.25
    assert aggregate["relative_elo"] == pytest.approx(0.0)
    assert [seat["seat"] for seat in aggregate["seats"]] == [0, 1, 2, 3]


def test_suite_aggregate_preserves_self_play_calibration() -> None:
    result = {
        "players": 3,
        "games": 6,
        "wins": 3,
        "draws": 0,
        "losses": 3,
        "truncations": 0,
        "terminal_draws": 0,
        "baseline_wins": 2,
        "baseline_draws": 0,
        "baseline_truncations": 0,
        "seats": [
            {
                "games": 2,
                "wins": wins,
                "draws": 0,
                "losses": 2 - wins,
                "truncations": 0,
                "terminal_draws": 0,
                "baseline_wins": baseline_wins,
                "baseline_draws": 0,
                "baseline_truncations": 0,
            }
            for wins, baseline_wins in ((2, 1), (1, 1), (0, 0))
        ],
    }

    aggregate = aggregate_results([result])

    assert aggregate["baseline_score"] == pytest.approx(1 / 3)
    assert aggregate["score_delta"] == pytest.approx(1 / 6)
    assert aggregate["seats"][2]["score_delta"] == pytest.approx(0.0)


def test_suite_aggregate_pools_matched_game_and_map_comparisons() -> None:
    outcomes = [
        {
            "players": 5,
            "games": 4,
            "wins": 2,
            "draws": 0,
            "losses": 2,
            "truncations": 0,
            "terminal_draws": 0,
            "paired_method_comparison": {
                "candidate_better": 2,
                "baseline_better": 1,
                "same": 1,
            },
            "paired_map_comparison": {
                "candidate_better": 1,
                "baseline_better": 0,
                "same": 1,
            },
        },
        {
            "players": 5,
            "games": 4,
            "wins": 1,
            "draws": 0,
            "losses": 3,
            "truncations": 0,
            "terminal_draws": 0,
            "paired_method_comparison": {
                "candidate_better": 1,
                "baseline_better": 2,
                "same": 1,
            },
            "paired_map_comparison": {
                "candidate_better": 0,
                "baseline_better": 1,
                "same": 1,
            },
        },
    ]

    aggregate = aggregate_outcomes(outcomes)

    assert aggregate["paired_method_comparison"] == {
        "candidate_better": 3,
        "baseline_better": 3,
        "same": 2,
        "discordant": 6,
        "net_improvements": 0,
        "exact_two_sided_sign_test_p": 1.0,
    }
    assert aggregate["paired_map_comparison"] == {
        "candidate_better": 1,
        "baseline_better": 1,
        "same": 2,
        "discordant": 2,
        "net_improvements": 0,
        "exact_two_sided_sign_test_p": 1.0,
    }


def test_suite_aggregate_rejects_partial_matched_map_evidence() -> None:
    paired = {
        "players": 2,
        "games": 2,
        "wins": 1,
        "draws": 0,
        "losses": 1,
        "truncations": 0,
        "terminal_draws": 0,
        "paired_method_comparison": {
            "candidate_better": 1,
            "baseline_better": 0,
            "same": 1,
        },
    }
    unpaired = {
        key: value for key, value in paired.items() if key != "paired_method_comparison"
    }

    with pytest.raises(ValueError, match="cannot mix paired and unpaired"):
        aggregate_outcomes([paired, unpaired])
    with pytest.raises(ValueError, match="cannot mix paired and unpaired map"):
        aggregate_outcomes(
            [
                paired,
                {**paired, "paired_map_comparison": paired["paired_method_comparison"]},
            ]
        )


def test_procedural_puct_report_combines_every_matched_map_window() -> None:
    report = json.loads(
        Path("benchmarks/2026-08-31-procedural-5p-puct-loop-rocm.json").read_text()
    )["ranking_value_ablation"]
    windows = report["puct_windows"]
    combined = report["combined"]
    comparisons = [window["paired_method_comparison"] for window in windows]

    assert combined["games"] == sum(window["games"] for window in windows)
    assert combined["paired_method_comparison"] == paired_comparison_summary(
        sum(comparison["candidate_better"] for comparison in comparisons),
        sum(comparison["baseline_better"] for comparison in comparisons),
        sum(comparison["same"] for comparison in comparisons),
    )


def test_vector_maxn_report_preserves_speed_fidelity_and_rejection_gates() -> None:
    report = json.loads(
        Path(
            "benchmarks/2026-08-31-one-pass-maxn-vector-distillation-rocm.json"
        ).read_text()
    )

    assert report["distillation"]["validation"]["mse_reduction_fraction"] > 0.87
    assert report["latency_and_fidelity_control"]["speedup"] > 4
    assert report["heldout_exact_fidelity"]["one_pass_maxn"]["same_model_score"] == 63
    assert (
        report["heldout_strength_gate"]["paired_method_comparison"]["net_improvements"]
        == -3
    )
    assert report["decision"]["agent_promotion"].startswith("rejected")


def test_outcome_vector_report_preserves_matched_teacher_comparison() -> None:
    report = json.loads(
        Path("benchmarks/2026-08-31-outcome-vector-value-rocm.json").read_text()
    )
    combined = report["combined"]

    assert combined["games"] == sum(
        window["games"] for window in report["matched_windows"]
    )
    assert combined["outcome_head"]["wins"] == 42
    assert combined["scalar_teacher_head"]["wins"] == 42
    assert combined["outcome_against_scalar"] == {
        "outcome_better": 5,
        "scalar_better": 5,
        "same": 246,
        "discordant": 10,
        "net_improvements": 0,
        "exact_two_sided_sign_test_p": 1.0,
    }
    assert report["decision"]["agent_promotion"].startswith("rejected")


def test_minimum_seat_slice_preserves_profile_and_seed() -> None:
    results = [
        {
            "profile": "classic_generic_2022",
            "seed": 100,
            "seats": [
                {"seat": 0, "score": 0.75},
                {"seat": 1, "score": 0.5},
            ],
        },
        {
            "profile": "online_duel_v1",
            "seed": 200,
            "seats": [
                {"seat": 0, "score": 0.625},
                {"seat": 1, "score": 0.125},
            ],
        },
    ]

    assert minimum_seat_slice(results) == {
        "profile": "online_duel_v1",
        "seed": 200,
        "seat": 1,
        "score": 0.125,
    }


def test_minimum_seat_slice_uses_self_play_calibrated_delta() -> None:
    results = [
        {
            "profile": "online_experimental_v2_260801",
            "seed": 100,
            "seats": [
                {"seat": 0, "score": 0.5, "score_delta": 0.0},
                {"seat": 1, "score": 0.25, "score_delta": 0.125},
            ],
        }
    ]

    assert minimum_seat_slice(results)["seat"] == 0


def test_minimum_profile_seat_slice_aggregates_seed_windows() -> None:
    results = [
        {
            "players": 2,
            "profile": "classic_generic_2022",
            "seed": seed,
            "seats": [
                {
                    "games": 2,
                    "wins": wins,
                    "draws": 0,
                    "losses": 2 - wins,
                    "truncations": 0,
                    "terminal_draws": 0,
                    "baseline_wins": baseline_wins,
                    "baseline_draws": 0,
                    "baseline_truncations": 0,
                },
                {
                    "games": 2,
                    "wins": 1,
                    "draws": 0,
                    "losses": 1,
                    "truncations": 0,
                    "terminal_draws": 0,
                    "baseline_wins": 1,
                    "baseline_draws": 0,
                    "baseline_truncations": 0,
                },
            ],
        }
        for seed, wins, baseline_wins in ((100, 0, 1), (200, 2, 1))
    ]

    weakest = minimum_profile_seat_slice(results)

    assert weakest["seat"] == 0
    assert weakest["seed_windows"] == [100, 200]
    assert weakest["score_delta"] == pytest.approx(0.0)
