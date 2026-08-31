import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl.model import UniversalPolicy
from antiyoy_rl.vector_value import (
    VECTOR_VALUE_ARTIFACT_KIND,
    VECTOR_VALUE_ARTIFACT_VERSION,
    RelativeValueHead,
    initialize_from_scalar_value_head,
)
from python.build_bundle import digest
from python.evaluate import (
    FIXED_SEAT_SCHEME,
    baseline_adjusted_elo_delta,
    evaluate,
    evaluation_schedule,
    named_action_counts,
    outcome_summary,
    paired_comparison_summary,
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
    assert result["policy_search"]["decisions"] == 0
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
    assert search["total_root_visits"] > 0
    assert search["root_value_weight"] == 0.0
    assert search["value_perspective"] == "active"
    assert search["opponent_horizon"] == "search"
    assert search["objective"] == "scalar"
    assert result["score_delta"] == pytest.approx(0.0)


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


def test_suite_aggregate_pools_matched_map_comparisons() -> None:
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
