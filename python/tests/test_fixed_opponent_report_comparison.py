import pytest

from python.compare_fixed_opponent_reports import compare


def report(checkpoint: str, winners: list[int]) -> dict[str, object]:
    return {
        "baseline": "reply_search",
        "games": 4,
        "seed": 100,
        "players": 2,
        "profile": "classic_generic_2022",
        "generator": "procedural_v2",
        "generator_config": {"schema_version": 2},
        "domain_descriptor": {"width": 11, "height": 9},
        "action_limit": 1000,
        "route_generator": "procedural_v1",
        "model_seat": 1,
        "model_seats": [1, 1, 1, 1],
        "game_seeds": [100, 101, 102, 103],
        "search_nodes": 256,
        "search_beam_width": 32,
        "search_branch_width": 48,
        "search_maximum_actions_per_turn": 24,
        "reply_search_nodes": 64,
        "reply_slate_size": 8,
        "baseline_self_play": {"winners": [0, 0, 1, 1]},
        "checkpoint": checkpoint,
        "winners": winners,
        "wins": sum(winner == 1 for winner in winners),
        "truncations": 0,
    }


def test_paired_comparison_uses_same_map_winner_ledgers() -> None:
    result = compare(
        report("student.pt", [1, 1, 0, 255]),
        report("source.pt", [0, 1, 1, 255]),
    )

    assert result["paired_independent_maps"]["candidate_better"] == 1
    assert result["paired_independent_maps"]["baseline_better"] == 1
    assert result["paired_independent_maps"]["same"] == 2
    assert result["candidate_score"] == 0.625
    assert result["reference_score"] == 0.625
    assert result["pool_relative_elo_delta"] == 0
    assert result["all_games_complete"] is True
    assert "All compared games reached a terminal state" in result["qualification"]


def test_paired_comparison_discloses_action_limit_adjudication() -> None:
    candidate = report("student.pt", [1, 1, 0, 0])
    reference = report("source.pt", [0, 1, 1, 0])
    reference["truncations"] = 1

    result = compare(candidate, reference)

    assert result["all_games_complete"] is False
    assert result["reference_truncations"] == 1
    assert "action-limit adjudicated" in result["qualification"]
    assert "complete-game" not in result["qualification"]


def test_paired_comparison_rejects_different_frozen_opponent_ledgers() -> None:
    candidate = report("student.pt", [1, 0, 1, 0])
    reference = report("source.pt", [1, 0, 1, 0])
    reference["baseline_self_play"] = {"winners": [1, 0, 1, 0]}

    with pytest.raises(ValueError, match="frozen opponent"):
        compare(candidate, reference)
