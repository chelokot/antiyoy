from __future__ import annotations

from copy import deepcopy

import pytest

from python.audit_three_turn_finite_horizon import RawReport, audit


def report() -> RawReport:
    return {
        "generator": {
            "schema_version": 2,
            "width": 11,
            "height": 9,
            "players": 2,
            "seed": 700,
            "land_density_per_million": 650_000,
            "starting_province_size": 5,
            "starting_money": 10,
            "tree_density_per_million": 150_000,
            "neutral_tower_density_per_million": 20_000,
            "neutral_capital_density_per_million": 10_000,
            "grave_density_per_million": 15_000,
        },
        "rules": "classic_generic_2022",
        "candidate": "three-turn-search",
        "baseline": "reply-search",
        "search_nodes": 256,
        "baseline_search_nodes": 256,
        "candidate_reply_nodes": 64,
        "baseline_reply_nodes": 64,
        "candidate_followup_nodes": 32,
        "candidate_slate_size": 8,
        "maps": 2,
        "games": 4,
        "baseline_reference_games": 2,
        "action_limit": 2400,
        "candidate_wins_by_seat": [1, 1],
        "baseline_wins_by_seat": [2, 0],
        "candidate_truncations": 1,
        "baseline_reference_truncations": 0,
        "records": [
            {
                "seed": 700,
                "candidate_seat": 0,
                "candidate": {"winner": 0, "truncated": False, "actions": 50},
                "baseline": {"winner": 0, "truncated": False, "actions": 51},
            },
            {
                "seed": 700,
                "candidate_seat": 1,
                "candidate": {"winner": 0, "truncated": False, "actions": 52},
                "baseline": {"winner": 0, "truncated": False, "actions": 51},
            },
            {
                "seed": 701,
                "candidate_seat": 0,
                "candidate": {"winner": 1, "truncated": False, "actions": 53},
                "baseline": {"winner": 0, "truncated": False, "actions": 54},
            },
            {
                "seed": 701,
                "candidate_seat": 1,
                "candidate": {"winner": 1, "truncated": True, "actions": 2400},
                "baseline": {"winner": 0, "truncated": False, "actions": 54},
            },
        ],
    }


def test_timeout_sensitivity_is_grouped_by_map() -> None:
    result = audit(report(), 700, 2)

    assert result["candidate_timeouts"] == 1
    assert result["baseline_reference_timeouts"] == 0
    assert result["paired_independent_maps"]["same"] == 2
    assert result["worst_case_timeout_independent_maps"]["baseline_better"] == 1
    assert result["fully_terminal_maps"] == 1
    assert result["fully_terminal_independent_maps"]["same"] == 1
    assert not result["primary_gate_without_runtime"]


def test_missing_or_duplicate_seat_is_rejected() -> None:
    raw = report()
    raw["records"][1]["candidate_seat"] = 0

    with pytest.raises(ValueError, match="map and seat ledger"):
        audit(raw, 700, 2)


def test_reference_must_be_same_within_map() -> None:
    raw = deepcopy(report())
    raw["records"][1]["baseline"]["winner"] = 1

    with pytest.raises(ValueError, match="baseline reference"):
        audit(raw, 700, 2)


def test_generator_profile_cannot_drift() -> None:
    raw = report()
    raw["generator"]["tree_density_per_million"] = 0

    with pytest.raises(ValueError, match="frozen finite-horizon"):
        audit(raw, 700, 2)


def test_win_summary_must_match_game_ledger() -> None:
    raw = report()
    raw["candidate_wins_by_seat"][1] = 0

    with pytest.raises(ValueError, match="seat wins"):
        audit(raw, 700, 2)


def test_nonterminal_episode_reaches_horizon() -> None:
    raw = report()
    raw["records"][3]["candidate"]["actions"] = 2399

    with pytest.raises(ValueError, match="nonterminal episode"):
        audit(raw, 700, 2)
