from __future__ import annotations

from copy import deepcopy
from math import log10

import pytest

from python.audit_three_turn_direct_model import DirectReport, audit


def report() -> DirectReport:
    return {
        "checkpoint": "routed-v6.pt",
        "baseline": "reply_search",
        "model_agent": "policy",
        "profile": "classic_generic_2022",
        "generator": "procedural_v2",
        "route_generator": "procedural_v1",
        "players": 2,
        "action_limit": 2400,
        "search_nodes": 256,
        "search_beam_width": 32,
        "search_branch_width": 48,
        "search_maximum_actions_per_turn": 24,
        "reply_search_nodes": 64,
        "reply_slate_size": 8,
        "followup_search_nodes": 32,
        "games": 4,
        "seed": 700,
        "game_seeds": [700, 700, 701, 701],
        "model_seats": [0, 1, 0, 1],
        "winners": [1, 0, 1, 1],
        "game_truncated": [False, False, True, False],
        "wins": 1,
        "losses": 3,
        "draws": 0,
        "truncations": 1,
        "baseline_self_play": {"truncations": 0},
        "pairing": {
            "first_seed": 700,
            "last_seed": 701,
            "unique_seeds": 2,
            "scheme": "adjacent_same_seed_opposite_seat_v1",
        },
    }


def test_timeout_sensitivity_keeps_rotated_seats_grouped() -> None:
    result = audit(report(), 700, 2)

    assert result["native_wins_by_model_seat"] == [2, 1]
    assert result["direct_model_wins_by_model_seat"] == [0, 1]
    assert result["paired_independent_maps"]["candidate_better"] == 1
    assert result["paired_independent_maps"]["same"] == 1
    assert result["worst_case_timeout_paired_maps"]["baseline_better"] == 1
    assert result["fully_terminal_maps"] == 1
    assert result["nonterminal_games"] == 1
    assert result["head_to_head_elo_native_over_direct"] == pytest.approx(
        400 * log10(3)
    )
    assert not result["direct_model_gate_passed"]


def test_different_map_or_seat_ledger_is_rejected() -> None:
    raw = report()
    raw["model_seats"][1] = 0

    with pytest.raises(ValueError, match="map, seat or winner"):
        audit(raw, 700, 2)


def test_model_or_search_configuration_is_frozen() -> None:
    raw = deepcopy(report())
    raw["followup_search_nodes"] = 16

    with pytest.raises(ValueError, match="frozen protocol"):
        audit(raw, 700, 2)


def test_summary_must_match_game_ledger() -> None:
    raw = report()
    raw["wins"] = 2

    with pytest.raises(ValueError, match="totals disagree"):
        audit(raw, 700, 2)
