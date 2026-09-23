from __future__ import annotations

from collections import Counter

import pytest

from python.audit_duel_opponent_response import (
    INVALID_HEX,
    action_feature,
    has_opponent_response,
    legal_action_index,
    summarize,
)


@pytest.mark.parametrize(
    ("action", "feature"),
    [
        ("EndTurn", ("EndTurn", INVALID_HEX, INVALID_HEX, 0)),
        ({"Move": {"source": 3, "target": 8}}, ("Move", 3, 8, 0)),
        (
            {"Recruit": {"province": 7, "target": 8, "strength": 2}},
            ("Recruit", 7, 8, 2),
        ),
        (
            {"Build": {"target": 8, "structure": "StrongTower"}},
            ("Build", INVALID_HEX, 8, 2),
        ),
        ({"PlantTree": {"target": 8}}, ("PlantTree", INVALID_HEX, 8, 0)),
        (
            {"Diplomacy": {"target": 1, "command": "DeclareWar"}},
            ("Diplomacy", INVALID_HEX, 1, 0),
        ),
    ],
)
def test_action_feature_matches_native_encoding(
    action: object, feature: tuple[str, int, int, int]
) -> None:
    assert action_feature(action) == feature


def test_searched_action_must_match_exactly_one_legal_action() -> None:
    legal: list[dict[str, object]] = [
        {"kind": "EndTurn", "source": INVALID_HEX, "target": INVALID_HEX, "parameter": 0},
        {"kind": "Recruit", "source": 7, "target": 8, "parameter": 1},
        {"kind": "Recruit", "source": 7, "target": 8, "parameter": 2},
    ]
    action = {"Recruit": {"province": 7, "target": 8, "strength": 2}}
    assert legal_action_index(action, legal) == 2
    with pytest.raises(ValueError, match="unique legal action"):
        legal_action_index(action, legal[:2])
    with pytest.raises(ValueError, match="unique legal action"):
        legal_action_index(action, [legal[2], legal[2]])


def test_terminal_candidate_has_no_opponent_turn() -> None:
    assert not has_opponent_response([], root_seat=1, active_seat=1)
    assert has_opponent_response(["EndTurn"], root_seat=1, active_seat=0)
    with pytest.raises(ValueError, match="wrong opponent seat"):
        has_opponent_response(["EndTurn"], root_seat=1, active_seat=1)


def test_empty_group_summary_has_no_division_error() -> None:
    assert summarize(Counter()) == {
        "candidates": 0,
        "top_one_matches": 0,
        "top_three_matches": 0,
        "top_one_rate": 0.0,
        "top_three_rate": 0.0,
        "mean_teacher_action_probability": 0.0,
    }
