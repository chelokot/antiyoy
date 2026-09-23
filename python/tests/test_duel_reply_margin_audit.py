import pytest

pytest.importorskip("torch")

import torch
from python.audit_duel_reply_margins import (
    Override,
    ResponsePattern,
    action_kind,
    first_action_kind,
    gated_choice,
    margin_prefixes,
    response_pattern_summary,
)


def test_margin_audit_uses_largest_predictions_and_independent_maps() -> None:
    overrides = [
        Override(margin=float(index), seed=70 + index, seat=index % 2, result=1)
        for index in range(1, 11)
    ]
    overrides[-2] = Override(margin=9.0, seed=79, seat=1, result=-1)

    report = margin_prefixes(overrides)

    assert report["10"]["positions"] == 1
    assert report["10"]["better"] == 1
    assert report["25"]["positions"] == 3
    assert report["25"]["better"] == 2
    assert report["25"]["worse"] == 1
    assert report["25"]["independent_maps"]["candidate_better"] == 2
    assert report["25"]["independent_maps"]["baseline_better"] == 1
    assert report["100"]["positions"] == 10
    assert margin_prefixes([]) == {}


def test_fixed_margin_keeps_static_turn_until_boundary() -> None:
    predictions = torch.as_tensor([0.0, 0.03125, 0.0625])

    assert gated_choice(predictions, 0.0626) == 0
    assert gated_choice(predictions, 0.0625) == 2


def test_opponent_response_audit_groups_plan_changes_by_map() -> None:
    patterns = [
        ResponsePattern(1, 0, 1, "Recruit", "Move", "EndTurn", 3, 1, True, True),
        ResponsePattern(1, 1, -1, "Move", "Move", "Move", 2, 2, False, True),
        ResponsePattern(2, 1, -1, "Build", "EndTurn", "EndTurn", 1, 1, False, False),
    ]

    report = response_pattern_summary(patterns, 8, 6, {1, 2, 3})

    assert report["positions_with_opponent_actions"] == 8
    assert report["slates_with_distinct_opponent_plans"] == 6
    assert report["model_overrides"] == {
        "positions": 3,
        "better": 1,
        "worse": 2,
        "same": 0,
    }
    assert report["independent_maps"]["baseline_better"] == 1
    assert report["independent_maps"]["same"] == 2
    assert report["by_opponent_first_kind_change"]["true"]["better"] == 1
    assert report["by_full_opponent_plan_change"]["true"]["worse"] == 1
    assert report["by_opponent_action_count_change"]["shorter"]["better"] == 1
    assert action_kind('"EndTurn"') == "EndTurn"
    assert action_kind('{"Move": {"source": 1, "target": 2}}') == "Move"
    assert first_action_kind(()) == "Terminal"
