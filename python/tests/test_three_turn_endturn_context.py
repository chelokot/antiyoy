import numpy as np
import pytest

from python.audit_three_turn_endturn_context import (
    end_turn_context,
    summarize,
    summarize_score_gaps,
)


def observation() -> dict[str, np.ndarray]:
    return {
        "action_offsets": np.asarray([0, 3]),
        "action_kinds": np.asarray([0, 1, 2]),
        "active_players": np.asarray([1]),
        "rounds": np.asarray([8]),
        "cell_offsets": np.asarray([0, 2]),
        "owners": np.asarray([1, 0]),
        "unit_strengths": np.asarray([2, 1]),
        "ready": np.asarray([1, 1]),
        "province_offsets": np.asarray([0, 2]),
        "province_owners": np.asarray([1, 0]),
        "province_money": np.asarray([12, 5]),
    }


def test_end_turn_context_counts_available_alternatives_and_owned_resources() -> None:
    context = end_turn_context(observation(), 0, 0)

    assert context.active_player == 1
    assert context.round == 8
    assert context.other_legal_actions == 2
    assert context.other_legal_action_kinds[1] == 1
    assert context.other_legal_action_kinds[2] == 1
    assert context.ready_owned_units == 1
    assert context.owned_province_money == 12
    assert summarize([context])["optional_endturn"] == 1


def test_end_turn_context_rejects_non_end_turn_label() -> None:
    with pytest.raises(ValueError, match="unique legal EndTurn"):
        end_turn_context(observation(), 0, 1)


def test_score_gap_summary_keeps_immediate_and_followup_distinct() -> None:
    summary = summarize_score_gaps([(-4, 12), (1, 0), (0, 3)])

    assert summary["immediate_teacher_better_equal_worse"] == [1, 1, 1]
    assert summary["followup_teacher_better_equal_worse"] == [2, 1, 0]
