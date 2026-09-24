import numpy as np

from python.audit_three_turn_plan_margins import (
    OverrideMargin,
    SharedStateStop,
    length_relation,
    shared_state_stop_indices,
    summarize_overrides,
    summarize_shared_stops,
)


def test_override_margin_summary_keeps_choices_and_length_distinct() -> None:
    rows = [
        OverrideMargin(
            1, 0, 0, 1, 1, -1.0, 0.5, 4, 2, True, -9.0, 0.0, -4.0, 0.0, -1.0, -0.5
        ),
        OverrideMargin(
            2, 1, 2, 1, 2, 0.2, -0.1, 2, 3, False, -1.0, 0.0, -0.5, 0.0, -0.2, -0.1
        ),
    ]

    assert [length_relation(row) for row in rows] == ["longer", "shorter"]
    summary = summarize_overrides(rows)
    assert summary["positions"] == 2
    assert summary["independent_maps"] == 2
    assert summary["source_selects_static"] == 1
    assert summary["student_selects_static"] == 0
    assert summary["source_selects_teacher"] == 1
    assert summary["student_selects_teacher"] == 1
    assert summary["margin_increased"] == 1
    assert summary["margin_decreased"] == 1
    assert summary["selected_longer"] == 1
    assert summary["selected_shorter"] == 1
    assert summary["selected_action_counts"] == {2: 1, 4: 1}
    assert summary["static_action_counts"] == {2: 1, 3: 1}
    assert summary["source_early_end_gap_median"] == -5.0
    assert summary["student_early_end_gap_median"] == -2.25


def test_shared_prefix_identifies_same_state_stop_and_continuation() -> None:
    plans = [[4, 7, 9, 0], [4, 7, 0], [6, 0]]
    offsets = np.asarray([0, 4, 7, 9])
    assert shared_state_stop_indices(plans, 1, offsets) == (6, 2)
    assert shared_state_stop_indices([[4, 0], [6, 0]], 1, np.asarray([0, 2, 4])) is None
    assert shared_state_stop_indices([[4, 0], [4, 0]], 1, np.asarray([0, 2, 4])) is None


def test_shared_state_stop_summary_preserves_action_and_score_evidence() -> None:
    rows = [
        SharedStateStop(1, 0, 1, 14, 8, 2, -5.0, -2.0, 20),
        SharedStateStop(2, 1, 2, 2, 0, 0, -0.5, 0.5, 5),
    ]
    summary = summarize_shared_stops(rows)
    assert summary["independent_maps"] == 2
    assert summary["next_action_kinds"] == {1: 1, 2: 1}
    assert summary["no_legal_non_end_actions"] == 0
    assert summary["source_prefers_stop"] == 0
    assert summary["student_prefers_stop"] == 1
    assert summary["teacher_score_advantage_positive"] == 2
    assert summarize_shared_stops([]) == {"positions": 0, "independent_maps": 0}
