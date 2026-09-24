from python.audit_three_turn_plan_margins import (
    OverrideMargin,
    length_relation,
    summarize_overrides,
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
