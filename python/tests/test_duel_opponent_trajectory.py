from __future__ import annotations

from collections import Counter

import numpy as np

from python.audit_duel_opponent_trajectory import first_mismatch, summarize


def test_first_mismatch_preserves_whole_plan_and_depth() -> None:
    expected = np.asarray([2, 1, 0], dtype=np.int64)

    assert first_mismatch(expected, np.asarray([2, 1, 0])) is None
    assert first_mismatch(expected, np.asarray([0, 1, 0])) == 0
    assert first_mismatch(expected, np.asarray([2, 0, 0])) == 1
    assert first_mismatch(expected, np.asarray([2, 1, 1])) == 2


def test_plan_summary_keeps_decision_and_plan_denominators_separate() -> None:
    counts = Counter(
        {
            "decisions": 5,
            "top_one_matches": 3,
            "top_three_matches": 4,
            "teacher_probability": 2.0,
            "plans": 2,
            "whole_plan_matches": 1,
            "first_mismatch_depth_1": 1,
        }
    )

    summary = summarize(counts)

    assert summary["top_one_rate"] == 0.6
    assert summary["top_three_rate"] == 0.8
    assert summary["mean_teacher_action_probability"] == 0.4
    assert summary["whole_plan_rate"] == 0.5
    assert summary["first_mismatch_depth_1"] == 1
