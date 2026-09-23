from __future__ import annotations

import pytest

from python.audit_duel_source_reply_ranking import audit, ranking_pairs


def record(
    seed: int,
    seat: int,
    native: list[int],
    predicted: list[int | None],
    teacher: int,
    source: int | None,
) -> dict[str, object]:
    return {
        "seed": seed,
        "root_seat": seat,
        "round": 8,
        "native_selected_index": teacher,
        "native_reply_scores": native,
        "static_scores": native,
        "autonomous_selected_indices": {"source": source, "student": 0},
        "autonomous_reply_scores": {"source": predicted, "student": predicted},
    }


def test_ranking_pairs_ignores_native_ties_and_reports_model_ties() -> None:
    assert ranking_pairs([3, 2, 2, 0], [0, 1, 2, 1]) == (1, 3, 1)
    assert ranking_pairs([2, 1], [4, 4]) == (0, 0, 1)


def test_audit_separates_seats_and_censored_positions() -> None:
    result = audit(
        [
            record(1, 0, [0, 10, -5], [0, -1, -5], 1, 0),
            record(2, 1, [10, 0], [0, 1], 0, 1),
            record(3, 1, [0, 1], [None, 1], 1, None),
        ]
    )

    assert result["positions"] == 3
    assert result["censored_positions"] == 1
    assert result["independent_maps"] == 2
    assert result["maps_with_strict_regret"] == 2
    assert result["by_root_seat"][0]["missed_native_nonstatic_choices"] == 1
    assert result["by_root_seat"][1]["harmful_source_nonstatic_choices"] == 1
    assert result["by_root_seat"][1]["source_nonstatic_below_static_choices"] == 1
    assert len(result["strict_regret_position_ledger"]) == 2


def test_audit_rejects_teacher_choice_below_native_maximum() -> None:
    with pytest.raises(ValueError, match="maximal"):
        audit([record(1, 0, [0, 10], [0, 10], 0, 0)])
