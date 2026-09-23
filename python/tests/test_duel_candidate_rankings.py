from __future__ import annotations

import copy

import pytest

from python.audit_duel_candidate_rankings import audit


def report() -> dict[str, object]:
    return {
        "dataset_sha256": "dataset",
        "source_sha256": "source",
        "candidate_score_records": [
            {
                "seed": 71,
                "root_seat": 1,
                "round": 8,
                "native_selected_index": 1,
                "static_scores": [10, 20],
                "native_reply_scores": [5, 9],
                "autonomous_reply_scores": {
                    "source": [3, 8],
                    "student": [8, 3],
                },
                "autonomous_selected_indices": {"source": 1, "student": 0},
            }
        ],
    }


def test_candidate_ranking_audit_tracks_native_improvement_by_map() -> None:
    first = report()
    corrective = copy.deepcopy(first)
    corrective["candidate_score_records"][0]["autonomous_selected_indices"][
        "student"
    ] = 1
    corrective["candidate_score_records"][0]["autonomous_reply_scores"]["student"] = [
        3,
        8,
    ]

    result = audit(first, corrective)

    assert result["changed_root_choices"] == 1
    assert result["teacher_choice_matches"] == {"first": 0, "corrective": 1}
    assert result["changed_positions"][0]["native_reply_score_delta"] == 4
    assert (
        result["independent_map_transformed_native_score_delta"]["candidate_better"]
        == 1
    )


def test_candidate_ranking_audit_rejects_unpaired_native_scores() -> None:
    first = report()
    corrective = copy.deepcopy(first)
    corrective["candidate_score_records"][0]["native_reply_scores"] = [5, 10]

    with pytest.raises(ValueError, match="unpaired root positions"):
        audit(first, corrective)
