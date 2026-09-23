from __future__ import annotations

import copy

import pytest

from python.compare_duel_corrective_reply import compare


def report() -> dict[str, object]:
    return {
        "kind": "procedural_duel_autonomous_opponent_reply_probe",
        "dataset_sha256": "dataset",
        "source_sha256": "source",
        "student_head_sha256": "first",
        "maps": 8,
        "completed_maps": 8,
        "truncated_maps": 0,
        "checked_positions": 8,
        "candidate_states": 8,
        "paired_non_censored_candidates": 8,
        "slates_with_all_responses": 8,
        "map_ledger": [
            {
                "seed": seed,
                "paired_candidates": 1,
                "source_mean_transformed_error": 2.0,
                "student_mean_transformed_error": 1.0,
            }
            for seed in range(8)
        ],
        "by_root_seat": {
            seat: {
                "paired_candidates": 4,
                "source_mean_transformed_error": 2.0,
                "student_mean_transformed_error": 1.0,
            }
            for seat in ("0", "1")
        },
        "native_teacher_selected_turn_matches": {"source": 4, "student": 5},
        "response_censored": {"source": 0, "student": 0},
    }


def test_corrective_gate_requires_map_and_seat_improvement() -> None:
    first = report()
    corrected = copy.deepcopy(first)
    corrected["student_head_sha256"] = "corrected"
    for item in corrected["map_ledger"]:
        item["student_mean_transformed_error"] = 0.5
    for item in corrected["by_root_seat"].values():
        item["student_mean_transformed_error"] = 0.5

    result = compare(first, corrected)

    assert result["independent_map_error_comparison"]["candidate_better"] == 8
    assert result["predeclared_autonomous_gate_passed"]
    corrected["by_root_seat"]["1"]["student_mean_transformed_error"] = 1.1
    assert not compare(first, corrected)["predeclared_autonomous_gate_passed"]


def test_corrective_comparison_rejects_unpaired_source_controls() -> None:
    first = report()
    corrected = copy.deepcopy(first)
    corrected["map_ledger"][0]["source_mean_transformed_error"] = 3.0

    with pytest.raises(ValueError, match="unpaired candidate scores"):
        compare(first, corrected)
