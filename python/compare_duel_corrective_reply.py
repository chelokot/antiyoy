from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import paired_comparison_summary


def compare(
    first: dict[str, object], corrected: dict[str, object]
) -> dict[str, object]:
    fields = (
        "kind",
        "dataset_sha256",
        "source_sha256",
        "maps",
        "completed_maps",
        "truncated_maps",
        "checked_positions",
        "candidate_states",
        "paired_non_censored_candidates",
        "slates_with_all_responses",
    )
    if any(first[field] != corrected[field] for field in fields):
        raise ValueError("autonomous reports do not share the same paired arena")
    old_ledger = {item["seed"]: item for item in first["map_ledger"]}
    new_ledger = {item["seed"]: item for item in corrected["map_ledger"]}
    if old_ledger.keys() != new_ledger.keys():
        raise ValueError("autonomous reports have different independent maps")
    better = worse = same = 0
    ledger = []
    for seed in sorted(old_ledger):
        old = old_ledger[seed]
        new = new_ledger[seed]
        if (
            old["paired_candidates"] != new["paired_candidates"]
            or old["source_mean_transformed_error"]
            != new["source_mean_transformed_error"]
        ):
            raise ValueError("autonomous reports have unpaired candidate scores")
        before = old["student_mean_transformed_error"]
        after = new["student_mean_transformed_error"]
        better += after < before
        worse += after > before
        same += after == before
        ledger.append(
            {
                "seed": seed,
                "paired_candidates": old["paired_candidates"],
                "first_student_mean_error": before,
                "corrective_student_mean_error": after,
            }
        )
    seats = {}
    for seat in ("0", "1"):
        old = first["by_root_seat"][seat]
        new = corrected["by_root_seat"][seat]
        if (
            old["paired_candidates"] != new["paired_candidates"]
            or old["source_mean_transformed_error"]
            != new["source_mean_transformed_error"]
        ):
            raise ValueError("autonomous reports have unpaired root seats")
        seats[seat] = {
            "paired_candidates": old["paired_candidates"],
            "first_student_mean_error": old["student_mean_transformed_error"],
            "corrective_student_mean_error": new["student_mean_transformed_error"],
        }
    first_matches = first["native_teacher_selected_turn_matches"]["student"]
    corrected_matches = corrected["native_teacher_selected_turn_matches"]["student"]
    if (
        first["native_teacher_selected_turn_matches"]["source"]
        != corrected["native_teacher_selected_turn_matches"]["source"]
        or first["response_censored"]["source"]
        != corrected["response_censored"]["source"]
    ):
        raise ValueError("autonomous reports have different source controls")
    comparison = paired_comparison_summary(better, worse, same)
    passed = (
        better > worse
        and comparison["exact_two_sided_sign_test_p"] < 0.05
        and all(
            value["corrective_student_mean_error"] <= value["first_student_mean_error"]
            for value in seats.values()
        )
        and corrected["response_censored"]["student"]
        <= first["response_censored"]["student"]
        and corrected_matches >= first_matches
    )
    return {
        "kind": "procedural_duel_corrective_autonomous_reply_comparison",
        "dataset_sha256": first["dataset_sha256"],
        "source_sha256": first["source_sha256"],
        "first_student_head_sha256": first["student_head_sha256"],
        "corrective_student_head_sha256": corrected["student_head_sha256"],
        "independent_map_error_comparison": comparison,
        "by_root_seat": seats,
        "first_student_teacher_root_choice_matches": first_matches,
        "corrective_student_teacher_root_choice_matches": corrected_matches,
        "first_student_censored_replies": first["response_censored"]["student"],
        "corrective_student_censored_replies": corrected["response_censored"][
            "student"
        ],
        "predeclared_autonomous_gate_passed": passed,
        "map_ledger": ledger,
        "qualification": "Autonomous one-turn response fidelity, not complete-game strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_student", type=Path)
    parser.add_argument("corrective_student", type=Path)
    arguments = parser.parse_args()
    first = json.loads(arguments.first_student.read_text(encoding="utf-8"))
    corrected = json.loads(arguments.corrective_student.read_text(encoding="utf-8"))
    print(json.dumps(compare(first, corrected), sort_keys=True))


if __name__ == "__main__":
    main()
