from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

from .build_bundle import digest


class SelectedIndices(TypedDict):
    source: int | None
    student: int | None


class AutonomousScores(TypedDict):
    source: list[int | None]
    student: list[int | None]


class CandidateRecord(TypedDict):
    seed: int
    root_seat: int
    round: int
    native_selected_index: int
    native_reply_scores: list[int]
    static_scores: list[int]
    autonomous_selected_indices: SelectedIndices
    autonomous_reply_scores: AutonomousScores


def ranking_pairs(
    native: list[int], predicted: list[int]
) -> tuple[int, int, int]:
    concordant = discordant = predicted_ties = 0
    for left in range(len(native)):
        for right in range(left + 1, len(native)):
            native_gap = native[left] - native[right]
            if native_gap == 0:
                continue
            predicted_gap = predicted[left] - predicted[right]
            concordant += int(native_gap * predicted_gap > 0)
            discordant += int(native_gap * predicted_gap < 0)
            predicted_ties += int(predicted_gap == 0)
    return concordant, discordant, predicted_ties


def empty_seat() -> dict[str, int]:
    return {
        "positions": 0,
        "source_matches_native_index": 0,
        "native_nonstatic_choices": 0,
        "source_nonstatic_choices": 0,
        "source_matches_native_nonstatic_choice": 0,
        "strict_native_score_regret_positions": 0,
        "native_score_regret_sum": 0,
        "missed_native_nonstatic_choices": 0,
        "harmful_source_nonstatic_choices": 0,
        "beneficial_source_nonstatic_choices": 0,
        "source_nonstatic_below_static_choices": 0,
        "source_native_score_ties_to_static": 0,
        "concordant_pairs": 0,
        "discordant_pairs": 0,
        "predicted_tie_pairs": 0,
    }


def audit(records: list[CandidateRecord]) -> dict[str, object]:
    by_seat = {0: empty_seat(), 1: empty_seat()}
    regret_by_map: dict[int, float] = defaultdict(float)
    position_ledger = []
    censored_positions = 0
    for record in records:
        native = record["native_reply_scores"]
        predicted = record["autonomous_reply_scores"]["source"]
        choice = record["autonomous_selected_indices"]["source"]
        if choice is None or any(score is None for score in predicted):
            censored_positions += 1
            continue
        complete_predicted = cast(list[int], predicted)
        if len(native) != len(predicted) or not 0 <= choice < len(native):
            raise ValueError("candidate score and choice counts differ")
        teacher = record["native_selected_index"]
        if not 0 <= teacher < len(native):
            raise ValueError("native teacher index is outside the slate")
        if native[teacher] != max(native):
            raise ValueError("native teacher did not select a maximal reply score")
        seat = record["root_seat"]
        counts = by_seat[seat]
        regret = native[teacher] - native[choice]
        concordant, discordant, ties = ranking_pairs(native, complete_predicted)
        counts["positions"] += 1
        counts["source_matches_native_index"] += int(choice == teacher)
        counts["native_nonstatic_choices"] += int(teacher != 0)
        counts["source_nonstatic_choices"] += int(choice != 0)
        counts["source_matches_native_nonstatic_choice"] += int(
            teacher != 0 and choice == teacher
        )
        counts["strict_native_score_regret_positions"] += int(regret > 0)
        counts["native_score_regret_sum"] += regret
        counts["missed_native_nonstatic_choices"] += int(
            teacher != 0 and choice == 0 and regret > 0
        )
        counts["harmful_source_nonstatic_choices"] += int(
            teacher == 0 and choice != 0 and regret > 0
        )
        counts["beneficial_source_nonstatic_choices"] += int(
            choice != 0 and native[choice] > native[0]
        )
        counts["source_nonstatic_below_static_choices"] += int(
            choice != 0 and native[choice] < native[0]
        )
        counts["source_native_score_ties_to_static"] += int(
            choice != 0 and native[choice] == native[0]
        )
        counts["concordant_pairs"] += concordant
        counts["discordant_pairs"] += discordant
        counts["predicted_tie_pairs"] += ties
        regret_by_map[record["seed"]] += math.asinh(
            native[teacher] / 1000
        ) - math.asinh(native[choice] / 1000)
        if regret > 0:
            position_ledger.append(
                {
                    "seed": record["seed"],
                    "root_seat": seat,
                    "round": record["round"],
                    "teacher_index": teacher,
                    "source_index": choice,
                    "native_score_regret": regret,
                    "native_advantage_over_static": native[teacher] - native[0],
                    "source_predicted_gap_to_teacher": (
                        complete_predicted[choice] - complete_predicted[teacher]
                    ),
                }
            )
    return {
        "kind": "read_only_source_vs_native_reply_candidate_ranking",
        "positions": len(records),
        "censored_positions": censored_positions,
        "independent_maps": len(regret_by_map),
        "maps_with_strict_regret": sum(value > 0 for value in regret_by_map.values()),
        "transformed_native_score_regret_by_map": dict(sorted(regret_by_map.items())),
        "by_root_seat": by_seat,
        "strict_regret_position_ledger": position_ledger,
        "qualification": "Conditional native one-turn reply-score rankings on teacher-sampled positions, not terminal outcomes, complete-game strength, or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_audit", type=Path)
    arguments = parser.parse_args()
    source = json.loads(arguments.candidate_audit.read_text(encoding="utf-8"))
    report = audit(cast(list[CandidateRecord], source["candidate_score_records"]))
    report["source_report_sha256"] = digest(arguments.candidate_audit)
    report["source_dataset_sha256"] = source["dataset_sha256"]
    report["source_policy_sha256"] = source["source_sha256"]
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
