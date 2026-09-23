from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .build_bundle import digest
from .evaluate import paired_comparison_summary


def transformed_score(score: int) -> float:
    return math.asinh(score / 1000)


def audit(first: dict[str, object], corrective: dict[str, object]) -> dict[str, object]:
    if (
        first["dataset_sha256"] != corrective["dataset_sha256"]
        or first["source_sha256"] != corrective["source_sha256"]
    ):
        raise ValueError("candidate ranking audits use different paired sources")
    before = first["candidate_score_records"]
    after = corrective["candidate_score_records"]
    if len(before) != len(after):
        raise ValueError("candidate ranking audits have different position counts")
    changed = []
    by_map: dict[int, float] = defaultdict(float)
    by_seat = {
        0: {"better": 0, "worse": 0, "same": 0},
        1: {"better": 0, "worse": 0, "same": 0},
    }
    teacher_match = {"first": 0, "corrective": 0}
    positions_with_complete_predictions = 0
    for old, new in zip(before, after, strict=True):
        identity = (
            "seed",
            "root_seat",
            "round",
            "native_selected_index",
            "static_scores",
            "native_reply_scores",
        )
        if any(old[key] != new[key] for key in identity):
            raise ValueError("candidate ranking audits have unpaired root positions")
        if (
            old["autonomous_reply_scores"]["source"]
            != new["autonomous_reply_scores"]["source"]
        ):
            raise ValueError("candidate ranking audits have different source responses")
        original = old["autonomous_selected_indices"]["student"]
        updated = new["autonomous_selected_indices"]["student"]
        if original is None or updated is None:
            continue
        positions_with_complete_predictions += 1
        teacher = old["native_selected_index"]
        teacher_match["first"] += original == teacher
        teacher_match["corrective"] += updated == teacher
        if original == updated:
            continue
        native_scores = old["native_reply_scores"]
        native_delta = native_scores[updated] - native_scores[original]
        transformed_delta = transformed_score(
            native_scores[updated]
        ) - transformed_score(native_scores[original])
        by_map[old["seed"]] += transformed_delta
        outcome = (
            "better" if native_delta > 0 else "worse" if native_delta < 0 else "same"
        )
        by_seat[old["root_seat"]][outcome] += 1
        changed.append(
            {
                "seed": old["seed"],
                "root_seat": old["root_seat"],
                "round": old["round"],
                "native_teacher_index": teacher,
                "first_student_index": original,
                "corrective_student_index": updated,
                "native_reply_score_first": native_scores[original],
                "native_reply_score_corrective": native_scores[updated],
                "native_reply_score_delta": native_delta,
                "transformed_native_reply_score_delta": transformed_delta,
                "native_reply_best_vs_second_margin": sorted(
                    native_scores, reverse=True
                )[0]
                - sorted(native_scores, reverse=True)[1],
                "first_autonomous_scores": old["autonomous_reply_scores"]["student"],
                "corrective_autonomous_scores": new["autonomous_reply_scores"][
                    "student"
                ],
            }
        )
    improved = sum(delta > 0 for delta in by_map.values())
    worsened = sum(delta < 0 for delta in by_map.values())
    unchanged = sum(delta == 0 for delta in by_map.values())
    return {
        "kind": "procedural_duel_autonomous_candidate_ranking_diagnostic",
        "dataset_sha256": first["dataset_sha256"],
        "positions": len(before),
        "positions_with_complete_predictions": positions_with_complete_predictions,
        "changed_root_choices": len(changed),
        "teacher_choice_matches": teacher_match,
        "changed_choices_by_root_seat": by_seat,
        "independent_maps_with_changed_choices": len(by_map),
        "independent_map_transformed_native_score_delta": paired_comparison_summary(
            improved, worsened, unchanged
        ),
        "changed_positions": changed,
        "qualification": "Post hoc native one-turn search-score comparison of frozen autonomous root choices, not terminal outcome, game strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_student", type=Path)
    parser.add_argument("corrective_student", type=Path)
    arguments = parser.parse_args()
    first = json.loads(arguments.first_student.read_text(encoding="utf-8"))
    corrective = json.loads(arguments.corrective_student.read_text(encoding="utf-8"))
    report = audit(first, corrective)
    report["first_audit_sha256"] = digest(arguments.first_student)
    report["corrective_audit_sha256"] = digest(arguments.corrective_student)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
