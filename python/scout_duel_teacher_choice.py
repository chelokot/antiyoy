from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import Tensor

from antiyoy_rl.turn_credit import load_turn_credit_positions

from .build_bundle import digest
from .evaluate import paired_comparison_summary
from .scout_duel_turn_value import (
    FEATURE_NAMES,
    DuelEmbedding,
    embed_position,
    evaluate,
    prediction_scores,
)
from .scout_turn_value import train_head


def teacher_choice_examples(positions: list[DuelEmbedding]) -> tuple[Tensor, Tensor]:
    groups = [position.reply_index != 0 for position in positions]
    maps_by_group = {
        group: {
            position.position.seed
            for position, label in zip(positions, groups, strict=True)
            if label == group
        }
        for group in (False, True)
    }
    if not all(maps_by_group.values()):
        raise ValueError(
            "teacher-choice training requires static and override positions"
        )
    positions_by_map_group = Counter(
        (group, position.position.seed)
        for position, group in zip(positions, groups, strict=True)
    )
    differences = []
    weights = []
    for position, group in zip(positions, groups, strict=True):
        features = position.position.features
        chosen = position.reply_index
        alternatives = [index for index in range(len(features)) if index != chosen]
        if not alternatives:
            continue
        position_weight = 1 / (
            2
            * len(maps_by_group[group])
            * positions_by_map_group[group, position.position.seed]
            * len(alternatives)
        )
        for index in alternatives:
            differences.append(features[chosen] - features[index])
            weights.append(position_weight)
    if not differences:
        raise ValueError("teacher-choice training requires competing turn candidates")
    return torch.stack(differences), torch.as_tensor(weights, dtype=torch.float32)


def agreement(
    positions: list[DuelEmbedding], weight: Tensor, scales: Tensor
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    by_seat: dict[int, Counter[str]] = {}
    map_deltas: dict[int, int] = {}
    for position in positions:
        example = position.position
        teacher = position.reply_index
        predicted = int(prediction_scores(position, weight, scales).argmax())
        baseline = 0
        seat = by_seat.setdefault(example.seat, Counter())
        counts["positions"] += 1
        counts["teacher_matches"] += predicted == teacher
        counts["static_matches"] += baseline == teacher
        counts["teacher_overrides"] += teacher != 0
        counts["predicted_overrides"] += predicted != 0
        counts["correct_overrides"] += teacher != 0 and predicted == teacher
        counts["false_overrides"] += predicted != 0 and predicted != teacher
        counts["first_action_matches"] += (
            position.first_actions[predicted] == position.first_actions[teacher]
        )
        counts["static_first_action_matches"] += (
            position.first_actions[baseline] == position.first_actions[teacher]
        )
        seat["positions"] += 1
        seat["teacher_overrides"] += teacher != 0
        seat["correct_overrides"] += teacher != 0 and predicted == teacher
        seat["false_overrides"] += predicted != 0 and predicted != teacher
        map_deltas[example.seed] = (
            map_deltas.get(example.seed, 0)
            + int(predicted == teacher)
            - int(baseline == teacher)
        )
    grouped = paired_comparison_summary(
        sum(value > 0 for value in map_deltas.values()),
        sum(value < 0 for value in map_deltas.values()),
        sum(value == 0 for value in map_deltas.values()),
    )
    return {
        **{
            field: counts[field]
            for field in (
                "positions",
                "teacher_matches",
                "static_matches",
                "teacher_overrides",
                "predicted_overrides",
                "correct_overrides",
                "false_overrides",
                "first_action_matches",
                "static_first_action_matches",
            )
        },
        "by_seat": {
            str(seat): {
                field: by_seat[seat][field]
                for field in (
                    "positions",
                    "teacher_overrides",
                    "correct_overrides",
                    "false_overrides",
                )
            }
            for seat in sorted(by_seat)
        },
        "independent_maps": grouped,
    }


def scout(
    training_paths: list[Path], validation_paths: list[Path]
) -> dict[str, object]:
    torch.set_num_threads(1)
    training = [
        embed_position(position)
        for path in training_paths
        for position in load_turn_credit_positions(path, "teacher")
    ]
    validation = [
        embed_position(position)
        for path in validation_paths
        for position in load_turn_credit_positions(path, "teacher")
    ]
    training_maps = {position.position.seed for position in training}
    validation_maps = {position.position.seed for position in validation}
    if training_maps & validation_maps:
        raise ValueError("teacher-choice training and validation maps overlap")
    differences, example_weights = teacher_choice_examples(training)
    weight, scales = train_head(differences, example_weights)
    return {
        "kind": "procedural_duel_reply_search_teacher_choice_scout",
        "feature_names": FEATURE_NAMES,
        "training_files": [
            {"name": path.name, "sha256": digest(path)} for path in training_paths
        ],
        "validation_files": [
            {"name": path.name, "sha256": digest(path)} for path in validation_paths
        ],
        "training_maps": len(training_maps),
        "validation_maps": len(validation_maps),
        "training_pairs": len(differences),
        "weights": weight.tolist(),
        "feature_scales": scales.tolist(),
        "training_agreement": agreement(training, weight, scales),
        "validation_agreement": agreement(validation, weight, scales),
        "training_terminal_outcomes": evaluate(training, weight, scales),
        "validation_terminal_outcomes": evaluate(validation, weight, scales),
        "qualification": "Offline teacher-choice imitation and conditional outcomes only; no complete-game or policy-strength claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, action="append", type=Path)
    parser.add_argument("--validation", required=True, action="append", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(scout(arguments.train, arguments.validation), sort_keys=True))


if __name__ == "__main__":
    main()
