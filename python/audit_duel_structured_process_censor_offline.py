from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .audit_duel_first_regret import CHECKPOINT_SHA256
from .audit_duel_structured_process_calibration import (
    COMPONENT_SCALES,
    StructuredResponseModel,
    process_scores,
)
from .fit_duel_structured_process_censor import (
    FIRST_SEED as FIT_FIRST_SEED,
)
from .fit_duel_structured_process_censor import (
    MAPS as FIT_MAPS,
)
from .fit_duel_structured_process_censor import (
    PROTOCOL,
    load_dataset,
    tensor_batch,
)


FIRST_SEED = 6612000
MAPS = 64


def exact_sign_p(positive: int, negative: int) -> float:
    compared = positive + negative
    if compared == 0:
        return 1.0
    tail = sum(
        math.comb(compared, index) for index in range(min(positive, negative) + 1)
    )
    return min(1.0, 2 * tail / 2**compared)


def audit(checkpoint_path: Path, dataset_path: Path) -> dict:
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["protocol"] != PROTOCOL
        or checkpoint["source_sha256"] != CHECKPOINT_SHA256
        or checkpoint["fit_seed"] != FIT_FIRST_SEED
        or checkpoint["fit_maps"] != FIT_MAPS
    ):
        raise ValueError("structured response checkpoint disagrees with protocol")
    model = StructuredResponseModel().eval()
    model.load_state_dict(checkpoint["model"])
    records, games = load_dataset(
        dataset_path, FIRST_SEED, MAPS, role="offline_validation"
    )
    map_metrics = []
    seat_student_correct = [0, 0]
    seat_static_correct = [0, 0]
    with torch.inference_mode():
        for seed in range(FIRST_SEED, FIRST_SEED + MAPS):
            samples = records[seed]
            inputs, _ = tensor_batch(samples)
            response, terminal = model(*inputs)
            offset = 0
            predicted_error = []
            baseline_error = []
            student_correct = 0
            static_correct = 0
            for sample in samples:
                candidates = sample["candidates"]
                end = offset + len(candidates)
                scores = [candidate["static_score"] for candidate in candidates]
                posts = [candidate["post_components"] for candidate in candidates]
                predicted = process_scores(
                    scores, posts, response[offset:end], terminal[offset:end]
                )
                student_index = max(
                    range(len(candidates)),
                    key=lambda index: (float(predicted[index]), scores[index], -index),
                )
                static_index = max(
                    range(len(candidates)),
                    key=lambda index: (scores[index], -index),
                )
                teacher_index = sample["teacher_selected_index"]
                student_match = int(student_index == teacher_index)
                static_match = int(static_index == teacher_index)
                student_correct += student_match
                static_correct += static_match
                seat_student_correct[sample["root_seat"]] += student_match
                seat_static_correct[sample["root_seat"]] += static_match
                post = np.asarray(posts, dtype=np.float32)
                target = np.asarray(
                    [candidate["followup_components"] for candidate in candidates],
                    dtype=np.float32,
                )
                prediction = post + response[offset:end, 1].numpy() * COMPONENT_SCALES
                predicted_error.extend(
                    np.abs((prediction - target) / COMPONENT_SCALES).mean(axis=1)
                )
                baseline_error.extend(
                    np.abs((post - target) / COMPONENT_SCALES).mean(axis=1)
                )
                offset = end
            if offset != len(response):
                raise ValueError("offline candidate predictions were not exhausted")
            map_metrics.append(
                {
                    "seed": seed,
                    "samples": len(samples),
                    "candidate_branches": offset,
                    "student_followup_mae": float(np.mean(predicted_error)),
                    "post_candidate_baseline_mae": float(np.mean(baseline_error)),
                    "student_teacher_choice_matches": student_correct,
                    "static_teacher_choice_matches": static_correct,
                }
            )
    student_mae = float(
        np.mean([metric["student_followup_mae"] for metric in map_metrics])
    )
    baseline_mae = float(
        np.mean([metric["post_candidate_baseline_mae"] for metric in map_metrics])
    )
    positive = sum(
        metric["student_teacher_choice_matches"]
        > metric["static_teacher_choice_matches"]
        for metric in map_metrics
    )
    negative = sum(
        metric["student_teacher_choice_matches"]
        < metric["static_teacher_choice_matches"]
        for metric in map_metrics
    )
    ties = MAPS - positive - negative
    gates = {
        "all_four_arms_attempted": len(games) == MAPS * 4,
        "all_branches_independently_replayed": True,
        "followup_mae_at_least_ten_percent_below_baseline": student_mae
        <= 0.9 * baseline_mae,
        "teacher_choice_map_sign_p_below_point_zero_five": exact_sign_p(
            positive, negative
        )
        < 0.05,
        "both_seats_nonnegative_choice_gain": all(
            student >= static
            for student, static in zip(
                seat_student_correct, seat_static_correct, strict=True
            )
        ),
    }
    with checkpoint_path.open("rb") as source:
        checkpoint_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    with dataset_path.open("rb") as source:
        raw_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "kind": "censor_aware_structured_response_process_offline_gate",
        "protocol": PROTOCOL,
        "checkpoint_sha256": checkpoint_sha256,
        "offline_raw_sha256": raw_sha256,
        "offline_maps": MAPS,
        "offline_games": len(games),
        "offline_terminal_games": sum(game["terminal"] for game in games.values()),
        "offline_censored_games": sum(game["truncated"] for game in games.values()),
        "offline_samples": sum(len(samples) for samples in records.values()),
        "student_followup_mae": student_mae,
        "post_candidate_baseline_mae": baseline_mae,
        "teacher_choice_map_sign": [positive, negative, ties],
        "teacher_choice_map_sign_p_two_sided": exact_sign_p(positive, negative),
        "student_teacher_choice_matches_by_seat": seat_student_correct,
        "static_teacher_choice_matches_by_seat": seat_static_correct,
        "gates": gates,
        "offline_gate_passed": all(gates.values()),
        "map_metrics": map_metrics,
        "qualification": "Branch-local offline gate only; no autonomous game strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit(arguments.checkpoint, arguments.dataset)
    summary = json.dumps(report, sort_keys=True)
    if arguments.output:
        arguments.output.write_text(f"{summary}\n")
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "map_metrics"},
                sort_keys=True,
            )
        )
    else:
        print(summary)


if __name__ == "__main__":
    main()
