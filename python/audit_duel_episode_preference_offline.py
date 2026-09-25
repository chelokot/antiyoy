from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

import torch
from torch.distributions import kl_divergence

from antiyoy_rl.model import action_distribution, concatenate_observations, encode_rules_batch
from antiyoy_rl.routed import RoutedPolicy

from .audit_duel_first_regret import CHECKPOINT_SHA256, load_routed_policy
from .build_bundle import digest
from .collect_duel_episode_preference_window import PROTOCOL
from .train_duel_episode_preference import (
    CHUNK_SIZE,
    DecisionTrace,
    checked_dataset,
    informative_pairs,
    replay_decisions,
)


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    trials = positive + negative
    if not trials:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(min(positive, negative) + 1))
    return min(1.0, 2 * tail / 2**trials)


def score_trace(
    student: RoutedPolicy, reference: RoutedPolicy, trace: DecisionTrace
) -> tuple[float, float, int]:
    margin = 0.0
    total_kl = 0.0
    with torch.inference_mode():
        for start in range(0, len(trace.actions), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(trace.actions))
            observation = concatenate_observations(trace.observations[start:end])
            rules = encode_rules_batch(
                [trace.rules_json] * (end - start), torch.device("cpu")
            )
            student_logits, _ = student(observation, rules)
            reference_logits, _ = reference(observation, rules)
            offsets = observation["action_offsets"]
            student_distribution = action_distribution(student_logits, offsets)
            reference_distribution = action_distribution(reference_logits, offsets)
            labels = torch.as_tensor(trace.actions[start:end], dtype=torch.long)
            margin += float(
                (
                    student_distribution.log_prob(labels)
                    - reference_distribution.log_prob(labels)
                ).sum()
            )
            total_kl += float(
                kl_divergence(reference_distribution, student_distribution).sum()
            )
    return margin, total_kl, len(trace.actions)


def margin_summary(
    margins: dict[int, list[tuple[int, float]]],
    informative_pairs_count: int,
    mean_kl: float,
    complete: bool,
) -> dict[str, object]:
    map_better = 0
    map_worse = 0
    map_same = 0
    by_seat = [
        {"seat": seat, "positive": 0, "negative": 0, "zero": 0} for seat in (0, 1)
    ]
    for map_pairs in margins.values():
        map_margin = sum(margin for _, margin in map_pairs)
        if map_margin > 0:
            map_better += 1
        elif map_margin < 0:
            map_worse += 1
        else:
            map_same += 1
        for seat, margin in map_pairs:
            key = "positive" if margin > 0 else "negative" if margin < 0 else "zero"
            by_seat[seat][key] += 1
    p_value = exact_two_sided_sign_p(map_better, map_worse)
    return {
        "informative_pairs": informative_pairs_count,
        "map_positive": map_better,
        "map_negative": map_worse,
        "map_zero": map_same,
        "exact_two_sided_map_sign_p": p_value,
        "by_root_seat": by_seat,
        "mean_reference_to_student_action_kl_nats": mean_kl,
        "offline_gate_passed": (
            complete
            and informative_pairs_count >= 24
            and map_better > map_worse
            and p_value < 0.05
            and all(item["positive"] >= item["negative"] for item in by_seat)
            and mean_kl <= 0.05
        ),
    }


def audit(
    source_path: Path, student_path: Path, validation_path: Path
) -> dict[str, object]:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    dataset = checked_dataset(validation_path, "offline_validation")
    summary = cast(dict[str, object], dataset["summary"])
    if not summary["data_gate_passed"]:
        raise ValueError("offline data availability gate failed")
    torch.set_num_threads(1)
    student, student_experts = load_routed_policy(student_path)
    reference, reference_experts = load_routed_policy(source_path)
    if student_experts != reference_experts:
        raise ValueError("student and reference routes differ")
    scores = {}
    total_kl = 0.0
    total_decisions = 0
    records = cast(list[dict[str, object]], dataset["records"])
    for record in records:
        seed = cast(int, record["seed"])
        arm = record["teacher_seat"]
        for seat in ((0, 1) if arm is None else (cast(int, arm),)):
            trace = replay_decisions(record, seat)
            margin, kl_sum, decisions = score_trace(student, reference, trace)
            scores[seed, arm, seat] = margin
            total_kl += kl_sum
            total_decisions += decisions
    pairs = informative_pairs(dataset)
    first_seed = cast(int, dataset["first_seed"])
    maps = cast(int, dataset["maps"])
    margins: dict[int, list[tuple[int, float]]] = {
        seed: [] for seed in range(first_seed, first_seed + maps)
    }
    for seed, map_pairs in pairs.items():
        for pair in map_pairs:
            preferred_arm = pair.preferred["teacher_seat"]
            dispreferred_arm = pair.dispreferred["teacher_seat"]
            margin = scores[seed, preferred_arm, pair.seat]
            margin -= scores[seed, dispreferred_arm, pair.seat]
            margins[seed].append((pair.seat, margin))
    mean_kl = total_kl / total_decisions
    result = margin_summary(
        margins,
        sum(len(map_pairs) for map_pairs in pairs.values()),
        mean_kl,
        summary["terminal_games"] == 192 and summary["censored_pairs"] == 0,
    )
    result["maps"] = dataset["maps"]
    result["games"] = len(records)
    result["terminal_games"] = summary["terminal_games"]
    result["censored_pairs"] = summary["censored_pairs"]
    result["scored_root_decisions"] = total_decisions
    return {
        "kind": "terminal_whole_episode_preference_offline_validation",
        "protocol": PROTOCOL,
        "source_sha256": CHECKPOINT_SHA256,
        "student_sha256": digest(student_path),
        "validation_dataset_sha256": digest(validation_path),
        "result": result,
        "qualification": "Untouched offline preference and source-retention gate only; no autonomous-game strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("student", type=Path)
    parser.add_argument("validation_dataset", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(
                arguments.source,
                arguments.student,
                arguments.validation_dataset,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
