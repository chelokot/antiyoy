from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.distributions import kl_divergence

from antiyoy_rl.model import UniversalPolicy, action_distribution, encode_rules
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_three_turn_plan_feasibility import candidate_decisions, selected_plan
from .build_bundle import digest
from .evaluate import load_policy, paired_comparison_summary
from .train_duel_opponent_plan import source_logits


FIT_SEED = 6451000
FIT_MAPS = 128
VALIDATION_SEED = 6452000
VALIDATION_MAPS = 64
EPOCHS = 2
KL_WEIGHT = 0.2
SOURCE_SHA256 = "68549a5af87e4d2d065a164e39bc515c2c1d48276f437494d757e25c3ef28867"


def plan_scores(log_probabilities: Tensor, offsets: np.ndarray) -> Tensor:
    return torch.stack(
        [
            log_probabilities[int(start) : int(end)].mean()
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        ]
    )


def listwise_objective(
    student_logits: Tensor,
    source_logits_tensor: Tensor,
    action_offsets: np.ndarray,
    action_indices: np.ndarray,
    plan_offsets: np.ndarray,
    selected_index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    student = action_distribution(student_logits, action_offsets)
    source = action_distribution(source_logits_tensor, action_offsets)
    labels = torch.as_tensor(action_indices, dtype=torch.long)
    scores = plan_scores(student.log_prob(labels), plan_offsets)
    cross_entropy = torch.nn.functional.cross_entropy(
        scores.unsqueeze(0), torch.tensor([selected_index])
    )
    retention = kl_divergence(source, student).mean()
    return cross_entropy + KL_WEIGHT * retention, cross_entropy, retention


def map_position_weights(records: list[dict[str, object]]) -> list[float]:
    counts = Counter(cast(int, record["selected_index"]) != 0 for record in records)
    groups = len(counts)
    return [
        1 / (groups * counts[cast(int, record["selected_index"]) != 0])
        for record in records
    ]


def checked_dataset(path: Path, seed: int, maps: int) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        dataset = cast(dict[str, object], json.load(source))
    generator = cast(dict[str, object], dataset["generator"])
    if (
        dataset["schema_version"] != 1
        or generator["seed"] != seed
        or generator["schema_version"] != 2
        or generator["players"] != 2
        or generator["width"] != 11
        or generator["height"] != 9
        or dataset["maps"] != maps
        or dataset["search_nodes"] != 256
        or dataset["beam_slate_size"] != 8
        or dataset["opponent_search_nodes"] != 64
        or dataset["followup_search_nodes"] != 32
        or dataset["sample_round_modulus"] != 8
        or dataset["sample_round_remainder"] != 0
        or dataset["action_limit"] != 2400
        or dataset["maximum_actions_per_turn"] != 24
        or dataset["rules"] != "classic_generic_2022"
    ):
        raise ValueError("three-turn plan dataset disagrees with the fixed protocol")
    if any(
        "candidate_action_indices" not in record
        for record in cast(list[dict[str, object]], dataset["records"])
    ):
        raise ValueError("three-turn plan dataset lacks candidate replay indices")
    return dataset


def fit_student(
    dataset: dict[str, object], student: UniversalPolicy, source: UniversalPolicy
) -> list[dict[str, float | int]]:
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    trainable = list(student.action_head.parameters())
    if student.action_residual is not None:
        trainable.extend(student.action_residual.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    source_head = copy.deepcopy(source.action_head).eval()
    source_residual = (
        copy.deepcopy(source.action_residual).eval()
        if source.action_residual is not None
        else None
    )
    optimizer = torch.optim.AdamW(trainable, lr=0.0001, weight_decay=0.01)
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in cast(list[dict[str, object]], dataset["records"]):
        by_seed[cast(int, record["seed"])].append(record)
    order = sorted(by_seed)
    shuffle = random.Random(FIT_SEED)
    history = []
    for epoch in range(EPOCHS):
        shuffle.shuffle(order)
        mean_losses = np.zeros(3, dtype=np.float64)
        updates = 0
        for seed in order:
            records = by_seed[seed]
            weights = map_position_weights(records)
            map_dataset = dict(dataset)
            map_dataset["records"] = records
            optimizer.zero_grad(set_to_none=True)
            map_losses = np.zeros(3, dtype=np.float64)
            for position, weight in zip(
                replay_slate_positions(map_dataset), weights, strict=True
            ):
                observations, indices, offsets = candidate_decisions(position)
                rules = encode_rules(position.root.rules_json(), torch.device("cpu"))
                logits, _, features = student.forward_with_action_features(
                    observations, rules
                )
                reference = source_logits(features, source_head, source_residual)
                objective, cross_entropy, retention = listwise_objective(
                    logits,
                    reference,
                    observations["action_offsets"],
                    indices,
                    offsets,
                    cast(int, position.record["selected_index"]),
                )
                (objective * weight).backward()
                map_losses += weight * np.asarray(
                    [
                        float(objective.detach()),
                        float(cross_entropy.detach()),
                        float(retention.detach()),
                    ]
                )
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            updates += 1
            mean_losses += map_losses
        history.append(
            {
                "epoch": epoch + 1,
                "map_updates": updates,
                "mean_map_objective": float(mean_losses[0] / updates),
                "mean_map_listwise_loss": float(mean_losses[1] / updates),
                "mean_map_source_kl": float(mean_losses[2] / updates),
            }
        )
    return history


def evaluate_plan_agreement(
    dataset: dict[str, object], student: UniversalPolicy, source: UniversalPolicy
) -> dict[str, object]:
    source_head = source.action_head
    source_residual = source.action_residual
    by_map: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_seat: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_override: dict[bool, list[int]] = defaultdict(lambda: [0, 0, 0])
    retained = source_first_matches = 0
    selected_lengths = []
    alternative_lengths = []
    positions = candidate_plans = decisions = 0
    for position in replay_slate_positions(dataset):
        observations, indices, offsets = candidate_decisions(position)
        rules = encode_rules(position.root.rules_json(), torch.device("cpu"))
        with torch.inference_mode():
            student_logits, _, features = student.forward_with_action_features(
                observations, rules
            )
            reference = source_logits(features, source_head, source_residual)
            labels = torch.as_tensor(indices, dtype=torch.long)
            source_scores = plan_scores(
                action_distribution(
                    reference, observations["action_offsets"]
                ).log_prob(labels),
                offsets,
            ).numpy()
            student_scores = plan_scores(
                action_distribution(
                    student_logits, observations["action_offsets"]
                ).log_prob(labels),
                offsets,
            ).numpy()
            root_logits, _, root_features = student.forward_with_action_features(
                position.root.observe(), rules
            )
            root_reference = source_logits(
                root_features, source_head, source_residual
            )
        record = position.record
        teacher = cast(int, record["selected_index"])
        static = cast(list[int], record["static_scores"])
        source_match = int(selected_plan(source_scores, static) == teacher)
        student_match = int(selected_plan(student_scores, static) == teacher)
        for bucket in (
            by_map[position.seed],
            by_seat[cast(int, record["seat"])],
            by_override[teacher != 0],
        ):
            bucket[0] += 1
            bucket[1] += source_match
            bucket[2] += student_match
        teacher_first = cast(list[list[int]], record["candidate_action_indices"])[
            teacher
        ][0]
        if int(root_reference.argmax()) == teacher_first:
            source_first_matches += 1
            retained += int(root_logits.argmax()) == teacher_first
        lengths = np.diff(offsets)
        selected_lengths.append(int(lengths[teacher]))
        alternative_lengths.extend(
            int(length) for index, length in enumerate(lengths) if index != teacher
        )
        positions += 1
        candidate_plans += len(static)
        decisions += len(indices)
    better = worse = same = 0
    for count, baseline, candidate in by_map.values():
        better += candidate > baseline
        worse += candidate < baseline
        same += candidate == baseline
    return {
        "positions": positions,
        "candidate_plans": candidate_plans,
        "candidate_decisions": decisions,
        "independent_map_comparison": paired_comparison_summary(better, worse, same),
        "by_seat": dict(sorted(by_seat.items())),
        "by_teacher_override": {
            str(key).lower(): value for key, value in sorted(by_override.items())
        },
        "source_first_action_matches_teacher": source_first_matches,
        "student_retains_those_first_actions": retained,
        "mean_selected_plan_actions": float(np.mean(selected_lengths)),
        "mean_alternative_plan_actions": float(np.mean(alternative_lengths)),
        "map_ledger": [
            {"seed": seed, "positions": values[0], "source": values[1], "student": values[2]}
            for seed, values in sorted(by_map.items())
        ],
    }


def run(
    fit_path: Path, validation_path: Path, source_path: Path, output_path: Path
) -> dict[str, object]:
    if digest(source_path) != SOURCE_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the fixed protocol")
    fit = checked_dataset(fit_path, FIT_SEED, FIT_MAPS)
    validation = checked_dataset(validation_path, VALIDATION_SEED, VALIDATION_MAPS)
    torch.set_num_threads(1)
    torch.manual_seed(FIT_SEED)
    source, config = load_policy(
        source_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    student = copy.deepcopy(source)
    started = time.perf_counter()
    history = fit_student(fit, student, source)
    training_seconds = time.perf_counter() - started
    student.eval()
    checkpoint = {
        "kind": "three_turn_whole_plan_action_head",
        "source_sha256": digest(source_path),
        "selected_expert": config["selected_expert"],
        "action_head": student.action_head.state_dict(),
        "action_residual": (
            student.action_residual.state_dict()
            if student.action_residual is not None
            else None
        ),
    }
    torch.save(checkpoint, output_path)
    return {
        "kind": "three_turn_whole_plan_distillation_offline",
        "source_sha256": digest(source_path),
        "selected_expert": config["selected_expert"],
        "fit_sha256": digest(fit_path),
        "validation_sha256": digest(validation_path),
        "fit_completed_maps": fit["completed_maps"],
        "fit_truncated_maps": fit["truncated_maps"],
        "validation_completed_maps": validation["completed_maps"],
        "validation_truncated_maps": validation["truncated_maps"],
        "checkpoint_sha256": digest(output_path),
        "checkpoint_bytes": output_path.stat().st_size,
        "training_seconds": training_seconds,
        "epochs": history,
        "fit": evaluate_plan_agreement(fit, student, source),
        "validation": evaluate_plan_agreement(validation, student, source),
        "qualification": "Offline complete-plan teacher agreement, not autonomous game strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.fit, arguments.validation, arguments.source, arguments.output), sort_keys=True))


if __name__ == "__main__":
    main()
