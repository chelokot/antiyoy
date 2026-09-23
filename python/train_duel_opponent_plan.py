from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import kl_divergence

from antiyoy_rl.model import UniversalPolicy, action_distribution, encode_rules_batch
from antiyoy_rl.slate_dataset import TeacherSlatePosition, load_teacher_slates

from .audit_duel_opponent_trajectory import measure_positions
from .build_bundle import digest
from .evaluate import load_policy, paired_comparison_summary


SEED = 6346000
EPOCHS = 4
KL_WEIGHT = 0.2


def action_loss(
    student_logits: Tensor,
    source_logits: Tensor,
    offsets: np.ndarray,
    labels: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor]:
    student = action_distribution(student_logits, offsets)
    source = action_distribution(source_logits, offsets)
    targets = torch.as_tensor(labels, dtype=torch.long, device=student_logits.device)
    cross_entropy = -student.log_prob(targets).mean()
    retention = kl_divergence(source, student).mean()
    return cross_entropy + KL_WEIGHT * retention, cross_entropy, retention


def source_logits(
    features: Tensor,
    action_head: nn.Module,
    action_residual: nn.Module | None,
) -> Tensor:
    with torch.no_grad():
        scores = action_head(features.detach())
        if action_residual is not None:
            scores = scores + action_residual(features.detach())
    return scores.squeeze(1)


def position_loss(
    position: TeacherSlatePosition,
    model: UniversalPolicy,
    action_head: nn.Module,
    action_residual: nn.Module | None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    trace = position.opponent_decisions
    if trace is None:
        raise ValueError("opponent-plan training requires searched decision traces")
    if len(trace.action_indices) == 0:
        return None
    rules = encode_rules_batch(list(trace.rules_json), torch.device("cpu"))
    logits, _, features = model.forward_with_action_features(trace.observation, rules)
    original = source_logits(features, action_head, action_residual)
    return action_loss(
        logits,
        original,
        trace.observation["action_offsets"],
        trace.action_indices,
    )


def train(
    positions: list[TeacherSlatePosition], model: UniversalPolicy
) -> list[dict[str, float | int]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source_head = copy.deepcopy(model.action_head).eval()
    source_residual = (
        copy.deepcopy(model.action_residual).eval()
        if model.action_residual is not None
        else None
    )
    trainable = [*model.action_head.parameters()]
    if model.action_residual is not None:
        trainable.extend(model.action_residual.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable, lr=0.0001, weight_decay=0.01)
    by_map: dict[int, list[TeacherSlatePosition]] = defaultdict(list)
    for position in positions:
        by_map[position.seed].append(position)
    order = sorted(by_map)
    shuffle = random.Random(SEED)
    epochs = []
    for epoch in range(EPOCHS):
        shuffle.shuffle(order)
        losses = np.zeros(3, dtype=np.float64)
        updates = 0
        for seed in order:
            eligible = [
                position
                for position in by_map[seed]
                if position.opponent_decisions is not None
                and len(position.opponent_decisions.action_indices) > 0
            ]
            if not eligible:
                continue
            optimizer.zero_grad(set_to_none=True)
            for position in eligible:
                result = position_loss(position, model, source_head, source_residual)
                if result is None:
                    raise AssertionError("eligible position had no searched decisions")
                objective, cross_entropy, retention = result
                (objective / len(eligible)).backward()
                losses += np.asarray(
                    [
                        float(objective.detach()),
                        float(cross_entropy.detach()),
                        float(retention.detach()),
                    ]
                ) / len(eligible)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            updates += 1
        epochs.append(
            {
                "epoch": epoch + 1,
                "map_updates": updates,
                "mean_map_objective": float(losses[0] / updates),
                "mean_map_cross_entropy": float(losses[1] / updates),
                "mean_map_source_kl": float(losses[2] / updates),
            }
        )
    return epochs


def compared_measurements(
    positions: list[TeacherSlatePosition],
    source: UniversalPolicy,
    student: UniversalPolicy,
) -> dict[str, object]:
    baseline = measure_positions(positions, source, include_maps=True)
    candidate = measure_positions(positions, student, include_maps=True)
    baseline_groups = baseline["groups"]
    candidate_groups = candidate["groups"]
    assert isinstance(baseline_groups, dict) and isinstance(candidate_groups, dict)
    seeds = sorted({position.seed for position in positions})
    ledger = []
    better = worse = same = 0
    for seed in seeds:
        key = f"map_{seed}"
        before = baseline_groups[key]
        after = candidate_groups[key]
        before_rate = before["whole_plan_rate"]
        after_rate = after["whole_plan_rate"]
        better += after_rate > before_rate
        worse += after_rate < before_rate
        same += after_rate == before_rate
        ledger.append(
            {
                "seed": seed,
                "plans": before["plans"],
                "source_exact": before["whole_plan_matches"],
                "student_exact": after["whole_plan_matches"],
            }
        )
    return {
        "positions": baseline["positions"],
        "independent_maps": baseline["independent_maps"],
        "terminal_candidates_without_opponent_plan": baseline[
            "terminal_candidates_without_opponent_plan"
        ],
        "source": {
            key: value
            for key, value in baseline_groups.items()
            if not key.startswith("map_")
        },
        "student": {
            key: value
            for key, value in candidate_groups.items()
            if not key.startswith("map_")
        },
        "independent_map_whole_plan_comparison": paired_comparison_summary(
            better, worse, same
        ),
        "map_ledger": ledger,
    }


def run(
    fit_path: Path,
    validation_path: Path,
    source_path: Path,
    output_path: Path,
) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    fit = load_teacher_slates(fit_path)
    validation = load_teacher_slates(validation_path)
    if set(position.seed for position in fit) & set(
        position.seed for position in validation
    ):
        raise ValueError("fit and validation maps overlap")
    model, config = load_policy(
        source_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    source = copy.deepcopy(model).eval()
    started = time.perf_counter()
    epochs = train(fit, model)
    elapsed = time.perf_counter() - started
    model.eval()
    checkpoint = {
        "kind": "opponent_action_head_scout",
        "source_sha256": digest(source_path),
        "selected_expert": config["selected_expert"],
        "action_head": model.action_head.state_dict(),
        "action_residual": (
            model.action_residual.state_dict()
            if model.action_residual is not None
            else None
        ),
    }
    torch.save(checkpoint, output_path)
    return {
        "kind": "procedural_duel_opponent_plan_imitation_scout",
        "source_sha256": digest(source_path),
        "selected_expert": config["selected_expert"],
        "fit_sha256": digest(fit_path),
        "validation_sha256": digest(validation_path),
        "checkpoint_sha256": digest(output_path),
        "checkpoint_size_bytes": output_path.stat().st_size,
        "training_elapsed_seconds": elapsed,
        "epochs": epochs,
        "fit": compared_measurements(fit, source, model),
        "validation": compared_measurements(validation, source, model),
        "qualification": "Teacher-forced searched opponent action imitation, not autonomous rollout, game strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.fit,
                arguments.validation,
                arguments.source,
                arguments.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
