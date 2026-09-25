from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import cast

import torch
from torch.distributions import kl_divergence

from antiyoy_rl.routed import RoutedPolicy

from .audit_duel_episode_preference_offline import batch_distributions
from .audit_duel_first_regret import CHECKPOINT_SHA256, load_routed_policy
from .build_bundle import digest
from .train_duel_episode_preference import (
    CHUNK_SIZE,
    DecisionTrace,
    checked_dataset,
    informative_pairs,
    replay_decisions,
)


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-episode-credit-diffusion-v1.json"
STUDENT_SHA256 = "d6014eac71789d0630837da99d971fec00ad59e0f81a3645acbb0d88360897a2"
FIT_SHA256 = "be6749499c43f857e3bc11c059bf6c8c66fa315e86b68e56132107e450f0bc97"


def common_prefix_actions(
    source: dict[str, object], teacher: dict[str, object]
) -> int:
    source_actions = cast(list[int], source["actions"])
    teacher_actions = cast(list[int], teacher["actions"])
    source_actors = cast(list[int], source["actors"])
    teacher_actors = cast(list[int], teacher["actors"])
    common = 0
    for source_move, teacher_move in zip(
        zip(source_actors, source_actions, strict=True),
        zip(teacher_actors, teacher_actions, strict=True),
    ):
        if source_move != teacher_move:
            break
        common += 1
    if common == min(len(source_actions), len(teacher_actions)):
        raise ValueError("outcome-discordant trajectories have no action divergence")
    return common


def new_bin() -> dict[str, float | int]:
    return {
        "decisions": 0,
        "source_matches_recorded": 0,
        "student_matches_recorded": 0,
        "student_loses_source_agreement": 0,
        "student_corrects_source_disagreement": 0,
        "source_to_student_kl_sum": 0.0,
    }


def audit_branch(
    student: RoutedPolicy,
    source: RoutedPolicy,
    trace: DecisionTrace,
    record: dict[str, object],
    seat: int,
    prefix: int,
    branch: str,
    bins: dict[str, dict[str, float | int]],
) -> None:
    positions = [
        index
        for index, actor in enumerate(cast(list[int], record["actors"]))
        if actor == seat
    ]
    if len(positions) != len(trace.actions):
        raise ValueError("root-decision positions disagree with replay")
    with torch.inference_mode():
        for start in range(0, len(trace.actions), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(trace.actions))
            candidate, reference = batch_distributions(
                student, source, trace, start, end
            )
            source_indices = reference.logits.argmax(dim=1).tolist()
            student_indices = candidate.logits.argmax(dim=1).tolist()
            divergences = kl_divergence(reference, candidate).tolist()
            for position, label, source_index, student_index, divergence in zip(
                positions[start:end],
                trace.actions[start:end],
                source_indices,
                student_indices,
                divergences,
                strict=True,
            ):
                if branch == "source" and source_index != label:
                    raise ValueError("frozen source disagrees with its saved action")
                if branch == "source" and position < prefix:
                    continue
                if position < prefix:
                    phase = "shared"
                elif branch == "teacher":
                    phase = "teacher_remainder"
                else:
                    phase = "source_remainder"
                item = bins[phase]
                item["decisions"] += 1
                item["source_matches_recorded"] += int(source_index == label)
                item["student_matches_recorded"] += int(student_index == label)
                item["student_loses_source_agreement"] += int(
                    source_index == label and student_index != label
                )
                item["student_corrects_source_disagreement"] += int(
                    source_index != label and student_index == label
                )
                item["source_to_student_kl_sum"] += divergence


def audit(source_path: Path, student_path: Path, fit_path: Path) -> dict[str, object]:
    if (
        digest(source_path) != CHECKPOINT_SHA256
        or digest(student_path) != STUDENT_SHA256
        or digest(fit_path) != FIT_SHA256
    ):
        raise ValueError("frozen fit evidence disagrees with the protocol")
    dataset = checked_dataset(fit_path, "fit")
    torch.set_num_threads(1)
    student, student_experts = load_routed_policy(student_path)
    source, source_experts = load_routed_policy(source_path)
    if student_experts != source_experts:
        raise ValueError("student and source routes differ")
    pairs = informative_pairs(dataset)
    bins = {
        preference: {
            phase: new_bin()
            for phase in ("shared", "teacher_remainder", "source_remainder")
        }
        for preference in ("teacher_preferred", "source_preferred")
    }
    prefixes = []
    for seed in sorted(pairs):
        for pair in pairs[seed]:
            if pair.preferred["teacher_seat"] is None:
                reference_record = pair.preferred
                teacher_record = pair.dispreferred
                preference = "source_preferred"
            else:
                reference_record = pair.dispreferred
                teacher_record = pair.preferred
                preference = "teacher_preferred"
            prefix = common_prefix_actions(reference_record, teacher_record)
            prefixes.append(prefix)
            teacher_trace = replay_decisions(teacher_record, pair.seat)
            source_trace = replay_decisions(reference_record, pair.seat)
            audit_branch(
                student,
                source,
                teacher_trace,
                teacher_record,
                pair.seat,
                prefix,
                "teacher",
                bins[preference],
            )
            audit_branch(
                student,
                source,
                source_trace,
                reference_record,
                pair.seat,
                prefix,
                "source",
                bins[preference],
            )
    for preference_bins in bins.values():
        for item in preference_bins.values():
            item["mean_source_to_student_kl"] = (
                item["source_to_student_kl_sum"] / item["decisions"]
                if item["decisions"]
                else 0.0
            )
    return {
        "kind": "read_only_fit_episode_credit_diffusion",
        "protocol": PROTOCOL,
        "fit_dataset_sha256": FIT_SHA256,
        "source_sha256": CHECKPOINT_SHA256,
        "rejected_student_sha256": STUDENT_SHA256,
        "informative_pairs": len(prefixes),
        "median_common_prefix_actions": statistics.median(prefixes),
        "bins": bins,
        "qualification": "Fit-only policy drift localization, not held-out strength, causal credit or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("rejected_student", type=Path)
    parser.add_argument("fit_dataset", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(arguments.source, arguments.rejected_student, arguments.fit_dataset),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
