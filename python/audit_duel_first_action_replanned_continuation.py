from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import encode_rules_batch
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    BranchOutcome,
    load_routed_policy,
    native_teacher_action,
    score,
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .audit_three_turn_shared_stop_outcomes import finish_persistently
from .build_bundle import digest
from .evaluate import paired_comparison_summary
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, checked_dataset


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-first-action-replanned-continuation-v1.json"
FIT_SHA256 = "252f82f070c4fee9c707365d72c96766385afb4827276ce7030d837f8e7a7191"
SAMPLES = 32


def summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    counts = {"teacher_better": 0, "source_better": 0, "same": 0, "censored": 0}
    by_seat = {str(seat): counts.copy() for seat in range(2)}
    for sample in samples:
        seat = cast(int, sample["root_seat"])
        branches = cast(dict[str, BranchOutcome], sample["outcomes"])
        teacher = score(branches["teacher"], seat)
        source = score(branches["source"], seat)
        if teacher is None or source is None:
            label = "censored"
        elif teacher > source:
            label = "teacher_better"
        elif teacher < source:
            label = "source_better"
        else:
            label = "same"
        counts[label] += 1
        by_seat[str(seat)][label] += 1
    paired = paired_comparison_summary(
        counts["teacher_better"], counts["source_better"], counts["same"]
    )
    worst = paired_comparison_summary(
        counts["teacher_better"],
        counts["source_better"] + counts["censored"],
        counts["same"],
    )
    return {
        **counts,
        "by_root_seat": by_seat,
        "exact_two_sided_map_sign_test_p": paired["exact_two_sided_sign_test_p"],
        "censored_as_source_better_sign_test_p": worst["exact_two_sided_sign_test_p"],
        "exploratory_advance_condition": (
            len(samples) == SAMPLES
            and counts["censored"] == 0
            and counts["teacher_better"] >= 2 * counts["source_better"]
            and all(
                seat_counts["teacher_better"] >= seat_counts["source_better"]
                for seat_counts in by_seat.values()
            )
            and cast(float, paired["exact_two_sided_sign_test_p"]) < 0.05
        ),
    }


def audit(dataset_path: Path, checkpoint_path: Path) -> dict[str, object]:
    if digest(dataset_path) != FIT_SHA256 or digest(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("counterfactual inputs disagree with the fixed protocol")
    dataset = checked_dataset(dataset_path, FIT_SEED, FIT_MAPS)
    torch.set_num_threads(1)
    policy, experts = load_routed_policy(checkpoint_path)
    samples: list[dict[str, object]] = []
    selected_maps: set[int] = set()
    with torch.inference_mode():
        for position in replay_slate_positions(dataset):
            if position.seed in selected_maps:
                continue
            observation = position.root.observe()
            record = position.record
            seat = cast(int, record["seat"])
            if int(observation["active_players"][0]) != seat:
                raise ValueError("replayed active seat differs from recorded root seat")
            teacher_plan = cast(list[list[int]], record["candidate_action_indices"])[
                cast(int, record["selected_index"])
            ]
            teacher_action = teacher_plan[0]
            rules = encode_rules_batch(position.root.rules_jsons(), torch.device("cpu"))
            source_action = int(policy.actions(observation, rules)[0])
            if source_action == teacher_action:
                continue
            replanned_action = int(
                native_teacher_action(position.root, FOLLOWUP_NODES, True)[0]
            )
            if replanned_action != teacher_action:
                raise ValueError("saved teacher first action differs from replanned search")
            legal_count = int(np.diff(observation["action_offsets"])[0])
            if not all(0 <= action < legal_count for action in (teacher_action, source_action)):
                raise ValueError("forced first action is not locally legal")
            outcomes = {}
            for label, action in (("teacher", teacher_action), ("source", source_action)):
                branch = position.root.fork(np.asarray([0], dtype=np.uint64))
                actual = branch.observe()
                if any(
                    not np.array_equal(actual[key], value)
                    for key, value in observation.items()
                ):
                    raise ValueError("counterfactual branch changed its starting observation")
                result = branch.step(np.asarray([action], dtype=np.uint64))
                outcomes[label] = finish_persistently(
                    branch,
                    result,
                    1,
                    seat,
                    policy,
                    rules,
                    root_replan_each_action=True,
                )
            samples.append(
                {
                    "seed": position.seed,
                    "root_seat": seat,
                    "round": record["round"],
                    "teacher_action_index": teacher_action,
                    "source_action_index": source_action,
                    "outcomes": outcomes,
                }
            )
            selected_maps.add(position.seed)
            if len(samples) == SAMPLES:
                break
    return {
        "kind": "fit_only_first_action_replanned_continuation_counterfactual",
        "protocol": PROTOCOL,
        "dataset_sha256": digest(dataset_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "selected_experts": experts,
        "eligible_sampled_maps": len(samples),
        "samples": samples,
        "summary": summarize(samples),
        "qualification": "Post hoc fit-map conditional branch outcomes, not fresh policy strength, a student, Elo or causal credit for later search actions",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
