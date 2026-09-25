from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import encode_rules_batch
from antiyoy_rl.routed import RoutedPolicy
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_duel_first_action_replanned_continuation import (
    FIT_SHA256,
    SAMPLES,
    summarize,
)
from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    BranchOutcome,
    load_routed_policy,
    native_teacher_action,
    outcome,
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, checked_dataset


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-four-turn-replanned-prefix-v1.json"
PREFIX_TURNS = 4


def run_branch(
    branch: VectorEnv,
    root: int,
    source: RoutedPolicy,
    rules: torch.Tensor,
    first_action: int,
    prefix: Literal["teacher", "source"],
) -> tuple[BranchOutcome, int, int]:
    actions = 0
    root_turns = 0
    prefix_root_actions = 0
    result = None
    while not branch.done()[0]:
        observation = branch.observe()
        active = int(observation["active_players"][0])
        if actions == 0:
            selected = np.asarray([first_action], dtype=np.uint64)
        elif active != root or (prefix == "source" and root_turns < PREFIX_TURNS):
            selected = source.actions(observation, rules)
        else:
            selected = native_teacher_action(branch, FOLLOWUP_NODES, True)
        if active == root and root_turns < PREFIX_TURNS:
            prefix_root_actions += 1
        result = branch.step(selected)
        actions += 1
        if active == root and (
            branch.done()[0] or int(branch.observe()["active_players"][0]) != root
        ):
            root_turns += 1
    if result is None:
        raise ValueError("counterfactual branch had no actions")
    return outcome(result, actions), min(root_turns, PREFIX_TURNS), prefix_root_actions


def audit(dataset_path: Path, checkpoint_path: Path) -> dict[str, object]:
    if (
        digest(dataset_path) != FIT_SHA256
        or digest(checkpoint_path) != CHECKPOINT_SHA256
    ):
        raise ValueError("four-turn counterfactual inputs disagree with the protocol")
    dataset = checked_dataset(dataset_path, FIT_SEED, FIT_MAPS)
    torch.set_num_threads(1)
    source, experts = load_routed_policy(checkpoint_path)
    samples: list[dict[str, object]] = []
    selected_maps: set[int] = set()
    with torch.inference_mode():
        for position in replay_slate_positions(dataset):
            if position.seed in selected_maps:
                continue
            observation = position.root.observe()
            record = position.record
            root = cast(int, record["seat"])
            if int(observation["active_players"][0]) != root:
                raise ValueError("replayed active seat differs from root seat")
            teacher_plan = cast(list[list[int]], record["candidate_action_indices"])[
                cast(int, record["selected_index"])
            ]
            teacher_action = teacher_plan[0]
            rules = encode_rules_batch(position.root.rules_jsons(), torch.device("cpu"))
            source_action = int(source.actions(observation, rules)[0])
            if source_action == teacher_action:
                continue
            replanned_action = int(
                native_teacher_action(position.root, FOLLOWUP_NODES, True)[0]
            )
            if replanned_action != teacher_action:
                raise ValueError("saved first action differs from replanned teacher")
            legal_count = int(np.diff(observation["action_offsets"])[0])
            if not all(
                0 <= action < legal_count for action in (teacher_action, source_action)
            ):
                raise ValueError("forced first action is not legal")
            outcomes: dict[str, BranchOutcome] = {}
            prefix_details = {}
            for label, action in (
                ("teacher", teacher_action),
                ("source", source_action),
            ):
                branch = position.root.fork(np.asarray([0], dtype=np.uint64))
                actual = branch.observe()
                if any(
                    not np.array_equal(actual[key], value)
                    for key, value in observation.items()
                ):
                    raise ValueError("branch differs from the pre-action observation")
                final, completed, prefix_actions = run_branch(
                    branch, root, source, rules, action, label
                )
                outcomes[label] = final
                prefix_details[label] = {
                    "completed_prefix_root_turns": completed,
                    "prefix_root_atomic_actions": prefix_actions,
                }
            samples.append(
                {
                    "seed": position.seed,
                    "root_seat": root,
                    "round": record["round"],
                    "teacher_first_action_index": teacher_action,
                    "source_first_action_index": source_action,
                    "prefix_details": prefix_details,
                    "outcomes": outcomes,
                }
            )
            selected_maps.add(position.seed)
            if len(samples) == SAMPLES:
                break
    return {
        "kind": "fit_only_four_turn_replanned_prefix_counterfactual",
        "protocol": PROTOCOL,
        "dataset_sha256": digest(dataset_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "selected_experts": experts,
        "eligible_sampled_maps": len(samples),
        "samples": samples,
        "summary": summarize(samples),
        "qualification": "Post hoc fit-map four-root-turn prefix outcomes, not a fresh policy-strength test, student, Elo or proof of unique contiguity",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
