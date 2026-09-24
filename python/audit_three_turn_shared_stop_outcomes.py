from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import encode_rules_batch, select_environments
from antiyoy_rl.routed import RoutedPolicy
from antiyoy_rl.slate_dataset import ReplayedSlatePosition, replay_slate_positions

from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    BranchOutcome,
    load_routed_policy,
    native_teacher_action,
    outcome,
    score,
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest
from .evaluate import paired_comparison_summary
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, checked_dataset


PROTOCOL = (
    "benchmarks/protocols/2026-09-24-duel-shared-stop-persistent-continuation-v1.json"
)
MAPS = 16


def shared_prefix_length(plans: list[list[int]], teacher: int) -> int | None:
    selected = plans[teacher]
    static = plans[0]
    prefix = len(selected) - 1
    if len(selected) >= len(static) or selected[:-1] != static[:prefix]:
        return None
    return prefix


def force_plan(
    position: ReplayedSlatePosition, candidate: int
) -> tuple[VectorEnv, Mapping[str, np.ndarray], int]:
    branch = position.root.fork(np.asarray([0], dtype=np.uint64))
    plan = cast(list[list[int]], position.record["candidate_action_indices"])[candidate]
    result: Mapping[str, np.ndarray] | None = None
    for action_number, index in enumerate(plan):
        result = branch.step(np.asarray([index], dtype=np.uint64))
        if branch.done()[0] and action_number + 1 != len(plan):
            raise ValueError("candidate plan continued after the game finished")
    if result is None:
        raise ValueError("candidate plan is empty")
    expected = select_environments(position.post_turn, [candidate])
    actual = branch.observe()
    if any(not np.array_equal(actual[key], value) for key, value in expected.items()):
        raise ValueError("forced plan disagrees with the native post-turn observation")
    return branch, result, len(plan)


def finish_persistently(
    branch: VectorEnv,
    last_result: Mapping[str, np.ndarray],
    actions: int,
    root: int,
    opponent: RoutedPolicy,
    rule_features: torch.Tensor,
) -> BranchOutcome:
    while not branch.done()[0]:
        observation = branch.observe()
        if int(observation["active_players"][0]) == root:
            selected = native_teacher_action(branch, FOLLOWUP_NODES)
        else:
            selected = opponent.actions(observation, rule_features)
        last_result = branch.step(selected)
        actions += 1
    return outcome(last_result, actions)


def summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    teacher_better = static_better = same = censored = 0
    by_seat = {
        seat: {"teacher_better": 0, "static_better": 0, "same": 0, "censored": 0}
        for seat in (0, 1)
    }
    for sample in samples:
        seat = cast(int, sample["root_seat"])
        branches = cast(dict[str, BranchOutcome], sample["outcomes"])
        teacher = score(branches["teacher"], seat)
        static = score(branches["static"], seat)
        if teacher is None or static is None:
            censored += 1
            by_seat[seat]["censored"] += 1
        elif teacher > static:
            teacher_better += 1
            by_seat[seat]["teacher_better"] += 1
        elif teacher < static:
            static_better += 1
            by_seat[seat]["static_better"] += 1
        else:
            same += 1
            by_seat[seat]["same"] += 1
    return {
        "independent_maps": len(samples),
        "teacher_better": teacher_better,
        "static_better": static_better,
        "same": same,
        "censored": censored,
        "by_root_seat": by_seat,
        "exact_two_sided_sign_test_p": paired_comparison_summary(
            teacher_better, static_better, same
        )["exact_two_sided_sign_test_p"],
    }


def audit(fit_path: Path, checkpoint_path: Path) -> dict[str, object]:
    if digest(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("opponent checkpoint disagrees with the fixed protocol")
    dataset = checked_dataset(fit_path, FIT_SEED, FIT_MAPS)
    torch.set_num_threads(1)
    opponent, experts = load_routed_policy(checkpoint_path)
    samples = []
    selected_maps: set[int] = set()
    for position in replay_slate_positions(dataset):
        if position.seed in selected_maps:
            continue
        record = position.record
        teacher_index = cast(int, record["selected_index"])
        if teacher_index == 0:
            continue
        plans = cast(list[list[int]], record["candidate_action_indices"])
        prefix = shared_prefix_length(plans, teacher_index)
        if prefix is None:
            continue
        seat = cast(int, record["seat"])
        rules = encode_rules_batch(position.root.rules_jsons(), torch.device("cpu"))
        branches = {}
        for label, candidate in (("teacher", teacher_index), ("static", 0)):
            branch, result, actions = force_plan(position, candidate)
            branches[label] = finish_persistently(
                branch, result, actions, seat, opponent, rules
            )
        followup = cast(list[int], record["followup_scores"])
        samples.append(
            {
                "seed": position.seed,
                "root_seat": seat,
                "round": record["round"],
                "identical_prefix_actions": prefix,
                "teacher_plan_actions": len(plans[teacher_index]),
                "static_plan_actions": len(plans[0]),
                "native_three_turn_score_gap": followup[teacher_index] - followup[0],
                "outcomes": branches,
            }
        )
        selected_maps.add(position.seed)
        if len(samples) == MAPS:
            break
    if len(samples) != MAPS:
        raise ValueError("insufficient independent maps with identical-prefix stops")
    return {
        "kind": "three_turn_shared_stop_persistent_continuation_probe",
        "protocol": PROTOCOL,
        "fit_sha256": digest(fit_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "selected_experts": experts,
        "samples": samples,
        "summary": summarize(samples),
        "qualification": "Post hoc fit-only terminal counterfactual under frozen persistent policies, not held-out policy strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.fit, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
