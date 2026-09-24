from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.counterfactual import PolicyActor
from antiyoy_rl.model import encode_rules_batch

from .audit_duel_first_regret import (
    ACTION_LIMIT,
    BEAM_WIDTH,
    BRANCH_WIDTH,
    CHECKPOINT_SHA256,
    MAXIMUM_ACTIONS_PER_TURN,
    REPLY_NODES,
    SEARCH_NODES,
    SLATE_SIZE,
    BranchOutcome,
    create_environment,
    load_routed_policy,
    native_teacher_action,
    outcome,
    score,
)
from .build_bundle import digest
from .evaluate import paired_comparison_summary


FOLLOWUP_NODES = 32
PROTOCOL = "benchmarks/protocols/2026-09-24-duel-three-turn-intervention-v1.json"


def complete_turn(
    environment: VectorEnv,
    root: int,
    choose: Callable[[VectorEnv], np.ndarray],
) -> tuple[list[int], Mapping[str, np.ndarray]]:
    plan: list[int] = []
    result: Mapping[str, np.ndarray] | None = None
    while (
        not environment.done()[0]
        and int(environment.observe()["active_players"][0]) == root
    ):
        action = int(choose(environment)[0])
        result = environment.step(np.asarray([action], dtype=np.uint64))
        plan.append(action)
    if result is None:
        raise ValueError("completed-turn intervention requires an active root")
    return plan, result


def finish_with_direct(
    environment: VectorEnv,
    policy: PolicyActor,
    rule_features: torch.Tensor,
    result: Mapping[str, np.ndarray],
    actions: int,
) -> BranchOutcome:
    while not environment.done()[0]:
        selected = policy.actions(environment.observe(), rule_features)
        result = environment.step(selected)
        actions += 1
    return outcome(result, actions)


def sample_first_disagreement(
    seed: int,
    root: int,
    policy: PolicyActor,
    maximum_round: int,
) -> dict[str, object]:
    environment = create_environment(seed)
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    previous_active: int | None = None
    observed_root_turns = 0
    last_result: Mapping[str, np.ndarray] | None = None
    while not environment.done()[0]:
        observation = environment.observe()
        active = int(observation["active_players"][0])
        round_number = int(observation["rounds"][0])
        root_turn_start = active == root and previous_active != root
        if root_turn_start and round_number > maximum_round:
            break
        direct = policy.actions(observation, rules)
        if root_turn_start:
            observed_root_turns += 1
            teacher = native_teacher_action(environment, FOLLOWUP_NODES)
            if int(direct[0]) != int(teacher[0]):
                direct_branch = environment.fork(np.asarray([0], dtype=np.uint64))
                teacher_branch = environment.fork(np.asarray([0], dtype=np.uint64))
                direct_plan, direct_result = complete_turn(
                    direct_branch,
                    root,
                    lambda branch: policy.actions(branch.observe(), rules),
                )
                teacher_plan, teacher_result = complete_turn(
                    teacher_branch,
                    root,
                    lambda branch: native_teacher_action(branch, FOLLOWUP_NODES),
                )
                if direct_plan[0] != int(direct[0]) or teacher_plan[0] != int(
                    teacher[0]
                ):
                    raise RuntimeError("forked first actions changed")
                return {
                    "seed": seed,
                    "root_seat": root,
                    "round": round_number,
                    "observed_root_turns": observed_root_turns,
                    "direct_plan": direct_plan,
                    "teacher_plan": teacher_plan,
                    "outcomes": {
                        "direct": finish_with_direct(
                            direct_branch,
                            policy,
                            rules,
                            direct_result,
                            len(direct_plan),
                        ),
                        "teacher": finish_with_direct(
                            teacher_branch,
                            policy,
                            rules,
                            teacher_result,
                            len(teacher_plan),
                        ),
                    },
                }
        last_result = environment.step(direct)
        previous_active = active
    stop_reason = (
        "round_limit"
        if not environment.done()[0]
        else "action_limit"
        if bool(last_result["truncated"][0])
        else "terminal"
    )
    return {
        "seed": seed,
        "root_seat": root,
        "observed_root_turns": observed_root_turns,
        "no_intervention_reason": stop_reason,
    }


def summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    map_differences: dict[int, list[float]] = defaultdict(list)
    by_seat = {
        seat: {"teacher_better": 0, "direct_better": 0, "same": 0, "censored": 0}
        for seat in (0, 1)
    }
    stop_reasons: dict[str, int] = defaultdict(int)
    for sample in samples:
        if "outcomes" not in sample:
            stop_reasons[cast(str, sample["no_intervention_reason"])] += 1
            continue
        seat = cast(int, sample["root_seat"])
        branches = cast(dict[str, BranchOutcome], sample["outcomes"])
        direct = score(branches["direct"], seat)
        teacher = score(branches["teacher"], seat)
        if direct is None or teacher is None:
            by_seat[seat]["censored"] += 1
            continue
        difference = teacher - direct
        map_differences[cast(int, sample["seed"])].append(difference)
        label = (
            "teacher_better"
            if difference > 0
            else "direct_better"
            if difference < 0
            else "same"
        )
        by_seat[seat][label] += 1
    map_nets = [sum(values) for values in map_differences.values()]
    teacher_better = sum(value > 0 for value in map_nets)
    direct_better = sum(value < 0 for value in map_nets)
    same = len(map_nets) - teacher_better - direct_better
    return {
        "games_examined": len(samples),
        "sampled_positions": sum("outcomes" in sample for sample in samples),
        "no_intervention_reasons": dict(stop_reasons),
        "by_root_seat": by_seat,
        "independent_maps_with_uncensored_pairs": len(map_nets),
        "maps_with_both_seats_uncensored": sum(
            len(values) == 2 for values in map_differences.values()
        ),
        "map_net_terminal_outcome": {
            "teacher_better": teacher_better,
            "direct_better": direct_better,
            "same": same,
            "exact_two_sided_sign_test_p": paired_comparison_summary(
                teacher_better, direct_better, same
            )["exact_two_sided_sign_test_p"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=6430000)
    parser.add_argument("--maps", type=int, default=64)
    parser.add_argument("--maximum-round", type=int, default=16)
    arguments = parser.parse_args()
    checkpoint_sha256 = digest(arguments.checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError(
            "intervention protocol requires the frozen routed-v6 checkpoint"
        )
    torch.set_num_threads(1)
    policy, experts = load_routed_policy(arguments.checkpoint)
    samples = [
        sample_first_disagreement(seed, seat, policy, arguments.maximum_round)
        for seed in range(arguments.seed, arguments.seed + arguments.maps)
        for seat in (0, 1)
    ]
    print(
        json.dumps(
            {
                "kind": "three_turn_teacher_first_action_intervention",
                "protocol": PROTOCOL,
                "checkpoint_sha256": checkpoint_sha256,
                "selected_experts": experts,
                "seed": arguments.seed,
                "maps": arguments.maps,
                "maximum_round": arguments.maximum_round,
                "action_limit": ACTION_LIMIT,
                "teacher": {
                    "root_nodes": SEARCH_NODES,
                    "slate_size": SLATE_SIZE,
                    "opponent_reply_nodes": REPLY_NODES,
                    "root_followup_nodes": FOLLOWUP_NODES,
                    "beam_width": BEAM_WIDTH,
                    "branch_width": BRANCH_WIDTH,
                    "maximum_actions_per_turn": MAXIMUM_ACTIONS_PER_TURN,
                },
                "samples": samples,
                "summary": summarize(samples),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
