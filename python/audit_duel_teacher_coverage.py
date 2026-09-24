from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch

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
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest
from .evaluate import paired_comparison_summary


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-persistent-teacher-coverage-v1.json"
TEACHER_PERCENTAGES = (0, 25, 50, 75, 100)


def teacher_turn(seed: int, seat: int, turn_index: int, percentage: int) -> bool:
    if not 0 <= percentage <= 100:
        raise ValueError("teacher turn percentage must be between zero and 100")
    key = f"{seed}:{seat}:{turn_index}".encode()
    bucket = (
        int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % 100
    )
    return bucket < percentage


def play_game(
    seed: int,
    root: int,
    percentage: int,
    policy: PolicyActor,
    on_root_turn: Callable[[Mapping[str, np.ndarray], int], None] | None = None,
    select_teacher: Callable[[int], bool] | None = None,
) -> dict[str, object]:
    environment = create_environment(seed)
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    previous_active: int | None = None
    selected_teacher = False
    root_turns = 0
    teacher_turns = 0
    teacher_actions = 0
    steps = 0
    result = None
    while not environment.done()[0]:
        observation = environment.observe()
        active = int(observation["active_players"][0])
        if active == root and previous_active != root:
            if on_root_turn is not None:
                on_root_turn(observation, root_turns)
            selected_teacher = (
                select_teacher(root_turns)
                if select_teacher is not None
                else teacher_turn(seed, root, root_turns, percentage)
            )
            root_turns += 1
            teacher_turns += int(selected_teacher)
        if active == root and selected_teacher:
            selected = native_teacher_action(environment, FOLLOWUP_NODES)
            teacher_actions += 1
        else:
            selected = policy.actions(observation, rules)
        result = environment.step(selected)
        steps += 1
        previous_active = active
    if result is None:
        raise ValueError("mixture game ended without an action")
    record: dict[str, object] = {
        "seed": seed,
        "root_seat": root,
        "root_turns": root_turns,
        "teacher_turns": teacher_turns,
        "teacher_actions": teacher_actions,
        "outcome": outcome(result, steps),
    }
    if select_teacher is None:
        record["teacher_percentage"] = percentage
    return record


def finite_score(branch: BranchOutcome, root: int) -> float:
    winner = branch["adjudicated_winner"] if branch["truncated"] else branch["winner"]
    if winner == 255:
        return 0.5
    return float(winner == root)


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    indexed = {
        (
            cast(int, record["seed"]),
            cast(int, record["root_seat"]),
            cast(int, record["teacher_percentage"]),
        ): record
        for record in records
    }
    maps = sorted({cast(int, record["seed"]) for record in records})
    comparisons = []
    for percentage in TEACHER_PERCENTAGES:
        wins_by_seat = [0, 0]
        draws_by_seat = [0, 0]
        nonterminal_by_seat = [0, 0]
        teacher_turns = root_turns = 0
        map_net: list[float] = []
        terminal_map_net: list[float] = []
        for seed in maps:
            difference = 0.0
            terminal = True
            for seat in (0, 1):
                record = indexed[seed, seat, percentage]
                baseline = indexed[seed, seat, 0]
                branch = cast(BranchOutcome, record["outcome"])
                reference = cast(BranchOutcome, baseline["outcome"])
                score = finite_score(branch, seat)
                difference += score - finite_score(reference, seat)
                wins_by_seat[seat] += int(score == 1)
                draws_by_seat[seat] += int(score == 0.5)
                nonterminal_by_seat[seat] += int(branch["truncated"])
                teacher_turns += cast(int, record["teacher_turns"])
                root_turns += cast(int, record["root_turns"])
                terminal &= not branch["truncated"] and not reference["truncated"]
            map_net.append(difference)
            if terminal:
                terminal_map_net.append(difference)
        better = sum(value > 0 for value in map_net)
        worse = sum(value < 0 for value in map_net)
        same = len(map_net) - better - worse
        terminal_better = sum(value > 0 for value in terminal_map_net)
        terminal_worse = sum(value < 0 for value in terminal_map_net)
        comparisons.append(
            {
                "teacher_percentage": percentage,
                "wins_by_root_seat": wins_by_seat,
                "draws_by_root_seat": draws_by_seat,
                "nonterminal_by_root_seat": nonterminal_by_seat,
                "actual_teacher_turns": teacher_turns,
                "total_root_turns": root_turns,
                "paired_finite_horizon_maps": paired_comparison_summary(
                    better, worse, same
                ),
                "paired_fully_terminal_maps": paired_comparison_summary(
                    terminal_better,
                    terminal_worse,
                    len(terminal_map_net) - terminal_better - terminal_worse,
                ),
                "fully_terminal_maps": len(terminal_map_net),
            }
        )
    return {"independent_maps": len(maps), "arms": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=6440000)
    parser.add_argument("--maps", type=int, default=64)
    arguments = parser.parse_args()
    checkpoint_sha256 = digest(arguments.checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("coverage protocol requires the frozen routed-v6 checkpoint")
    torch.set_num_threads(1)
    policy, experts = load_routed_policy(arguments.checkpoint)
    records = [
        play_game(seed, seat, percentage, policy)
        for seed in range(arguments.seed, arguments.seed + arguments.maps)
        for seat in (0, 1)
        for percentage in TEACHER_PERCENTAGES
    ]
    print(
        json.dumps(
            {
                "kind": "persistent_teacher_whole_turn_coverage",
                "protocol": PROTOCOL,
                "checkpoint_sha256": checkpoint_sha256,
                "selected_experts": experts,
                "seed": arguments.seed,
                "maps": arguments.maps,
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
                "records": records,
                "summary": summarize(records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
