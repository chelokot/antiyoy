from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import encode_rules_batch
from antiyoy_rl.routed import RoutedPolicy

from .audit_duel_first_regret import (
    ACTION_LIMIT,
    CHECKPOINT_SHA256,
    BranchOutcome,
    create_environment,
    load_routed_policy,
    native_teacher_action,
    outcome,
)
from .audit_duel_teacher_blocks import paired_outcomes
from .audit_duel_teacher_coverage import finite_score
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-teacher-plan-cache-v1.json"
SEED_FIRST = 6490000
MAPS = 16
ARMS = ("direct", "cached_teacher", "replanned_teacher")


def replanned_teacher_action(environment: VectorEnv) -> np.ndarray:
    branch = environment.fork(np.asarray([0], dtype=np.uint64))
    np.testing.assert_array_equal(
        environment.observe()["action_offsets"], branch.observe()["action_offsets"]
    )
    return native_teacher_action(branch, FOLLOWUP_NODES)


def play_game(
    seed: int, root: int, arm: str, policy: RoutedPolicy
) -> dict[str, object]:
    environment = create_environment(seed)
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    previous_active: int | None = None
    root_turns = 0
    root_actions = 0
    action_in_turn = 0
    compared = [0, 0, 0, 0]
    disagreements = [0, 0, 0, 0]
    steps = 0
    result = None
    while not environment.done()[0]:
        observation = environment.observe()
        active = int(observation["active_players"][0])
        if active == root:
            if previous_active != root:
                root_turns += 1
                action_in_turn = 0
            if arm == "cached_teacher":
                selected = native_teacher_action(environment, FOLLOWUP_NODES)
                replanned = replanned_teacher_action(environment)
                bucket = min(action_in_turn, 3)
                compared[bucket] += 1
                disagreements[bucket] += int(int(selected[0]) != int(replanned[0]))
            elif arm == "replanned_teacher":
                selected = replanned_teacher_action(environment)
            else:
                selected = policy.actions(observation, rules)
            root_actions += 1
            action_in_turn += 1
        else:
            selected = policy.actions(observation, rules)
        result = environment.step(np.asarray(selected, dtype=np.uint64))
        previous_active = active
        steps += 1
    if result is None:
        raise ValueError("teacher plan-cache game ended without an action")
    return {
        "seed": seed,
        "root_seat": root,
        "arm": arm,
        "root_turns": root_turns,
        "root_actions": root_actions,
        "cache_replan_comparisons_by_action_position": compared,
        "cache_replan_disagreements_by_action_position": disagreements,
        "outcome": outcome(result, steps),
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    indexed = {
        (cast(int, row["seed"]), cast(int, row["root_seat"]), cast(str, row["arm"])): row
        for row in records
    }
    maps = sorted({cast(int, row["seed"]) for row in records})
    arms = []
    for arm in ARMS:
        subset = [row for row in records if row["arm"] == arm]
        arms.append(
            {
                "arm": arm,
                "games": len(subset),
                "wins_by_root_seat": [
                    sum(
                        finite_score(cast(BranchOutcome, row["outcome"]), seat) == 1
                        for row in subset
                        if row["root_seat"] == seat
                    )
                    for seat in (0, 1)
                ],
                "truncated": sum(
                    cast(BranchOutcome, row["outcome"])["truncated"] for row in subset
                ),
                "root_turns": sum(cast(int, row["root_turns"]) for row in subset),
                "root_actions": sum(cast(int, row["root_actions"]) for row in subset),
                "versus_direct": paired_outcomes(indexed, maps, arm, "direct"),
            }
        )
    cached = [row for row in records if row["arm"] == "cached_teacher"]
    return {
        "independent_maps": len(maps),
        "arms": arms,
        "cached_vs_fresh_replan": {
            "compared_by_action_position": [
                sum(
                    cast(list[int], row["cache_replan_comparisons_by_action_position"])[
                        position
                    ]
                    for row in cached
                )
                for position in range(4)
            ],
            "disagreements_by_action_position": [
                sum(
                    cast(list[int], row["cache_replan_disagreements_by_action_position"])[
                        position
                    ]
                    for row in cached
                )
                for position in range(4)
            ],
        },
        "cached_vs_replanned_games": paired_outcomes(
            indexed, maps, "cached_teacher", "replanned_teacher"
        ),
    }


def audit(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_sha256 = digest(checkpoint_path)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("plan-cache audit requires the frozen routed-v6 checkpoint")
    torch.set_num_threads(1)
    policy, experts = load_routed_policy(checkpoint_path)
    records = [
        play_game(seed, seat, arm, policy)
        for seed in range(SEED_FIRST, SEED_FIRST + MAPS)
        for seat in (0, 1)
        for arm in ARMS
    ]
    return {
        "kind": "native_teacher_plan_cache_and_replanning_diagnostic",
        "protocol": PROTOCOL,
        "seed_first": SEED_FIRST,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_experts": experts,
        "records": records,
        "summary": summarize(records),
        "qualification": "Fresh same-state plan-cache diagnostic and matched complete games, not trained-student strength or global Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
