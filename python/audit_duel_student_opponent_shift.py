from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import encode_rules_batch
from antiyoy_rl.routed import RoutedPolicy

from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    create_environment,
    load_routed_policy,
    native_teacher_action,
    outcome,
)
from .audit_duel_teacher_trajectory import FEATURES, turn_state
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest
from .evaluate import load_policy


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-student-opponent-state-shift-v1.json"
STUDENT_SHA256 = "6f0670cccff396de22e780cc54bb4125d8b3267e28cc8bf0d29d0c0c6aa961f8"
SEED_FIRST = 6540000
MAPS = 16
CHECKPOINTS = (1, 4, 8, 12)
ARMS = ("student_selfplay", "student_vs_source")


def play_game(
    seed: int,
    root: int,
    arm: str,
    student: RoutedPolicy,
    source: RoutedPolicy,
) -> dict[str, object]:
    environment = create_environment(seed)
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    previous_active: int | None = None
    root_turns = 0
    actions = 0
    checkpoints: dict[str, dict[str, object]] = {}
    result = None
    while not environment.done()[0]:
        observation = environment.observe()
        active = int(observation["active_players"][0])
        if active == root or arm == "student_selfplay":
            selected = int(student.actions(observation, rules)[0])
        else:
            selected = int(source.actions(observation, rules)[0])
        if active == root and previous_active != root:
            root_turns += 1
            if root_turns in CHECKPOINTS:
                teacher = int(
                    native_teacher_action(environment, FOLLOWUP_NODES, True)[0]
                )
                checkpoints[str(root_turns)] = {
                    "state": turn_state(observation, root, root_turns - 1),
                    "student_action": selected,
                    "teacher_action": teacher,
                    "student_matches_teacher": selected == teacher,
                }
        result = environment.step(np.asarray([selected], dtype=np.uint64))
        previous_active = active
        actions += 1
    if result is None:
        raise ValueError("student trajectory ended without an action")
    return {
        "seed": seed,
        "root_seat": root,
        "arm": arm,
        "root_turns": root_turns,
        "checkpoints": checkpoints,
        "outcome": outcome(result, actions),
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    indexed = {
        (
            cast(int, row["seed"]),
            cast(int, row["root_seat"]),
            cast(str, row["arm"]),
        ): row
        for row in records
    }
    maps = sorted({cast(int, row["seed"]) for row in records})
    checkpoints = []
    for turn in CHECKPOINTS:
        pairs = []
        for seed in maps:
            for seat in (0, 1):
                selfplay = cast(
                    dict[str, dict[str, object]],
                    indexed[seed, seat, ARMS[0]]["checkpoints"],
                )
                source = cast(
                    dict[str, dict[str, object]],
                    indexed[seed, seat, ARMS[1]]["checkpoints"],
                )
                if str(turn) in selfplay and str(turn) in source:
                    pairs.append((seat, selfplay[str(turn)], source[str(turn)]))
        by_seat = []
        for seat in (0, 1):
            subset = [
                (first, second)
                for actual_seat, first, second in pairs
                if actual_seat == seat
            ]
            by_seat.append(
                {
                    "seat": seat,
                    "pairs": len(subset),
                    "selfplay_teacher_matches": sum(
                        cast(bool, first["student_matches_teacher"])
                        for first, _ in subset
                    ),
                    "source_opponent_teacher_matches": sum(
                        cast(bool, second["student_matches_teacher"])
                        for _, second in subset
                    ),
                }
            )
        checkpoints.append(
            {
                "root_turn": turn,
                "paired_seed_seat_games": len(pairs),
                "excluded_seed_seat_games": len(maps) * 2 - len(pairs),
                "by_root_seat": by_seat,
                "median_source_opponent_minus_selfplay_lead": {
                    feature: (
                        float(
                            np.median(
                                [
                                    cast(dict[str, int], second["state"])[feature]
                                    - cast(dict[str, int], first["state"])[feature]
                                    for _, first, second in pairs
                                ]
                            )
                        )
                        if pairs
                        else None
                    )
                    for feature in FEATURES
                },
            }
        )
    return {
        "maps": len(maps),
        "games": len(records),
        "terminal_games": sum(
            cast(dict[str, object], row["outcome"])["terminal"] for row in records
        ),
        "checkpoints": checkpoints,
    }


def audit(student_path: Path, source_path: Path) -> dict[str, object]:
    if (
        digest(student_path) != STUDENT_SHA256
        or digest(source_path) != CHECKPOINT_SHA256
    ):
        raise ValueError("student state-shift inputs disagree with the protocol")
    torch.set_num_threads(1)
    student_model, config = load_policy(student_path, torch.device("cpu"))
    student = RoutedPolicy({"student": student_model}, ["student", "student"])
    source, experts = load_routed_policy(source_path)
    records = []
    with torch.inference_mode():
        for seed in range(SEED_FIRST, SEED_FIRST + MAPS):
            for root in (0, 1):
                for arm in ARMS:
                    records.append(play_game(seed, root, arm, student, source))
    return {
        "kind": "read_only_rejected_student_opponent_state_shift",
        "protocol": PROTOCOL,
        "student_sha256": STUDENT_SHA256,
        "source_sha256": CHECKPOINT_SHA256,
        "student_expert": config["selected_expert"],
        "source_experts": experts,
        "records": records,
        "summary": summarize(records),
        "qualification": "Already inspected fit-map trajectories and post-treatment checkpoint survival; descriptive state distribution and same-state teacher agreement only, not game strength, causal failure mechanism, Elo or student promotion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    parser.add_argument("source", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.student, arguments.source), sort_keys=True))


if __name__ == "__main__":
    main()
