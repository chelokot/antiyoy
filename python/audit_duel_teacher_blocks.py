from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import torch

from .audit_duel_first_regret import (
    ACTION_LIMIT,
    CHECKPOINT_SHA256,
    BranchOutcome,
    load_routed_policy,
)
from .audit_duel_teacher_coverage import finite_score, play_game, teacher_turn
from .build_bundle import digest
from .evaluate import paired_comparison_summary


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-teacher-block-length-v1.json"
SEED_FIRST = 6470000
MAPS = 64
ARMS = (
    "direct",
    "first_1",
    "first_2",
    "first_4",
    "first_8",
    "distributed_25",
    "full_teacher",
)
BLOCK_LENGTHS = {"first_1": 1, "first_2": 2, "first_4": 4, "first_8": 8}


def scheduled_teacher(arm: str, seed: int, seat: int, turn: int) -> bool:
    if arm == "direct":
        return False
    if arm == "distributed_25":
        return teacher_turn(seed, seat, turn, 25)
    if arm == "full_teacher":
        return True
    return turn < BLOCK_LENGTHS[arm]


def paired_outcomes(
    indexed: dict[tuple[int, int, str], dict[str, object]],
    maps: list[int],
    arm: str,
    baseline_arm: str,
) -> dict[str, object]:
    nets = []
    terminal_nets = []
    for seed in maps:
        net = 0.0
        terminal = True
        for seat in (0, 1):
            candidate = cast(BranchOutcome, indexed[seed, seat, arm]["outcome"])
            baseline = cast(BranchOutcome, indexed[seed, seat, baseline_arm]["outcome"])
            net += finite_score(candidate, seat) - finite_score(baseline, seat)
            terminal &= not candidate["truncated"] and not baseline["truncated"]
        nets.append(net)
        if terminal:
            terminal_nets.append(net)

    def comparison(values: list[float]) -> dict[str, object]:
        better = sum(value > 0 for value in values)
        worse = sum(value < 0 for value in values)
        return paired_comparison_summary(better, worse, len(values) - better - worse)

    return {
        "baseline_arm": baseline_arm,
        "finite_horizon_maps": comparison(nets),
        "fully_terminal_maps": comparison(terminal_nets),
        "fully_terminal_map_count": len(terminal_nets),
    }


def summarize(
    records: list[dict[str, object]], candidate_arms: tuple[str, ...] = ARMS
) -> dict[str, object]:
    indexed = {
        (cast(int, row["seed"]), cast(int, row["root_seat"]), cast(str, row["arm"])): row
        for row in records
    }
    maps = sorted({cast(int, row["seed"]) for row in records})
    arms = []
    for arm in candidate_arms:
        subset = [row for row in records if row["arm"] == arm]
        outcomes = [cast(BranchOutcome, row["outcome"]) for row in subset]
        arms.append(
            {
                "arm": arm,
                "games": len(subset),
                "wins_by_root_seat": [
                    sum(
                        finite_score(cast(BranchOutcome, row["outcome"]), seat)
                        == 1
                        for row in subset
                        if row["root_seat"] == seat
                    )
                    for seat in (0, 1)
                ],
                "action_limit_adjudications": sum(
                    outcome["truncated"] for outcome in outcomes
                ),
                "actual_teacher_turns": sum(
                    cast(int, row["teacher_turns"]) for row in subset
                ),
                "total_root_turns": sum(
                    cast(int, row["root_turns"]) for row in subset
                ),
                "median_actions": float(
                    np.median([row["actions_after_intervention"] for row in outcomes])
                ),
                "versus_direct": paired_outcomes(indexed, maps, arm, "direct"),
                "versus_first_1": paired_outcomes(indexed, maps, arm, "first_1")
                if arm in ("first_2", "first_4", "first_8")
                else None,
            }
        )
    return {"independent_maps": len(maps), "arms": arms}


def audit(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_sha256 = digest(checkpoint_path)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("teacher block protocol requires the frozen routed-v6 checkpoint")
    torch.set_num_threads(1)
    policy, experts = load_routed_policy(checkpoint_path)
    records = []
    for seed in range(SEED_FIRST, SEED_FIRST + MAPS):
        for seat in (0, 1):
            for arm in ARMS:
                record = play_game(
                    seed,
                    seat,
                    0,
                    policy,
                    select_teacher=partial(scheduled_teacher, arm, seed, seat),
                )
                record["arm"] = arm
                records.append(record)
    return {
        "kind": "exact_teacher_contiguous_whole_turn_block_diagnostic",
        "protocol": PROTOCOL,
        "seed_first": SEED_FIRST,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_experts": experts,
        "records": records,
        "summary": summarize(records),
        "qualification": "Fresh complete-game mechanism diagnostic with explicit censoring, not a trained student, promotion gate or global Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
