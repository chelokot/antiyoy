from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch

from .audit_duel_first_regret import CHECKPOINT_SHA256, load_routed_policy
from .audit_duel_teacher_coverage import play_game
from .build_bundle import digest


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-teacher-trajectory-state-v1.json"
SOURCE_SHA256 = "e8c6653c212cff19c7998a8ead8e05315b5aee03153d6e212ca808f7a64a94fb"
SEED_FIRST = 6440000
MAPS = 64
ARMS = (0, 25, 100)
CHECKPOINTS = (2, 4, 8, 12)
FEATURES = ("owned_cells", "province_money", "province_profit", "unit_strength")


def turn_state(
    observation: Mapping[str, np.ndarray], root: int, turn_index: int
) -> dict[str, int]:
    if int(observation["active_players"][0]) != root:
        raise ValueError("turn trace must observe the root player")
    cell_start, cell_end = observation["cell_offsets"][:2]
    province_start, province_end = observation["province_offsets"][:2]
    owners = observation["owners"][cell_start:cell_end]
    strengths = observation["unit_strengths"][cell_start:cell_end]
    province_owners = observation["province_owners"][
        province_start:province_end
    ]
    money = observation["province_money"][province_start:province_end]
    profit = observation["province_profit"][province_start:province_end]

    def lead(values: np.ndarray, players: np.ndarray) -> int:
        return int(
            np.sum(values[players == root], dtype=np.int64)
            - np.sum(values[players == 1 - root], dtype=np.int64)
        )

    return {
        "turn": turn_index + 1,
        "round": int(observation["rounds"][0]),
        "owned_cells": lead(np.ones_like(owners), owners),
        "province_money": lead(money, province_owners),
        "province_profit": lead(profit, province_owners),
        "unit_strength": lead(strengths, owners),
    }


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
    for percentage in ARMS[1:]:
        checkpoints = []
        for checkpoint in CHECKPOINTS:
            paired: list[tuple[int, int, dict[str, int], dict[str, int]]] = []
            for seed in maps:
                for seat in (0, 1):
                    direct = cast(
                        list[dict[str, int]], indexed[seed, seat, 0]["trajectory"]
                    )
                    candidate = cast(
                        list[dict[str, int]],
                        indexed[seed, seat, percentage]["trajectory"],
                    )
                    if len(direct) >= checkpoint and len(candidate) >= checkpoint:
                        paired.append(
                            (seed, seat, direct[checkpoint - 1], candidate[checkpoint - 1])
                        )
            surviving = {
                seed
                for seed in maps
                if {seat for current, seat, _, _ in paired if current == seed} == {0, 1}
            }
            feature_differences = {}
            for feature in FEATURES:
                differences = [
                    candidate[feature] - direct[feature]
                    for _, _, direct, candidate in paired
                ]
                feature_differences[feature] = {
                    "median_paired_lead_change": float(np.median(differences))
                    if differences
                    else None,
                    "seat_median_paired_lead_change": [
                        float(np.median(seat_differences))
                        if seat_differences
                        else None
                        for seat_differences in (
                            [
                                candidate[feature] - direct[feature]
                                for _, actual_seat, direct, candidate in paired
                                if actual_seat == seat
                            ]
                            for seat in (0, 1)
                        )
                    ],
                }
            checkpoints.append(
                {
                    "root_turn": checkpoint,
                    "paired_seed_seat_games": len(paired),
                    "excluded_seed_seat_games": 2 * len(maps) - len(paired),
                    "independent_maps_with_both_seats": len(surviving),
                    "feature_differences": feature_differences,
                }
            )
        comparisons.append(
            {"teacher_percentage": percentage, "checkpoints": checkpoints}
        )
    return {
        "independent_maps": len(maps),
        "replayed_games": len(records),
        "comparisons": comparisons,
        "arm_lengths": [
            {
                "teacher_percentage": percentage,
                "median_actions": float(
                    np.median(
                        [
                            cast(dict[str, object], record["outcome"])[
                                "actions_after_intervention"
                            ]
                            for record in records
                            if record["teacher_percentage"] == percentage
                        ]
                    )
                ),
                "median_root_turns": float(
                    np.median(
                        [
                            record["root_turns"]
                            for record in records
                            if record["teacher_percentage"] == percentage
                        ]
                    )
                ),
            }
            for percentage in ARMS
        ],
    }


def audit(source_path: Path, checkpoint_path: Path) -> dict[str, object]:
    if digest(source_path) != SOURCE_SHA256:
        raise ValueError("trajectory source report hash changed")
    if digest(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("trajectory checkpoint hash changed")
    with source_path.open(encoding="utf-8") as source:
        prior = json.load(source)
    if prior["seed"] != SEED_FIRST or prior["maps"] != MAPS:
        raise ValueError("trajectory source map window changed")
    indexed = {
        (record["seed"], record["root_seat"], record["teacher_percentage"]): record
        for record in prior["records"]
    }
    torch.set_num_threads(1)
    policy, experts = load_routed_policy(checkpoint_path)
    records = []
    for seed in range(SEED_FIRST, SEED_FIRST + MAPS):
        for seat in (0, 1):
            for percentage in ARMS:
                trajectory: list[dict[str, int]] = []
                record = play_game(
                    seed,
                    seat,
                    percentage,
                    policy,
                    on_root_turn=lambda observation, turn_index: trajectory.append(
                        turn_state(observation, seat, turn_index)
                    ),
                )
                reference = indexed[seed, seat, percentage]
                for field in (
                    "outcome",
                    "root_turns",
                    "teacher_turns",
                    "teacher_actions",
                ):
                    if record[field] != reference[field]:
                        raise ValueError(f"trajectory replay changed {field}")
                record["trajectory"] = trajectory
                records.append(record)
    return {
        "kind": "persistent_teacher_root_turn_state_distribution",
        "protocol": PROTOCOL,
        "source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_experts": experts,
        "records": records,
        "summary": summarize(records),
        "qualification": "Exact replay on already inspected maps; descriptive post-treatment state distributions, not fresh strength, terminal credit, a trained student, or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.source, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
