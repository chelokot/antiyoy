from __future__ import annotations

import numpy as np

from python.audit_duel_teacher_trajectory import summarize, turn_state


def test_turn_state_reads_root_relative_material_at_turn_start() -> None:
    observation = {
        "active_players": np.asarray([1]),
        "cell_offsets": np.asarray([0, 4]),
        "owners": np.asarray([1, 0, 1, 255]),
        "unit_strengths": np.asarray([2, 1, 0, 0]),
        "province_offsets": np.asarray([0, 2]),
        "province_owners": np.asarray([1, 0]),
        "province_money": np.asarray([12, 5]),
        "province_profit": np.asarray([3, -1]),
        "rounds": np.asarray([8]),
    }

    assert turn_state(observation, 1, 2) == {
        "turn": 3,
        "round": 8,
        "owned_cells": 1,
        "province_money": 7,
        "province_profit": 4,
        "unit_strength": 1,
    }


def test_trajectory_summary_excludes_games_missing_a_matched_turn() -> None:
    def record(seat: int, percentage: int, turns: int) -> dict[str, object]:
        return {
            "seed": 1,
            "root_seat": seat,
            "teacher_percentage": percentage,
            "root_turns": turns,
            "outcome": {"actions_after_intervention": 10},
            "trajectory": [
                {
                    "turn": turn,
                    "round": turn,
                    "owned_cells": seat + 2 * int(percentage == 100),
                    "province_money": 0,
                    "province_profit": 0,
                    "unit_strength": 0,
                }
                for turn in range(1, turns + 1)
            ],
        }

    records = [
        record(seat, percentage, 1 if seat == 1 and percentage == 100 else 2)
        for seat in (0, 1)
        for percentage in (0, 25, 100)
    ]
    summary = summarize(records)
    direct = summary["comparisons"][0]["checkpoints"][0]
    searched = summary["comparisons"][1]["checkpoints"][0]

    assert direct["paired_seed_seat_games"] == 2
    assert direct["independent_maps_with_both_seats"] == 1
    assert searched["paired_seed_seat_games"] == 1
    assert searched["excluded_seed_seat_games"] == 1
    assert searched["independent_maps_with_both_seats"] == 0
    assert searched["feature_differences"]["owned_cells"]["median_paired_lead_change"] == 2
