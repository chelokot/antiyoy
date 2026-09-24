from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import cast

import numpy as np

from .audit_duel_first_regret import (
    ACTION_LIMIT,
    BranchOutcome,
    create_environment,
    native_teacher_action,
    outcome,
)
from .audit_duel_teacher_coverage import finite_score
from .audit_duel_teacher_plan_cache import replanned_teacher_action
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .evaluate import paired_comparison_summary


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-replan-head-to-head-v1.json"
SEED_FIRST = 6492000
MAPS = 128


def play_match(seed: int, replanned_seat: int) -> dict[str, object]:
    environment = create_environment(seed)
    previous_active: int | None = None
    current_turn_milliseconds = 0.0
    turn_milliseconds: list[list[float]] = [[], []]
    action_counts = [0, 0]
    steps = 0
    result = None
    while not environment.done()[0]:
        active = int(environment.observe()["active_players"][0])
        if previous_active is not None and active != previous_active:
            turn_milliseconds[previous_active].append(current_turn_milliseconds)
            current_turn_milliseconds = 0.0
        start = time.perf_counter()
        selected = (
            replanned_teacher_action(environment)
            if active == replanned_seat
            else native_teacher_action(environment, FOLLOWUP_NODES)
        )
        current_turn_milliseconds += (time.perf_counter() - start) * 1000
        result = environment.step(selected)
        action_counts[active] += 1
        steps += 1
        previous_active = active
    if result is None or previous_active is None:
        raise ValueError("replanning match ended without an action")
    turn_milliseconds[previous_active].append(current_turn_milliseconds)
    return {
        "seed": seed,
        "replanned_seat": replanned_seat,
        "outcome": outcome(result, steps),
        "actions_by_seat": action_counts,
        "turn_milliseconds_by_seat": turn_milliseconds,
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    indexed = {
        (cast(int, record["seed"]), cast(int, record["replanned_seat"])): record
        for record in records
    }
    maps = sorted({cast(int, record["seed"]) for record in records})
    paired_scores = [
        sum(
            finite_score(cast(BranchOutcome, indexed[seed, seat]["outcome"]), seat)
            for seat in (0, 1)
        )
        for seed in maps
    ]
    better = sum(score > 1 for score in paired_scores)
    worse = sum(score < 1 for score in paired_scores)
    turn_times: dict[str, list[float]] = defaultdict(list)
    for record in records:
        replanned = cast(int, record["replanned_seat"])
        per_seat = cast(list[list[float]], record["turn_milliseconds_by_seat"])
        turn_times["replanned"].extend(per_seat[replanned])
        turn_times["cached"].extend(per_seat[1 - replanned])

    def timing(values: list[float]) -> dict[str, float | int]:
        return {
            "turns": len(values),
            "median_milliseconds": float(np.median(values)),
            "p95_milliseconds": float(np.percentile(values, 95)),
        }

    return {
        "independent_maps": len(maps),
        "games": len(records),
        "terminal_games": sum(
            cast(BranchOutcome, record["outcome"])["terminal"] for record in records
        ),
        "action_limit_adjudications": sum(
            cast(BranchOutcome, record["outcome"])["truncated"] for record in records
        ),
        "replanned_wins_by_seat": [
            sum(
                finite_score(cast(BranchOutcome, record["outcome"]), seat) == 1
                for record in records
                if record["replanned_seat"] == seat
            )
            for seat in (0, 1)
        ],
        "cached_wins_by_seat": [
            sum(
                finite_score(cast(BranchOutcome, record["outcome"]), 1 - seat) == 1
                for record in records
                if record["replanned_seat"] == seat
            )
            for seat in (0, 1)
        ],
        "replanned_vs_cached_paired_maps": paired_comparison_summary(
            better, worse, len(maps) - better - worse
        ),
        "turn_timing": {
            "replanned": timing(turn_times["replanned"]),
            "cached": timing(turn_times["cached"]),
        },
    }


def audit() -> dict[str, object]:
    records = [
        play_match(seed, replanned_seat)
        for seed in range(SEED_FIRST, SEED_FIRST + MAPS)
        for replanned_seat in (0, 1)
    ]
    return {
        "kind": "native_three_turn_replanning_vs_cached_direct_head_to_head",
        "protocol": PROTOCOL,
        "seed_first": SEED_FIRST,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "records": records,
        "summary": summarize(records),
        "qualification": "Fresh complete-game direct head-to-head with native CPU timings, not browser runtime, global Elo or trained student strength",
    }


def main() -> None:
    print(json.dumps(audit(), sort_keys=True))


if __name__ == "__main__":
    main()
