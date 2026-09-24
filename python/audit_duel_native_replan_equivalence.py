from __future__ import annotations

import json
import time
from typing import Callable, cast

import numpy as np

from .audit_duel_first_regret import (
    ACTION_LIMIT,
    BranchOutcome,
    create_environment,
    native_teacher_action,
    outcome,
)
from .audit_duel_teacher_plan_cache import replanned_teacher_action
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-native-replan-equivalence-v1.json"
SEED_FIRST = 6493000
MAPS = 16


def timed_selection(selection: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    action = selection()
    return action, (time.perf_counter() - start) * 1000


def replay_pair(seed: int, replanned_seat: int) -> dict[str, object]:
    fork_environment = create_environment(seed)
    native_environment = create_environment(seed)

    def fork_selection() -> np.ndarray:
        return replanned_teacher_action(fork_environment)

    def native_selection() -> np.ndarray:
        return native_teacher_action(
            native_environment, FOLLOWUP_NODES, replan_each_action=True
        )

    previous_active: int | None = None
    fork_turn_milliseconds: list[float] = []
    native_turn_milliseconds: list[float] = []
    current_fork_milliseconds = 0.0
    current_native_milliseconds = 0.0
    steps = 0
    fork_result = None
    native_result = None
    while not fork_environment.done()[0]:
        fork_observation = fork_environment.observe()
        native_observation = native_environment.observe()
        for name, values in fork_observation.items():
            np.testing.assert_array_equal(values, native_observation[name])
        active = int(fork_observation["active_players"][0])
        if previous_active == replanned_seat and active != replanned_seat:
            fork_turn_milliseconds.append(current_fork_milliseconds)
            native_turn_milliseconds.append(current_native_milliseconds)
            current_fork_milliseconds = 0.0
            current_native_milliseconds = 0.0
        if active == replanned_seat:
            if (seed + replanned_seat + steps) % 2 == 0:
                fork_action, fork_milliseconds = timed_selection(fork_selection)
                native_action, native_milliseconds = timed_selection(native_selection)
            else:
                native_action, native_milliseconds = timed_selection(native_selection)
                fork_action, fork_milliseconds = timed_selection(fork_selection)
            current_fork_milliseconds += fork_milliseconds
            current_native_milliseconds += native_milliseconds
        else:
            fork_action = native_teacher_action(fork_environment, FOLLOWUP_NODES)
            native_action = native_teacher_action(native_environment, FOLLOWUP_NODES)
        np.testing.assert_array_equal(fork_action, native_action)
        fork_result = fork_environment.step(fork_action)
        native_result = native_environment.step(native_action)
        previous_active = active
        steps += 1
    if previous_active == replanned_seat:
        fork_turn_milliseconds.append(current_fork_milliseconds)
        native_turn_milliseconds.append(current_native_milliseconds)
    if fork_result is None or native_result is None:
        raise ValueError("paired replay ended without an action")
    fork_outcome = outcome(fork_result, steps)
    if fork_outcome != outcome(native_result, steps):
        raise ValueError("native replanning changed the terminal outcome")
    return {
        "seed": seed,
        "replanned_seat": replanned_seat,
        "outcome": fork_outcome,
        "actions_compared": steps,
        "fork_turn_milliseconds": fork_turn_milliseconds,
        "native_turn_milliseconds": native_turn_milliseconds,
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    fork_times = [
        timing
        for record in records
        for timing in cast(list[float], record["fork_turn_milliseconds"])
    ]
    native_times = [
        timing
        for record in records
        for timing in cast(list[float], record["native_turn_milliseconds"])
    ]
    return {
        "independent_maps": len({cast(int, record["seed"]) for record in records}),
        "games": len(records),
        "actions_compared": sum(
            cast(int, record["actions_compared"]) for record in records
        ),
        "terminal_games": sum(
            cast(BranchOutcome, record["outcome"])["terminal"] for record in records
        ),
        "action_limit_adjudications": sum(
            cast(BranchOutcome, record["outcome"])["truncated"] for record in records
        ),
        "fork_full_turn_milliseconds": {
            "turns": len(fork_times),
            "median": float(np.median(fork_times)),
            "p95": float(np.percentile(fork_times, 95)),
        },
        "native_full_turn_milliseconds": {
            "turns": len(native_times),
            "median": float(np.median(native_times)),
            "p95": float(np.percentile(native_times, 95)),
        },
    }


def audit() -> dict[str, object]:
    records = [
        replay_pair(seed, replanned_seat)
        for seed in range(SEED_FIRST, SEED_FIRST + MAPS)
        for replanned_seat in (0, 1)
    ]
    return {
        "kind": "native_replanning_exact_whole_game_equivalence",
        "protocol": PROTOCOL,
        "seed_first": SEED_FIRST,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "records": records,
        "summary": summarize(records),
        "qualification": "Exact paired state, action and terminal-outcome equivalence; timing is native CPU, not browser runtime or strength",
    }


def main() -> None:
    print(json.dumps(audit(), sort_keys=True))


if __name__ == "__main__":
    main()
