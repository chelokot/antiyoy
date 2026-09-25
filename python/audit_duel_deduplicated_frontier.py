from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import numpy as np

from .audit_duel_first_regret import (
    ACTION_LIMIT,
    BranchOutcome,
    create_environment,
    outcome,
)
from .audit_duel_teacher_coverage import finite_score
from .evaluate import paired_comparison_summary, relative_skill_delta


PROTOCOL = (
    "benchmarks/protocols/2026-09-25-duel-deduplicated-frontier-head-to-head-v1.json"
)
SEED_FIRST = 6641000
MAPS = 128
SEARCH = {
    "node_budget": 256,
    "reply_nodes": 64,
    "followup_nodes": 32,
    "slate_size": 8,
    "beam_width": 32,
    "branch_width": 48,
    "maximum_actions_per_turn": 24,
    "replan_each_action": True,
    "score_cache": True,
}


def play(seed: int, candidate_seat: int) -> dict[str, object]:
    environment = create_environment(seed)
    turn_cpu = [0.0, 0.0]
    turn_wall = [0.0, 0.0]
    turns: list[list[list[float]]] = [[], []]
    previous_seat: int | None = None
    result = None
    actions = 0
    while not environment.done()[0]:
        seat = int(environment.observe()["active_players"][0])
        if previous_seat is not None and seat != previous_seat:
            turns[previous_seat].append(
                [turn_cpu[previous_seat], turn_wall[previous_seat]]
            )
            turn_cpu[previous_seat] = 0.0
            turn_wall[previous_seat] = 0.0
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        selected = environment.reply_search_actions(
            **SEARCH, frontier_dedup=seat == candidate_seat
        )
        turn_cpu[seat] += time.process_time() - start_cpu
        turn_wall[seat] += time.perf_counter() - start_wall
        result = environment.step(selected)
        actions += 1
        previous_seat = seat
    if result is None or previous_seat is None:
        raise ValueError(f"seed {seed} ended without an action")
    turns[previous_seat].append([turn_cpu[previous_seat], turn_wall[previous_seat]])
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "outcome": outcome(result, actions),
        "actions": actions,
        "candidate_turns": turns[candidate_seat],
        "baseline_turns": turns[1 - candidate_seat],
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    indexed = {
        (cast(int, record["seed"]), cast(int, record["candidate_seat"])): record
        for record in records
    }
    maps = sorted({cast(int, record["seed"]) for record in records})
    if len(records) != 2 * len(maps) or len(indexed) != len(records):
        raise ValueError("each map must have two unique rotated-seat games")
    censored = sum(
        cast(BranchOutcome, record["outcome"])["truncated"] for record in records
    )
    map_scores = np.array(
        [
            sum(
                finite_score(cast(BranchOutcome, indexed[seed, seat]["outcome"]), seat)
                for seat in (0, 1)
            )
            / 2
            for seed in maps
        ]
    )
    map_signs = paired_comparison_summary(
        int(np.count_nonzero(map_scores > 0.5)),
        int(np.count_nonzero(map_scores < 0.5)),
        int(np.count_nonzero(map_scores == 0.5)),
    )
    candidate_wins_by_seat = [
        sum(
            finite_score(cast(BranchOutcome, indexed[seed, seat]["outcome"]), seat) == 1
            for seed in maps
        )
        for seat in (0, 1)
    ]
    timings = {
        name: np.array(
            [
                turn
                for record in records
                for turn in cast(list[list[float]], record[f"{name}_turns"])
            ]
        )
        for name in ("candidate", "baseline")
    }
    runtime = {
        name: {
            "turns": len(values),
            "median_process_cpu_ms": float(np.median(values[:, 0]) * 1000),
            "median_wall_ms": float(np.median(values[:, 1]) * 1000),
            "p95_wall_ms": float(np.percentile(values[:, 1], 95) * 1000),
        }
        for name, values in timings.items()
    }
    gate = {
        "zero_action_limit_censors": censored == 0,
        "candidate_at_least_64_wins_each_seat": all(
            wins >= 64 for wins in candidate_wins_by_seat
        ),
        "independent_map_sign": (
            map_signs["candidate_better"] > map_signs["baseline_better"]
            and map_signs["exact_two_sided_sign_test_p"] < 0.05
        ),
        "median_process_cpu_at_most_125_percent_baseline": (
            runtime["candidate"]["median_process_cpu_ms"]
            <= 1.25 * runtime["baseline"]["median_process_cpu_ms"]
        ),
        "candidate_p95_wall_below_1000ms": runtime["candidate"]["p95_wall_ms"] < 1000,
    }
    elo = None
    bootstrap = None
    if censored == 0:
        candidate_score = float(map_scores.sum() * 2)
        elo = relative_skill_delta(candidate_score / len(records), len(records), 2)
        random = np.random.default_rng(SEED_FIRST ^ 0xB0057A)
        sampled = map_scores[
            random.integers(0, len(maps), size=(4096, len(maps)))
        ].mean(axis=1)
        clipped = np.clip(sampled, 0.5 / len(records), 1 - 0.5 / len(records))
        bootstrap = np.quantile(
            400 * np.log10(clipped / (1 - clipped)), [0.025, 0.975]
        ).tolist()
    return {
        "independent_maps": len(maps),
        "games": len(records),
        "terminal_games": len(records) - censored,
        "action_limit_censored_games": censored,
        "candidate_wins_by_seat": candidate_wins_by_seat,
        "paired_map_signs": map_signs,
        "runtime": runtime,
        "fixed_pool_head_to_head_elo_candidate_over_baseline": elo,
        "map_bootstrap_95_fixed_pool_elo": bootstrap,
        "gate": gate,
        "advance_gate_passed": all(gate.values()),
    }


def audit() -> dict[str, object]:
    records = [
        play(seed, seat)
        for seed in range(SEED_FIRST, SEED_FIRST + MAPS)
        for seat in (0, 1)
    ]
    return {
        "kind": "native_deduplicated_frontier_vs_unchanged_replanned_search",
        "protocol": PROTOCOL,
        "first_seed": SEED_FIRST,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "records": records,
        "summary": summarize(records),
        "qualification": "Direct independent-map both-seat head-to-head; fixed-pool Elo only when all games are terminal, not global Elo, browser speed or a neural student",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit()
    if arguments.output is None:
        print(json.dumps(report, sort_keys=True))
    else:
        arguments.output.write_text(json.dumps(report, sort_keys=True) + "\n")
        print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
