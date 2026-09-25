from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Mapping, cast

import numpy as np

from .audit_duel_first_regret import ACTION_LIMIT, create_environment, outcome
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-exact-replan-score-cache-v1.json"
SEED_FIRST = 6580000
MAPS = 16
SEARCH_CONFIGURATION = {
    "node_budget": 256,
    "reply_nodes": 64,
    "slate_size": 8,
    "beam_width": 32,
    "branch_width": 48,
    "maximum_actions_per_turn": 24,
    "followup_nodes": FOLLOWUP_NODES,
    "replan_each_action": True,
}


def same_arrays(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> bool:
    return left.keys() == right.keys() and all(
        np.array_equal(values, right[name]) for name, values in left.items()
    )


def percentile95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def replay(seed: int, profile: str = "classic_generic_2022") -> dict[str, object]:
    plain = create_environment(seed, profile)
    cached = create_environment(seed, profile)
    cpu = [[0.0, 0.0], [0.0, 0.0]]
    wall: list[list[float]] = [[], []]
    actions = 0
    plain_result = None
    cached_result = None
    action_digest = hashlib.sha256()
    while not plain.done()[0]:
        plain_observation = plain.observe()
        cached_observation = cached.observe()
        if not same_arrays(plain_observation, cached_observation):
            raise ValueError(f"observation mismatch at seed {seed} action {actions}")
        seat = int(plain_observation["active_players"][0])
        selected = []
        for cache_enabled in ((seed + actions) % 2 == 1, (seed + actions) % 2 == 0):
            environment = cached if cache_enabled else plain
            start_cpu = time.process_time()
            start_wall = time.perf_counter()
            action = environment.reply_search_actions(
                **SEARCH_CONFIGURATION, score_cache=cache_enabled
            )
            elapsed_wall = time.perf_counter() - start_wall
            elapsed_cpu = time.process_time() - start_cpu
            index = int(cache_enabled)
            cpu[index][seat] += elapsed_cpu
            wall[index].append(elapsed_wall)
            selected.append(np.asarray(action, dtype=np.uint64))
        if not np.array_equal(selected[0], selected[1]):
            raise ValueError(f"action mismatch at seed {seed} action {actions}")
        action_digest.update(selected[0].tobytes())
        plain_result = plain.step(selected[0])
        cached_result = cached.step(selected[1])
        if not same_arrays(plain_result, cached_result):
            raise ValueError(f"step mismatch at seed {seed} action {actions}")
        actions += 1
    if plain_result is None or cached_result is None:
        raise ValueError(f"seed {seed} ended without an action")
    if bool(cached.done()[0]) is not True:
        raise ValueError(f"cached seed {seed} did not finish")
    plain_outcome = outcome(plain_result, actions)
    if plain_outcome != outcome(cached_result, actions):
        raise ValueError(f"terminal outcome mismatch at seed {seed}")
    return {
        "seed": seed,
        "actions": actions,
        "action_sha256": action_digest.hexdigest(),
        "outcome": plain_outcome,
        "cache_hits": int(cached.search_cache_hits()[0]),
        "plain_search_cpu_seconds_by_seat": cpu[0],
        "cached_search_cpu_seconds_by_seat": cpu[1],
        "plain_action_wall_seconds": wall[0],
        "cached_action_wall_seconds": wall[1],
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    cpu_improvements = []
    seat_improvements: list[list[float]] = [[], []]
    for record in records:
        plain = cast(list[float], record["plain_search_cpu_seconds_by_seat"])
        cached = cast(list[float], record["cached_search_cpu_seconds_by_seat"])
        cpu_improvements.append(1 - sum(cached) / sum(plain))
        for seat in range(2):
            seat_improvements[seat].append(1 - cached[seat] / plain[seat])
    plain_wall = [
        elapsed
        for record in records
        for elapsed in cast(list[float], record["plain_action_wall_seconds"])
    ]
    cached_wall = [
        elapsed
        for record in records
        for elapsed in cast(list[float], record["cached_action_wall_seconds"])
    ]
    plain_p95 = percentile95(plain_wall)
    cached_p95 = percentile95(cached_wall)
    gate = {
        "all_games_terminal": all(
            cast(dict[str, object], record["outcome"])["terminal"] is True
            and cast(dict[str, object], record["outcome"])["truncated"] is False
            for record in records
        ),
        "exact_observations_actions_and_outcomes": True,
        "cache_reused_a_score": sum(cast(int, record["cache_hits"]) for record in records) > 0,
        "median_map_cpu_improvement_at_least_ten_percent": statistics.median(cpu_improvements)
        >= 0.10,
        "both_seats_median_cpu_nonnegative": all(
            statistics.median(improvements) >= 0 for improvements in seat_improvements
        ),
        "candidate_p95_action_wall_within_ten_percent": cached_p95 <= 1.1 * plain_p95,
    }
    return {
        "games": len(records),
        "actions_compared": sum(cast(int, record["actions"]) for record in records),
        "total_cache_hits": sum(cast(int, record["cache_hits"]) for record in records),
        "median_map_cpu_improvement": statistics.median(cpu_improvements),
        "median_cpu_improvement_by_seat": [
            statistics.median(improvements) for improvements in seat_improvements
        ],
        "action_wall_seconds": {
            "plain_median": statistics.median(plain_wall),
            "cached_median": statistics.median(cached_wall),
            "plain_p95": plain_p95,
            "cached_p95": cached_p95,
        },
        "gate": gate,
        "advance_gate_passed": all(gate.values()),
    }


def audit(
    seed_first: int = SEED_FIRST,
    maps: int = MAPS,
    protocol: str = PROTOCOL,
) -> dict[str, object]:
    records = [replay(seed) for seed in range(seed_first, seed_first + maps)]
    return {
        "kind": "native_replanned_exact_score_cache_cpu_and_equivalence",
        "protocol": protocol,
        "seed_first": seed_first,
        "maps": maps,
        "action_limit": ACTION_LIMIT,
        "search_configuration": SEARCH_CONFIGURATION,
        "records": records,
        "summary": summarize(records),
        "qualification": "Exact paired action equivalence and one-core native CPU only; no policy strength, Elo, or browser latency claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed-first", type=int, default=SEED_FIRST)
    parser.add_argument("--maps", type=int, default=MAPS)
    parser.add_argument("--protocol", default=PROTOCOL)
    arguments = parser.parse_args()
    if arguments.maps <= 0:
        parser.error("--maps must be positive")
    report = audit(arguments.seed_first, arguments.maps, arguments.protocol)
    if arguments.output is None:
        print(json.dumps(report, sort_keys=True))
    else:
        arguments.output.write_text(json.dumps(report, sort_keys=True) + "\n")
        print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
