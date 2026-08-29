from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from antiyoy_rl import ProceduralConfig, VectorEnv


PROFILES = [
    "classic_generic_2022",
    "classic_slay_2022",
    "online_default_v1",
    "online_classic_v1",
    "online_duel_v1",
    "online_experimental_v1",
    "online_experimental_v2_260801",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    environments: int
    transitions: int
    procedural: bool
    width: int
    height: int
    players: int
    seed: int
    land_density_per_million: int
    action_limit: int
    profiles: list[str]
    search_nodes: int
    search_beam_width: int
    search_branch_width: int
    search_maximum_actions_per_turn: int


def make_environment(config: BenchmarkConfig) -> VectorEnv:
    schedule = [
        config.profiles[index % len(config.profiles)]
        for index in range(config.environments)
    ]
    if not config.procedural:
        return VectorEnv.mixed(
            schedule,
            width=config.width,
            height=config.height,
            seed=config.seed,
            action_limit=config.action_limit,
        )
    generator = ProceduralConfig(
        width=config.width,
        height=config.height,
        players=config.players,
        seed=config.seed,
        land_density_per_million=config.land_density_per_million,
    )
    return VectorEnv.procedural_mixed(
        schedule,
        generator,
        action_limit=config.action_limit,
    )


def benchmark(config: BenchmarkConfig) -> dict[str, float | int | str | list[str]]:
    if config.environments < 1 or config.transitions < 1:
        raise ValueError("environments and transitions must be positive")
    if not config.profiles:
        raise ValueError("profiles must not be empty")
    setup_started = time.perf_counter()
    environment = make_environment(config)
    setup_seconds = time.perf_counter() - setup_started
    batches = (config.transitions + config.environments - 1) // config.environments
    reset_seed = config.seed + config.environments
    search_seconds = 0.0
    step_seconds = 0.0
    checksum = 0
    started = time.perf_counter()
    for batch in range(batches):
        search_started = time.perf_counter()
        actions = np.asarray(
            environment.search_actions(
                node_budget=config.search_nodes,
                beam_width=config.search_beam_width,
                branch_width=config.search_branch_width,
                maximum_actions_per_turn=config.search_maximum_actions_per_turn,
            ),
            dtype=np.uint64,
        )
        search_seconds += time.perf_counter() - search_started
        step_started = time.perf_counter()
        result = environment.step(actions)
        done = np.logical_or(result["terminal"], result["truncated"])
        for index in np.flatnonzero(done):
            environment.reset(int(index), reset_seed)
            reset_seed += 1
        step_seconds += time.perf_counter() - step_started
        checksum = (
            checksum * 0x9E3779B185EBCA87
            + int(actions.sum(dtype=np.uint64))
            + batch
        ) & 0xFFFFFFFFFFFFFFFF
    elapsed_seconds = time.perf_counter() - started
    executed_transitions = batches * config.environments
    return {
        "map": "procedural_v1" if config.procedural else "symmetric_duel_v1",
        "profiles": config.profiles,
        "environments": config.environments,
        "requested_transitions": config.transitions,
        "executed_transitions": executed_transitions,
        "search_nodes": config.search_nodes,
        "setup_seconds": setup_seconds,
        "search_seconds": search_seconds,
        "step_seconds": step_seconds,
        "elapsed_seconds": elapsed_seconds,
        "teacher_transitions_per_second": executed_transitions / elapsed_seconds,
        "checksum": checksum,
    }


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--transitions", type=int, default=10_000)
    parser.add_argument("--procedural", action="store_true")
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--land-density-per-million", type=int, default=650_000)
    parser.add_argument("--action-limit", type=int, default=1000)
    parser.add_argument("--profiles", nargs="+", default=PROFILES)
    parser.add_argument("--search-nodes", type=int, default=2048)
    parser.add_argument("--search-beam-width", type=int, default=32)
    parser.add_argument("--search-branch-width", type=int, default=48)
    parser.add_argument("--search-maximum-actions-per-turn", type=int, default=24)
    return BenchmarkConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(asdict(configuration), sort_keys=True), flush=True)
    print(json.dumps(benchmark(configuration), sort_keys=True), flush=True)
