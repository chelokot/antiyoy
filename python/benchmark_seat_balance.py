from __future__ import annotations

import argparse
import json
import time
from collections import deque
from collections.abc import Mapping

import numpy as np

from antiyoy_rl import (
    GENERATOR_ROTATED_SCHEMA_VERSION,
    GENERATOR_SCHEMA_VERSION,
    ProceduralConfig,
    VectorEnv,
)


def initial_region_sizes(
    observation: Mapping[str, np.ndarray], environment: int, owner_offset: int = 0
) -> np.ndarray:
    width = int(observation["widths"][environment])
    height = int(observation["heights"][environment])
    players = int(observation["player_counts"][environment])
    cell_start, cell_end = observation["cell_offsets"][environment : environment + 2]
    playable = np.asarray(observation["playable"][cell_start:cell_end], dtype=np.bool_)
    province_start, province_end = observation["province_offsets"][
        environment : environment + 2
    ]
    seeds = np.empty(players, dtype=np.int64)
    for owner, capital in zip(
        observation["province_owners"][province_start:province_end],
        observation["province_capitals"][province_start:province_end],
        strict=True,
    ):
        seeds[int(owner)] = int(capital)
    regions = np.full(width * height, -1, dtype=np.int16)
    queue: deque[int] = deque()
    for position in range(players):
        owner = (position + owner_offset) % players
        regions[seeds[owner]] = owner
        queue.append(int(seeds[owner]))
    directions = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
    while queue:
        hex_index = queue.popleft()
        row, column = divmod(hex_index, width)
        for delta_q, delta_r in directions:
            neighbour_column = column + delta_q
            neighbour_row = row + delta_r
            if not (0 <= neighbour_column < width and 0 <= neighbour_row < height):
                continue
            neighbour = neighbour_row * width + neighbour_column
            if playable[neighbour] and regions[neighbour] == -1:
                regions[neighbour] = regions[hex_index]
                queue.append(neighbour)
    if np.any(regions[playable] == -1):
        raise RuntimeError("generated land is disconnected from its capitals")
    return np.bincount(regions[playable], minlength=players)


def benchmark_seat_balance(
    generator: ProceduralConfig,
    profile: str,
    games: int,
    batch_size: int,
    action_limit: int,
) -> dict[str, object]:
    if games < 1 or batch_size < 1 or action_limit < 1:
        raise ValueError("games, batch size, and action limit must be positive")
    generator_config = json.loads(generator.to_json())
    first_seed = int(generator_config["seed"])
    players = int(generator_config["players"])
    rotated = (
        int(generator_config["schema_version"]) == GENERATOR_ROTATED_SCHEMA_VERSION
    )
    environments = min(games, batch_size)
    environment = VectorEnv.procedural(
        environments, generator, profile=profile, action_limit=action_limit
    )
    for index in range(environments):
        environment.reset(index, first_seed + index)
    finished = np.zeros(environments, dtype=np.bool_)
    game_ids = np.arange(environments, dtype=np.int32)
    wins = np.zeros(players, dtype=np.int64)
    winners = np.full(games, 255, dtype=np.uint8)
    regions = np.empty((games, players), dtype=np.int32)
    observation = environment.observe()
    for index in range(environments):
        owner_offset = (first_seed + index) % players if rotated else 0
        regions[index] = initial_region_sizes(observation, index, owner_offset)
    draws = 0
    truncations = 0
    started = environments
    completed = 0
    transitions = 0
    started_at = time.perf_counter()
    while completed < games:
        actions = np.asarray(environment.greedy_actions(), dtype=np.uint64)
        result = environment.step(actions)
        transitions += environments
        done = np.logical_or(result["terminal"], result["truncated"])
        for index in np.flatnonzero(done):
            if not finished[index]:
                truncated = bool(result["truncated"][index])
                winner_key = "adjudicated_winners" if truncated else "winners"
                winner = int(result[winner_key][index])
                if winner == 255:
                    draws += 1
                else:
                    wins[winner] += 1
                winners[game_ids[index]] = winner
                truncations += int(truncated)
                completed += 1
            if started < games:
                next_seed = first_seed + started
                environment.reset(int(index), next_seed)
                owner_offset = next_seed % players if rotated else 0
                regions[started] = initial_region_sizes(
                    environment.observe(), int(index), owner_offset
                )
                game_ids[index] = started
                started += 1
                finished[index] = False
            else:
                environment.reset(int(index), first_seed + games + int(index))
                finished[index] = True
    winner_region_ranks = np.zeros(players, dtype=np.int64)
    for game, winner in enumerate(winners):
        if winner != 255:
            winner_region_ranks[
                np.count_nonzero(regions[game] > regions[game, winner])
            ] += 1
    return {
        "schema_version": 1,
        "kind": "procedural_greedy_self_play_seat_balance",
        "generator": generator_config,
        "profile": profile,
        "games": games,
        "batch_size": environments,
        "action_limit": action_limit,
        "wins_by_seat": wins.tolist(),
        "winners_by_seed": winners.tolist(),
        "initial_region_sizes_by_seed": regions.tolist(),
        "initial_region_mean_by_seat": regions.mean(axis=0).tolist(),
        "initial_region_median_by_seat": np.median(regions, axis=0).tolist(),
        "maps_tied_for_smallest_region_by_seat": np.count_nonzero(
            regions == regions.min(axis=1, keepdims=True), axis=0
        ).tolist(),
        "winner_region_rank_counts": winner_region_ranks.tolist(),
        "draws": draws,
        "truncations": truncations,
        "environment_transitions": transitions,
        "seconds": time.perf_counter() - started_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="classic_generic_2022")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4_300_000)
    parser.add_argument("--width", type=int, default=19)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument(
        "--generator-schema-version", type=int, default=GENERATOR_SCHEMA_VERSION
    )
    parser.add_argument("--action-limit", type=int, default=2_400)
    parser.add_argument("--land-density-per-million", type=int, default=650_000)
    parser.add_argument("--starting-province-size", type=int, default=5)
    parser.add_argument("--starting-money", type=int, default=10)
    parser.add_argument("--tree-density-per-million", type=int, default=150_000)
    parser.add_argument("--neutral-tower-density-per-million", type=int, default=20_000)
    parser.add_argument(
        "--neutral-capital-density-per-million", type=int, default=10_000
    )
    parser.add_argument("--grave-density-per-million", type=int, default=15_000)
    arguments = parser.parse_args()
    generator = ProceduralConfig(
        width=arguments.width,
        height=arguments.height,
        players=arguments.players,
        seed=arguments.seed,
        land_density_per_million=arguments.land_density_per_million,
        starting_province_size=arguments.starting_province_size,
        starting_money=arguments.starting_money,
        tree_density_per_million=arguments.tree_density_per_million,
        neutral_tower_density_per_million=arguments.neutral_tower_density_per_million,
        neutral_capital_density_per_million=arguments.neutral_capital_density_per_million,
        grave_density_per_million=arguments.grave_density_per_million,
        schema_version=arguments.generator_schema_version,
    )
    report = benchmark_seat_balance(
        generator,
        arguments.profile,
        arguments.games,
        arguments.batch_size,
        arguments.action_limit,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
