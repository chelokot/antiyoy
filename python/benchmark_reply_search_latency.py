from __future__ import annotations

import argparse
import json
import time

import numpy as np

from antiyoy_rl import GENERATOR_ROTATED_SCHEMA_VERSION, ProceduralConfig, VectorEnv


def benchmark(
    first_seed: int,
    maps: int,
    width: int = 11,
    height: int = 9,
    action_limit: int = 2400,
    node_budget: int = 256,
    reply_nodes: int = 64,
    slate_size: int = 8,
) -> dict[str, object]:
    if maps < 1:
        raise ValueError("benchmark requires at least one map")
    decision_seconds: list[float] = []
    map_ledger: list[dict[str, float | int | bool]] = []
    for seed in range(first_seed, first_seed + maps):
        generator = ProceduralConfig(
            width=width,
            height=height,
            players=2,
            seed=seed,
            schema_version=GENERATOR_ROTATED_SCHEMA_VERSION,
        )
        environment = VectorEnv.procedural(
            1,
            generator,
            profile="classic_generic_2022",
            action_limit=action_limit,
        )
        environment.reset(0, seed)
        map_decisions: list[float] = []
        actions = 0
        truncated = False
        started = time.perf_counter()
        while not environment.done()[0]:
            prior_counts = environment.search_counts()
            previous = int(prior_counts[0]) if len(prior_counts) else 0
            decision_started = time.perf_counter()
            selected = environment.reply_search_actions(
                node_budget=node_budget,
                reply_nodes=reply_nodes,
                slate_size=slate_size,
            )
            elapsed = time.perf_counter() - decision_started
            current = int(environment.search_counts()[0])
            if current == previous + 1:
                decision_seconds.append(elapsed)
                map_decisions.append(elapsed)
            elif current != previous:
                raise RuntimeError("native search count changed by more than one")
            result = environment.step(selected)
            actions += 1
            truncated = bool(result["truncated"][0])
        map_ledger.append(
            {
                "seed": seed,
                "root_decisions": len(map_decisions),
                "atomic_actions": actions,
                "truncated": truncated,
                "elapsed_wall_seconds": time.perf_counter() - started,
                "median_root_decision_wall_seconds": float(np.median(map_decisions)),
            }
        )
    return {
        "arena": "Classic Generic procedural_v2 two-player",
        "first_seed": first_seed,
        "maps": maps,
        "width": width,
        "height": height,
        "action_limit": action_limit,
        "node_budget": node_budget,
        "reply_nodes": reply_nodes,
        "slate_size": slate_size,
        "root_decisions": len(decision_seconds),
        "median_root_decision_wall_seconds": float(np.median(decision_seconds)),
        "p95_root_decision_wall_seconds": float(np.quantile(decision_seconds, 0.95)),
        "truncated_maps": sum(int(record["truncated"]) for record in map_ledger),
        "map_ledger": map_ledger,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--maps", type=int, required=True)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--action-limit", type=int, default=2400)
    parser.add_argument("--node-budget", type=int, default=256)
    parser.add_argument("--reply-nodes", type=int, default=64)
    parser.add_argument("--slate-size", type=int, default=8)
    arguments = parser.parse_args()
    print(json.dumps(benchmark(**vars(arguments)), sort_keys=True))


if __name__ == "__main__":
    main()
