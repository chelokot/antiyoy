from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

try:
    from .evaluate import (
        PAIRING_SCHEME,
        SEAT_ROTATION_SCHEME,
        evaluate,
        relative_skill_delta,
    )
except ImportError:
    from evaluate import (
        PAIRING_SCHEME,
        SEAT_ROTATION_SCHEME,
        evaluate,
        relative_skill_delta,
    )


ALL_PROFILES = (
    "classic_generic_2022",
    "classic_slay_2022",
    "online_default_v1",
    "online_classic_v1",
    "online_duel_v1",
    "online_experimental_v1",
    "online_experimental_v2_260801",
)


def aggregate_outcomes(
    results: list[dict[str, object]], seat: int | None = None
) -> dict[str, float | int]:
    outcomes = (
        results
        if seat is None
        else [result["seats"][seat] for result in results]
    )
    games = sum(int(result["games"]) for result in outcomes)
    wins = sum(int(result["wins"]) for result in outcomes)
    draws = sum(int(result["draws"]) for result in outcomes)
    losses = sum(int(result["losses"]) for result in outcomes)
    truncations = sum(int(result["truncations"]) for result in outcomes)
    adjudications = sum(int(result.get("adjudications", 0)) for result in outcomes)
    terminal_draws = sum(int(result["terminal_draws"]) for result in outcomes)
    score = (wins + 0.5 * draws) / games
    players = int(results[0].get("players", 2))
    if any(int(result.get("players", 2)) != players for result in results):
        raise ValueError("cannot aggregate evaluations with different player counts")
    aggregate = {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "truncations": truncations,
        "adjudications": adjudications,
        "terminal_draws": terminal_draws,
        "score": score,
        "relative_elo": relative_skill_delta(score, games, players),
    }
    reference_outcomes = ["baseline_wins" in result for result in outcomes]
    if any(reference_outcomes) and not all(reference_outcomes):
        raise ValueError("cannot mix seat-calibrated and uncalibrated evaluations")
    if all(reference_outcomes):
        baseline_wins = sum(int(result["baseline_wins"]) for result in outcomes)
        baseline_draws = sum(int(result["baseline_draws"]) for result in outcomes)
        baseline_truncations = sum(
            int(result["baseline_truncations"]) for result in outcomes
        )
        baseline_score = (baseline_wins + 0.5 * baseline_draws) / games
        baseline_truncation_rate = baseline_truncations / games
        aggregate.update(
            {
                "baseline_wins": baseline_wins,
                "baseline_draws": baseline_draws,
                "baseline_truncations": baseline_truncations,
                "baseline_score": baseline_score,
                "score_delta": score - baseline_score,
                "baseline_truncation_rate": baseline_truncation_rate,
                "truncation_rate_delta": (
                    truncations / games - baseline_truncation_rate
                ),
            }
        )
    return aggregate


def aggregate_results(results: list[dict[str, object]]) -> dict[str, object]:
    seats = len(results[0]["seats"])
    if any(len(result["seats"]) != seats for result in results):
        raise ValueError("cannot aggregate evaluations with different seat counts")
    return {
        **aggregate_outcomes(results),
        "seats": [
            {"seat": seat, **aggregate_outcomes(results, seat)} for seat in range(seats)
        ],
    }


def minimum_seat_slice(results: list[dict[str, object]]) -> dict[str, object]:
    slices = [
        {
            "profile": result["profile"],
            "seed": result["seed"],
            **seat_result,
        }
        for result in results
        for seat_result in result["seats"]
    ]
    metric = "score_delta" if "score_delta" in slices[0] else "score"
    return min(slices, key=lambda result: float(result[metric]))


def minimum_profile_seat_slice(
    results: list[dict[str, object]],
) -> dict[str, object]:
    slices = []
    profiles = dict.fromkeys(str(result["profile"]) for result in results)
    for profile in profiles:
        profile_results = [
            result for result in results if result["profile"] == profile
        ]
        seats = len(profile_results[0]["seats"])
        for seat in range(seats):
            slices.append(
                {
                    "profile": profile,
                    "seat": seat,
                    "seed_windows": [int(result["seed"]) for result in profile_results],
                    **aggregate_outcomes(profile_results, seat),
                }
            )
    metric = "score_delta" if "score_delta" in slices[0] else "score"
    return min(slices, key=lambda result: float(result[metric]))


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while block := checkpoint.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100000, 130000])
    parser.add_argument("--profiles", nargs="+", choices=ALL_PROFILES, default=ALL_PROFILES)
    parser.add_argument(
        "--baseline", choices=("search", "greedy", "random"), default="search"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--search-nodes", type=int, default=2048)
    parser.add_argument("--search-beam-width", type=int, default=32)
    parser.add_argument("--search-branch-width", type=int, default=48)
    parser.add_argument("--search-maximum-actions-per-turn", type=int, default=24)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--action-limit", type=int, default=1000)
    parser.add_argument("--procedural", action="store_true")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--land-density-per-million", type=int, default=650_000)
    parser.add_argument("--starting-province-size", type=int, default=5)
    parser.add_argument("--starting-money", type=int, default=10)
    parser.add_argument("--tree-density-per-million", type=int, default=150_000)
    parser.add_argument("--neutral-tower-density-per-million", type=int, default=20_000)
    parser.add_argument("--neutral-capital-density-per-million", type=int, default=10_000)
    parser.add_argument("--grave-density-per-million", type=int, default=15_000)
    parser.add_argument("--minimum-aggregate-score", type=float)
    parser.add_argument("--minimum-seat-score", type=float)
    parser.add_argument("--minimum-seat-score-delta", type=float)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.players < 2:
        parser.error("players must be at least two")
    if arguments.games < arguments.players or arguments.games % arguments.players != 0:
        parser.error("games must be a positive multiple of players")
    if not arguments.procedural and arguments.players != 2:
        parser.error("symmetric duel evaluation requires exactly two players")
    if arguments.minimum_aggregate_score is not None and not (
        0 <= arguments.minimum_aggregate_score <= 1
    ):
        parser.error("minimum aggregate score must be between zero and one")
    if arguments.minimum_seat_score is not None and not (
        0 <= arguments.minimum_seat_score <= 1
    ):
        parser.error("minimum seat score must be between zero and one")
    if arguments.minimum_seat_score_delta is not None and not (
        -1 <= arguments.minimum_seat_score_delta <= 1
    ):
        parser.error("minimum seat score delta must be between minus one and one")

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    for profile in arguments.profiles:
        for seed in arguments.seeds:
            print(f"evaluating {profile} seed={seed}", file=sys.stderr, flush=True)
            results.append(
                evaluate(
                    arguments.checkpoint,
                    arguments.games,
                    seed,
                    arguments.device,
                    arguments.baseline,
                    profile,
                    arguments.search_nodes,
                    arguments.search_beam_width,
                    arguments.search_branch_width,
                    arguments.search_maximum_actions_per_turn,
                    arguments.width,
                    arguments.height,
                    arguments.action_limit,
                    arguments.procedural,
                    arguments.players,
                    arguments.land_density_per_million,
                    arguments.starting_province_size,
                    arguments.starting_money,
                    arguments.tree_density_per_million,
                    arguments.neutral_tower_density_per_million,
                    arguments.neutral_capital_density_per_million,
                    arguments.grave_density_per_million,
                )
            )
    aggregate = aggregate_results(results)
    weakest_seed_seat = minimum_seat_slice(results)
    weakest_seat = minimum_profile_seat_slice(results)
    result: dict[str, object] = {
        "schema_version": 3,
        "kind": "universal_policy_seed_sweep",
        "checkpoint": {
            "path": str(arguments.checkpoint),
            "sha256": checkpoint_digest(arguments.checkpoint),
            "size_bytes": arguments.checkpoint.stat().st_size,
        },
        "baseline": arguments.baseline,
        "search": {
            "nodes": arguments.search_nodes if arguments.baseline == "search" else 0,
            "beam_width": (
                arguments.search_beam_width if arguments.baseline == "search" else 0
            ),
            "branch_width": (
                arguments.search_branch_width if arguments.baseline == "search" else 0
            ),
            "maximum_actions_per_turn": (
                arguments.search_maximum_actions_per_turn
                if arguments.baseline == "search"
                else 0
            ),
        },
        "arena": {
            "generator": (
                "procedural_v1" if arguments.procedural else "symmetric_duel_v1"
            ),
            "width": arguments.width,
            "height": arguments.height,
            "players": arguments.players,
            "action_limit": arguments.action_limit,
            "games_per_profile_seed": arguments.games,
            "unique_maps_per_profile_seed": arguments.games // arguments.players,
            "pairing": (
                PAIRING_SCHEME if arguments.players == 2 else SEAT_ROTATION_SCHEME
            ),
            "profiles": arguments.profiles,
            "seeds": arguments.seeds,
            "generator_config": (
                {
                    "land_density_per_million": arguments.land_density_per_million,
                    "starting_province_size": arguments.starting_province_size,
                    "starting_money": arguments.starting_money,
                    "tree_density_per_million": arguments.tree_density_per_million,
                    "neutral_tower_density_per_million": arguments.neutral_tower_density_per_million,
                    "neutral_capital_density_per_million": arguments.neutral_capital_density_per_million,
                    "grave_density_per_million": arguments.grave_density_per_million,
                }
                if arguments.procedural
                else None
            ),
        },
        "device": arguments.device,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
        "aggregate": aggregate,
        "weakest_seat": weakest_seat,
        "weakest_seed_seat": weakest_seed_seat,
    }
    serialized = json.dumps(result, sort_keys=True)
    print(serialized)
    if arguments.output is not None:
        write_result(arguments.output, result)
    minimum_score = arguments.minimum_aggregate_score
    if minimum_score is not None and float(aggregate["score"]) < minimum_score:
        raise SystemExit(
            f"aggregate score {aggregate['score']:.6f} is below required {minimum_score:.6f}"
        )
    minimum_seat_score = arguments.minimum_seat_score
    if minimum_seat_score is not None and float(weakest_seat["score"]) < minimum_seat_score:
        raise SystemExit(
            f"weakest seat score {weakest_seat['score']:.6f} is below required "
            f"{minimum_seat_score:.6f}"
        )
    minimum_seat_score_delta = arguments.minimum_seat_score_delta
    if minimum_seat_score_delta is not None and float(
        weakest_seat["score_delta"]
    ) < minimum_seat_score_delta:
        raise SystemExit(
            f"weakest seat score delta {weakest_seat['score_delta']:.6f} is below required "
            f"{minimum_seat_score_delta:.6f}"
        )


if __name__ == "__main__":
    main()
