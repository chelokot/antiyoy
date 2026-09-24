from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from .evaluate import (
    paired_map_comparison,
    relative_skill_delta,
    winner_score,
)


class Pairing(TypedDict):
    first_seed: int
    last_seed: int
    unique_seeds: int
    scheme: str


class DirectReport(TypedDict):
    checkpoint: str
    baseline: str
    model_agent: str
    profile: str
    generator: str
    route_generator: str
    players: int
    action_limit: int
    search_nodes: int
    search_beam_width: int
    search_branch_width: int
    search_maximum_actions_per_turn: int
    reply_search_nodes: int
    reply_slate_size: int
    followup_search_nodes: int
    games: int
    seed: int
    game_seeds: list[int]
    model_seats: list[int]
    winners: list[int]
    game_truncated: list[bool]
    wins: int
    losses: int
    draws: int
    truncations: int
    baseline_self_play: dict[str, object]
    pairing: Pairing


def audit(
    raw: DirectReport, expected_seed: int, expected_maps: int
) -> dict[str, object]:
    games = expected_maps * 2
    expected_configuration: dict[str, object] = {
        "checkpoint": "routed-v6.pt",
        "baseline": "reply_search",
        "model_agent": "policy",
        "profile": "classic_generic_2022",
        "generator": "procedural_v2",
        "route_generator": "procedural_v1",
        "players": 2,
        "action_limit": 2400,
        "search_nodes": 256,
        "search_beam_width": 32,
        "search_branch_width": 48,
        "search_maximum_actions_per_turn": 24,
        "reply_search_nodes": 64,
        "reply_slate_size": 8,
        "followup_search_nodes": 32,
        "games": games,
        "seed": expected_seed,
    }
    if any(
        raw[field] != expected for field, expected in expected_configuration.items()
    ):
        raise ValueError("direct match differs from the frozen protocol")
    if raw["pairing"] != {
        "first_seed": expected_seed,
        "last_seed": expected_seed + expected_maps - 1,
        "unique_seeds": expected_maps,
        "scheme": "adjacent_same_seed_opposite_seat_v1",
    }:
        raise ValueError("direct match pairing differs from the protocol")
    if (
        len(raw["game_seeds"]) != games
        or len(raw["model_seats"]) != games
        or len(raw["winners"]) != games
        or len(raw["game_truncated"]) != games
    ):
        raise ValueError("direct match does not cover every map and seat")
    for index in range(games):
        if (
            raw["game_seeds"][index] != expected_seed + index // 2
            or raw["model_seats"][index] != index % 2
            or raw["winners"][index] not in (0, 1, 255)
        ):
            raise ValueError("direct match map, seat or winner ledger is invalid")
    if (
        raw["wins"]
        != sum(
            winner == seat
            for winner, seat in zip(raw["winners"], raw["model_seats"], strict=True)
        )
        or raw["draws"] != raw["winners"].count(255)
        or raw["losses"] != games - raw["wins"] - raw["draws"]
        or raw["truncations"] != sum(raw["game_truncated"])
    ):
        raise ValueError("direct match totals disagree with the game ledger")

    native_scores = np.asarray(
        [
            winner_score(winner, 1 - model_seat)
            for winner, model_seat in zip(
                raw["winners"], raw["model_seats"], strict=True
            )
        ]
    )
    model_scores = 1 - native_scores
    truncated = np.asarray(raw["game_truncated"], dtype=np.bool_)
    complete_maps = ~truncated.reshape(-1, 2).any(axis=1)
    worst_native_scores = np.where(truncated, 0.0, native_scores)
    worst_model_scores = 1 - worst_native_scores
    paired_maps = paired_map_comparison(native_scores, model_scores, 2, None)
    worst_maps = paired_map_comparison(worst_native_scores, worst_model_scores, 2, None)
    native_map_scores = native_scores.reshape(-1, 2).mean(axis=1)
    random = np.random.default_rng(expected_seed ^ 0xB0057A)
    sampled_means = native_map_scores[
        random.integers(0, expected_maps, size=(4_096, expected_maps))
    ].mean(axis=1)
    clipped_means = np.clip(sampled_means, 0.5 / games, 1 - 0.5 / games)
    head_to_head_interval = np.quantile(
        400 * np.log10(clipped_means / (1 - clipped_means)), [0.025, 0.975]
    ).tolist()
    native_wins_by_seat = [
        sum(raw["winners"][index] == 1 - seat for index in range(seat, games, 2))
        for seat in range(2)
    ]
    model_wins_by_seat = [
        sum(raw["winners"][index] == seat for index in range(seat, games, 2))
        for seat in range(2)
    ]
    native_mean = float(native_scores.mean())
    model_mean = float(model_scores.mean())
    gate = {
        "more_native_wins": sum(native_wins_by_seat) > sum(model_wins_by_seat),
        "both_seats_native_nonnegative": all(
            native >= model
            for native, model in zip(
                native_wins_by_seat, model_wins_by_seat, strict=True
            )
        ),
        "independent_map_sign": (
            paired_maps["candidate_better"] > paired_maps["baseline_better"]
            and paired_maps["exact_two_sided_sign_test_p"] < 0.05
        ),
        "bootstrap_lower_elo_positive": head_to_head_interval[0] > 0,
        "nonterminal_rate_at_most_one_percent": int(truncated.sum()) <= games * 0.01,
        "worst_case_timeout_sign": (
            worst_maps["candidate_better"] > worst_maps["baseline_better"]
            and worst_maps["exact_two_sided_sign_test_p"] < 0.05
        ),
    }
    return {
        "kind": "three_turn_vs_frozen_direct_model_fresh_matched_audit",
        "protocol": "benchmarks/protocols/2026-09-24-duel-three-turn-finite-horizon-v1.json",
        "seed_first": expected_seed,
        "independent_maps": expected_maps,
        "rotated_seat_games": games,
        "native_wins_by_model_seat": native_wins_by_seat,
        "direct_model_wins_by_model_seat": model_wins_by_seat,
        "native_finite_horizon_score": native_mean,
        "direct_model_finite_horizon_score": model_mean,
        "head_to_head_elo_native_over_direct": relative_skill_delta(
            native_mean, games, 2
        ),
        "map_bootstrap_95_head_to_head_elo": head_to_head_interval,
        "paired_independent_maps": paired_maps,
        "fully_terminal_maps": int(complete_maps.sum()),
        "fully_terminal_paired_maps": (
            paired_map_comparison(
                native_scores.reshape(-1, 2)[complete_maps].reshape(-1),
                model_scores.reshape(-1, 2)[complete_maps].reshape(-1),
                2,
                None,
            )
            if complete_maps.any()
            else None
        ),
        "nonterminal_games": int(truncated.sum()),
        "nonterminal_ledger": [
            {
                "seed": raw["game_seeds"][index],
                "model_seat": raw["model_seats"][index],
                "adjudicated_winner": raw["winners"][index],
            }
            for index in np.flatnonzero(truncated)
        ],
        "worst_case_timeout_paired_maps": worst_maps,
        "native_self_play_reference_nonterminal_games": raw["baseline_self_play"][
            "truncations"
        ],
        "gate": gate,
        "direct_model_gate_passed": all(gate.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.input.open(encoding="utf-8") as stream:
        raw = cast(DirectReport, json.load(stream))
    print(json.dumps(audit(raw, 6_413_000, 256), sort_keys=True))


if __name__ == "__main__":
    main()
