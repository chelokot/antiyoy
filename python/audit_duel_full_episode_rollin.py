from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from .evaluate import paired_map_comparison, relative_skill_delta, winner_score


FIRST_SEED = 6550000
MAPS = 128
PROTOCOL = (
    "benchmarks/protocols/2026-09-24-duel-full-episode-replan-distillation-v1.json"
)
CHECKPOINTS = {
    "full_episode": "full-episode-replan-fit-6540000.pt",
    "reset_64_control": "reset64-replan-fit-6540000.pt",
}


def audit(
    raw: dict[str, object], arm: str, first_seed: int = FIRST_SEED, maps: int = MAPS
) -> dict[str, object]:
    games = maps * 2
    expected = {
        "checkpoint": CHECKPOINTS[arm],
        "baseline_checkpoint": "routed-v6.pt",
        "baseline": "policy",
        "model_agent": "policy",
        "profile": "classic_generic_2022",
        "generator": "procedural_v2",
        "route_generator": "procedural_v1",
        "players": 2,
        "action_limit": 2400,
        "games": games,
        "seed": first_seed,
    }
    if any(raw[field] != value for field, value in expected.items()):
        raise ValueError("direct-source report differs from the fixed protocol")
    if raw["pairing"] != {
        "first_seed": first_seed,
        "last_seed": first_seed + maps - 1,
        "unique_seeds": maps,
        "scheme": "adjacent_same_seed_opposite_seat_v1",
    }:
        raise ValueError("direct-source map pairing differs from the protocol")
    if raw["generator_config"] != {
        "schema_version": 2,
        "land_density_per_million": 650000,
        "starting_province_size": 5,
        "starting_money": 10,
        "tree_density_per_million": 150000,
        "neutral_tower_density_per_million": 20000,
        "neutral_capital_density_per_million": 10000,
        "grave_density_per_million": 15000,
    }:
        raise ValueError("direct-source generator differs from the protocol")
    domain = cast(dict[str, object], raw["domain_descriptor"])
    if any(
        domain[field] != value
        for field, value in {
            "width": 11,
            "height": 9,
            "players": 2,
            "action_limit": 2400,
            "fog": False,
            "diplomacy": False,
        }.items()
    ):
        raise ValueError("direct-source domain differs from the protocol")
    seeds = cast(list[int], raw["game_seeds"])
    seats = cast(list[int], raw["model_seats"])
    winners = cast(list[int], raw["winners"])
    truncated = cast(list[bool], raw["game_truncated"])
    source = cast(dict[str, object], raw["baseline_self_play"])
    source_winners = cast(list[int], source["winners"])
    source_truncated = cast(list[bool], source["game_truncated"])
    if any(
        len(values) != length
        for values, length in (
            (seeds, games),
            (seats, games),
            (winners, games),
            (truncated, games),
            (source_winners, maps),
            (source_truncated, maps),
        )
    ):
        raise ValueError("direct-source game ledger is incomplete")
    if any(
        seeds[index] != first_seed + index // 2
        or seats[index] != index % 2
        or winners[index] not in (0, 1, 255)
        for index in range(games)
    ) or any(winner not in (0, 1, 255) for winner in source_winners):
        raise ValueError("direct-source map, seat or winner ledger is invalid")
    if (
        raw["wins"]
        != sum(winner == seat for winner, seat in zip(winners, seats, strict=True))
        or raw["draws"] != winners.count(255)
        or raw["losses"] != games - raw["wins"] - raw["draws"]
        or raw["truncations"] != sum(truncated)
        or source["games"] != maps
        or source["truncations"] != sum(source_truncated)
    ):
        raise ValueError("direct-source outcome totals disagree with the ledger")
    candidate_scores = np.asarray(
        [
            winner_score(winner, seat)
            for winner, seat in zip(winners, seats, strict=True)
        ]
    )
    source_scores = np.asarray(
        [
            winner_score(source_winners[index // 2], seat)
            for index, seat in enumerate(seats)
        ]
    )
    paired = paired_map_comparison(candidate_scores, source_scores, 2, None)
    if raw["paired_map_comparison"] != paired:
        raise ValueError("direct-source paired-map summary disagrees with the ledger")
    candidate_by_seat = [
        sum(winners[index] == seat for index in range(seat, games, 2))
        for seat in range(2)
    ]
    source_by_seat = [
        sum(winner == seat for winner in source_winners) for seat in range(2)
    ]
    map_scores = candidate_scores.reshape(maps, 2).mean(axis=1)
    random = np.random.default_rng(first_seed ^ 0xB0057A)
    sampled = map_scores[random.integers(0, maps, size=(4096, maps))].mean(axis=1)
    clipped = np.clip(sampled, 0.5 / games, 1 - 0.5 / games)
    elo_interval = np.quantile(
        400 * np.log10(clipped / (1 - clipped)), [0.025, 0.975]
    ).tolist()
    gate = {
        "all_games_terminal": not any(truncated) and not any(source_truncated),
        "both_seats_improve": all(
            candidate > reference
            for candidate, reference in zip(
                candidate_by_seat, source_by_seat, strict=True
            )
        ),
        "independent_map_sign_p_below_0_05": (
            paired["candidate_better"] > paired["baseline_better"]
            and paired["exact_two_sided_sign_test_p"] < 0.05
        ),
        "bootstrap_lower_elo_positive": elo_interval[0] > 0,
    }
    return {
        "kind": "full_episode_replan_student_direct_source_game_audit",
        "protocol": PROTOCOL,
        "arm": arm,
        "first_seed": first_seed,
        "independent_maps": maps,
        "both_seat_games": games,
        "candidate_wins_by_seat": candidate_by_seat,
        "source_reference_wins_by_seat": source_by_seat,
        "candidate_direct_wins": sum(candidate_by_seat),
        "candidate_nonterminal_games": sum(truncated),
        "source_reference_nonterminal_games": sum(source_truncated),
        "paired_independent_maps": paired,
        "actual_head_to_head_elo_over_source": relative_skill_delta(
            float(candidate_scores.mean()), games, 2
        ),
        "map_bootstrap_95_actual_head_to_head_elo": elo_interval,
        "gate": gate,
        "direct_source_gate_passed": arm == "full_episode" and all(gate.values()),
        "qualification": "Actual fixed-pool direct-game Elo, not global Elo or evidence of beating the native teacher",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(CHECKPOINTS))
    arguments = parser.parse_args()
    with arguments.input.open(encoding="utf-8") as stream:
        raw = cast(dict[str, object], json.load(stream))
    print(json.dumps(audit(raw, arguments.arm), sort_keys=True))


if __name__ == "__main__":
    main()
