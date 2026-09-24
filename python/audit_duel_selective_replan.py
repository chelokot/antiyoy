from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from .evaluate import paired_map_comparison, relative_skill_delta, winner_score


FIRST_SEED = 6530000
MAPS = 128
GAMES = MAPS * 2
PROTOCOL = "benchmarks/protocols/2026-09-24-duel-selective-replan-v1.json"


def audit(raw: dict[str, object]) -> dict[str, object]:
    expected: dict[str, object] = {
        "checkpoint": "replan-markov-student-6500000.pt",
        "baseline_checkpoint": "routed-v6.pt",
        "baseline": "policy",
        "model_agent": "selective_reply_search",
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
        "replan_reply_search": True,
        "games": GAMES,
        "seed": FIRST_SEED,
    }
    if any(raw[field] != value for field, value in expected.items()):
        raise ValueError("selective search report differs from the fixed protocol")
    if raw["pairing"] != {
        "first_seed": FIRST_SEED,
        "last_seed": FIRST_SEED + MAPS - 1,
        "unique_seeds": MAPS,
        "scheme": "adjacent_same_seed_opposite_seat_v1",
    }:
        raise ValueError("selective search map pairing differs from the protocol")
    seeds = cast(list[int], raw["game_seeds"])
    seats = cast(list[int], raw["model_seats"])
    winners = cast(list[int], raw["winners"])
    truncated = cast(list[bool], raw["game_truncated"])
    reference = cast(dict[str, object], raw["baseline_self_play"])
    reference_winners = cast(list[int], reference["winners"])
    reference_truncated = cast(list[bool], reference["game_truncated"])
    if any(
        len(values) != expected_length
        for values, expected_length in (
            (seeds, GAMES),
            (seats, GAMES),
            (winners, GAMES),
            (truncated, GAMES),
            (reference_winners, MAPS),
            (reference_truncated, MAPS),
        )
    ):
        raise ValueError("selective search game ledger is incomplete")
    if any(
        seeds[index] != FIRST_SEED + index // 2
        or seats[index] != index % 2
        or winners[index] not in (0, 1, 255)
        for index in range(GAMES)
    ):
        raise ValueError("selective search game seeds, seats or winners are invalid")
    if (
        any(winner not in (0, 1, 255) for winner in reference_winners)
        or raw["wins"]
        != sum(winner == seat for winner, seat in zip(winners, seats, strict=True))
        or raw["truncations"] != sum(truncated)
    ):
        raise ValueError("selective search outcome totals disagree with ledger")
    candidate_scores = np.asarray(
        [
            winner_score(winner, seat)
            for winner, seat in zip(winners, seats, strict=True)
        ]
    )
    reference_scores = np.asarray(
        [
            winner_score(reference_winners[index // 2], seat)
            for index, seat in enumerate(seats)
        ]
    )
    paired = paired_map_comparison(candidate_scores, reference_scores, 2, None)
    if raw["paired_map_comparison"] != paired:
        raise ValueError("selective search paired-map summary disagrees with ledger")
    direct_wins_by_seat = [
        sum(winners[index] == seat for index in range(seat, GAMES, 2))
        for seat in range(2)
    ]
    source_reference_wins_by_seat = [
        sum(winner == seat for winner in reference_winners) for seat in range(2)
    ]
    queries = cast(dict[str, int | float], raw["selective_reply_search"])
    query_count = cast(int, queries["queries"])
    candidate_decisions = cast(int, queries["candidate_decisions"])
    query_fraction = query_count / candidate_decisions
    policy_actions = cast(dict[str, int | float], raw["model_baseline_policy_actions"])
    final_overrides = cast(int, policy_actions["disagreements"])
    if (
        not 0 <= query_count <= candidate_decisions
        or abs(query_fraction - cast(float, queries["query_fraction"])) > 1e-12
        or policy_actions["decisions"] != candidate_decisions
        or not 0 <= final_overrides <= query_count
    ):
        raise ValueError("selective search query accounting is inconsistent")
    map_scores = candidate_scores.reshape(MAPS, 2).mean(axis=1)
    random = np.random.default_rng(FIRST_SEED ^ 0xB0057A)
    sampled = map_scores[random.integers(0, MAPS, size=(4096, MAPS))].mean(axis=1)
    clipped = np.clip(sampled, 0.5 / GAMES, 1 - 0.5 / GAMES)
    elo_interval = np.quantile(
        400 * np.log10(clipped / (1 - clipped)), [0.025, 0.975]
    ).tolist()
    gate = {
        "all_games_terminal": not any(truncated),
        "candidate_more_than_half_direct_wins": sum(direct_wins_by_seat) > MAPS,
        "positive_source_relative_wins_in_both_seats": all(
            candidate > source
            for candidate, source in zip(
                direct_wins_by_seat, source_reference_wins_by_seat, strict=True
            )
        ),
        "independent_map_sign_p_below_0_05": (
            paired["candidate_better"] > paired["baseline_better"]
            and paired["exact_two_sided_sign_test_p"] < 0.05
        ),
        "teacher_query_fraction_at_most_half": query_fraction <= 0.5,
    }
    return {
        "kind": "selective_replanned_teacher_vs_frozen_source_fresh_game_audit",
        "protocol": PROTOCOL,
        "independent_maps": MAPS,
        "direct_games": GAMES,
        "candidate_direct_wins_by_seat": direct_wins_by_seat,
        "source_direct_wins": GAMES - sum(direct_wins_by_seat) - winners.count(255),
        "source_self_play_reference_wins_by_seat": source_reference_wins_by_seat,
        "paired_independent_maps": paired,
        "candidate_direct_head_to_head_elo": relative_skill_delta(
            float(candidate_scores.mean()), GAMES, 2
        ),
        "map_bootstrap_95_direct_head_to_head_elo": elo_interval,
        "candidate_nonterminal_games": sum(truncated),
        "source_reference_nonterminal_games": sum(reference_truncated),
        "teacher_queries": query_count,
        "final_actions_different_from_source": final_overrides,
        "candidate_decisions": candidate_decisions,
        "teacher_query_fraction": query_fraction,
        "gate": gate,
        "exploratory_gate_passed": all(gate.values()),
        "qualification": "Search-assisted controller in one fixed pool, not a standalone neural student or global Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.input.open(encoding="utf-8") as stream:
        raw = cast(dict[str, object], json.load(stream))
    print(json.dumps(audit(raw), sort_keys=True))


if __name__ == "__main__":
    main()
