from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from .evaluate import (
    baseline_adjusted_elo_delta,
    paired_map_bootstrap_interval,
    paired_map_comparison,
    winner_score,
)


FIRST_SEED = 6531000
MAPS = 64
GAMES = MAPS * 2
PROTOCOL = "benchmarks/protocols/2026-09-24-duel-selective-replan-confirmation-v1.json"


def audit(candidate: dict[str, object], source: dict[str, object]) -> dict[str, object]:
    shared = {
        "baseline": "reply_search",
        "profile": "classic_generic_2022",
        "generator": "procedural_v2",
        "route_generator": "procedural_v1",
        "players": 2,
        "arena_width": 11,
        "arena_height": 9,
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
        "pairing": {
            "first_seed": FIRST_SEED,
            "last_seed": FIRST_SEED + MAPS - 1,
            "unique_seeds": MAPS,
            "scheme": "adjacent_same_seed_opposite_seat_v1",
        },
    }
    for name, raw, agent in (
        ("candidate", candidate, "selective_reply_search"),
        ("source", source, "policy"),
    ):
        if any(raw.get(field) != value for field, value in shared.items()):
            raise ValueError(f"{name} arena differs from the confirmation protocol")
        if raw.get("model_agent") != agent:
            raise ValueError(f"{name} policy differs from the confirmation protocol")
        if raw.get("baseline_checkpoint") is not None:
            raise ValueError(f"{name} must face the full search baseline")
    if candidate.get("checkpoint") != "replan-markov-student-6500000.pt":
        raise ValueError("candidate checkpoint differs from the frozen student")
    if candidate.get("selective_source_checkpoint") != "routed-v6.pt":
        raise ValueError("candidate source checkpoint differs from the frozen source")
    if source.get("checkpoint") != "routed-v6.pt":
        raise ValueError("source checkpoint differs from the frozen source")
    if candidate.get("generator_config") != source.get("generator_config"):
        raise ValueError("candidate and source map generators differ")
    if candidate.get("baseline_self_play") != source.get("baseline_self_play"):
        raise ValueError("candidate and source opponent self-play ledgers differ")

    scores = []
    wins_by_seat = []
    nonterminal_games = []
    for name, raw in (("candidate", candidate), ("source", source)):
        seeds = cast(list[int], raw["game_seeds"])
        seats = cast(list[int], raw["model_seats"])
        winners = cast(list[int], raw["winners"])
        truncated = cast(list[bool], raw["game_truncated"])
        if any(len(values) != GAMES for values in (seeds, seats, winners, truncated)):
            raise ValueError(f"{name} game ledger is incomplete")
        if any(
            seeds[index] != FIRST_SEED + index // 2
            or seats[index] != index % 2
            or winners[index] not in (0, 1, 255)
            for index in range(GAMES)
        ):
            raise ValueError(f"{name} game ledger is mispaired")
        if raw["wins"] != sum(
            winner == seat for winner, seat in zip(winners, seats, strict=True)
        ) or raw["truncations"] != sum(truncated):
            raise ValueError(f"{name} outcome totals disagree with game ledger")
        nonterminal_games.append(sum(truncated))
        scores.append(
            np.asarray(
                [
                    winner_score(winner, seat)
                    for winner, seat in zip(winners, seats, strict=True)
                ]
            )
        )
        wins_by_seat.append(
            [
                sum(winners[index] == seat for index in range(seat, GAMES, 2))
                for seat in range(2)
            ]
        )
    candidate_scores, source_scores = scores
    paired = paired_map_comparison(candidate_scores, source_scores, 2, None)
    bootstrap = paired_map_bootstrap_interval(
        candidate_scores, source_scores, 2, None, FIRST_SEED
    )
    queries = cast(dict[str, int | float], candidate["selective_reply_search"])
    query_count = cast(int, queries["queries"])
    decisions = cast(int, queries["candidate_decisions"])
    overrides = cast(int, queries["final_actions_different_from_source"])
    if (
        decisions < 1
        or not 0 <= overrides <= query_count <= decisions
        or abs(cast(float, queries["query_fraction"]) - query_count / decisions) > 1e-12
    ):
        raise ValueError("selective search query accounting is inconsistent")
    gate = {
        "all_256_games_terminal": sum(nonterminal_games) == 0,
        "both_seats_positive": all(
            candidate_wins > source_wins
            for candidate_wins, source_wins in zip(
                wins_by_seat[0], wins_by_seat[1], strict=True
            )
        ),
        "map_sign_p_below_0_05": (
            paired["candidate_better"] > paired["baseline_better"]
            and paired["exact_two_sided_sign_test_p"] < 0.05
        ),
        "bootstrap_lower_elo_positive": bootstrap["baseline_adjusted_elo_delta"][0] > 0,
        "teacher_query_fraction_at_most_half": query_count / decisions <= 0.5,
    }
    return {
        "kind": "selective_replan_vs_direct_under_same_full_teacher_audit",
        "protocol": PROTOCOL,
        "maps": MAPS,
        "terminal_games_per_controller": GAMES,
        "candidate_wins_by_seat": wins_by_seat[0],
        "source_wins_by_seat": wins_by_seat[1],
        "candidate_nonterminal_games": nonterminal_games[0],
        "source_nonterminal_games": nonterminal_games[1],
        "paired_independent_maps": paired,
        "pool_relative_elo_delta": baseline_adjusted_elo_delta(
            float(candidate_scores.mean()), float(source_scores.mean()), GAMES
        ),
        "map_bootstrap_95_pool_relative_elo_delta": bootstrap[
            "baseline_adjusted_elo_delta"
        ],
        "teacher_queries": query_count,
        "candidate_decisions": decisions,
        "teacher_query_fraction": query_count / decisions,
        "final_actions_different_from_source": overrides,
        "gate": gate,
        "confirmation_gate_passed": all(gate.values()),
        "qualification": "Separate complete games against one frozen full-search opponent, not direct head-to-head or global Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.candidate.open(encoding="utf-8") as stream:
        candidate = cast(dict[str, object], json.load(stream))
    with arguments.source.open(encoding="utf-8") as stream:
        source = cast(dict[str, object], json.load(stream))
    print(json.dumps(audit(candidate, source), sort_keys=True))


if __name__ == "__main__":
    main()
