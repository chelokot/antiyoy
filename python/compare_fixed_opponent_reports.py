from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from .build_bundle import digest
from .evaluate import (
    baseline_adjusted_elo_delta,
    paired_map_bootstrap_interval,
    paired_method_comparison,
    winner_score,
)


def compare(
    candidate: dict[str, object], reference: dict[str, object]
) -> dict[str, object]:
    shared_fields = (
        "baseline",
        "games",
        "seed",
        "players",
        "profile",
        "generator",
        "generator_config",
        "domain_descriptor",
        "action_limit",
        "route_generator",
        "model_seat",
        "model_seats",
        "game_seeds",
        "search_nodes",
        "search_beam_width",
        "search_branch_width",
        "search_maximum_actions_per_turn",
        "reply_search_nodes",
        "reply_slate_size",
    )
    if any(candidate[field] != reference[field] for field in shared_fields):
        raise ValueError("paired fixed-opponent reports use different arenas or maps")
    if candidate["baseline_self_play"] != reference["baseline_self_play"]:
        raise ValueError("paired fixed-opponent reports disagree on frozen opponent")
    if candidate["checkpoint"] == reference["checkpoint"]:
        raise ValueError("paired comparison requires two distinct policy checkpoints")
    seat = cast(int, candidate["model_seat"])
    if seat not in (0, 1) or candidate["players"] != 2:
        raise ValueError("paired comparison requires a fixed-seat two-player arena")
    games = cast(int, candidate["games"])
    candidate_winners = cast(list[int], candidate["winners"])
    reference_winners = cast(list[int], reference["winners"])
    if len(candidate_winners) != games or len(reference_winners) != games:
        raise ValueError("paired winner ledgers must cover every map")
    candidate_scores = np.asarray(
        [winner_score(winner, seat) for winner in candidate_winners]
    )
    reference_scores = np.asarray(
        [winner_score(winner, seat) for winner in reference_winners]
    )
    comparison = paired_method_comparison(candidate_scores, reference_scores)
    return {
        "kind": "matched_fixed_opponent_policy_comparison",
        "baseline": candidate["baseline"],
        "maps": games,
        "first_seed": candidate["seed"],
        "seat": seat,
        "candidate_checkpoint": candidate["checkpoint"],
        "reference_checkpoint": reference["checkpoint"],
        "candidate_wins": candidate["wins"],
        "reference_wins": reference["wins"],
        "candidate_truncations": candidate["truncations"],
        "reference_truncations": reference["truncations"],
        "paired_independent_maps": comparison,
        "candidate_score": float(candidate_scores.mean()),
        "reference_score": float(reference_scores.mean()),
        "pool_relative_elo_delta": baseline_adjusted_elo_delta(
            float(candidate_scores.mean()), float(reference_scores.mean()), games
        ),
        "map_bootstrap_95": paired_map_bootstrap_interval(
            candidate_scores, reference_scores, 2, seat, cast(int, candidate["seed"])
        ),
        "all_games_complete": (
            candidate["truncations"] == 0 and reference["truncations"] == 0
        ),
        "qualification": (
            "Fresh fixed-seat matched complete-game comparison against one frozen "
            "opponent. This is not global Elo or all-seat strength."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    arguments = parser.parse_args()
    with arguments.candidate.open(encoding="utf-8") as stream:
        candidate = cast(dict[str, object], json.load(stream))
    with arguments.reference.open(encoding="utf-8") as stream:
        reference = cast(dict[str, object], json.load(stream))
    report = compare(candidate, reference)
    report["candidate_report_sha256"] = digest(arguments.candidate)
    report["reference_report_sha256"] = digest(arguments.reference)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
