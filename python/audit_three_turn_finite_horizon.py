from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from .evaluate import (
    baseline_adjusted_elo_delta,
    paired_map_bootstrap_interval,
    paired_map_comparison,
    paired_method_comparison,
    winner_score,
)


class Episode(TypedDict):
    winner: int | None
    truncated: bool
    actions: int


class MatchRecord(TypedDict):
    seed: int
    candidate_seat: int
    candidate: Episode
    baseline: Episode


class RawReport(TypedDict):
    generator: dict[str, int]
    rules: str
    candidate: str
    baseline: str
    search_nodes: int
    baseline_search_nodes: int
    candidate_reply_nodes: int
    baseline_reply_nodes: int
    candidate_followup_nodes: int
    candidate_slate_size: int
    maps: int
    games: int
    baseline_reference_games: int
    action_limit: int
    candidate_wins_by_seat: list[int]
    baseline_wins_by_seat: list[int]
    candidate_truncations: int
    baseline_reference_truncations: int
    records: list[MatchRecord]


def episode_score(episode: Episode, seat: int) -> float:
    winner = episode["winner"]
    return winner_score(255 if winner is None else winner, seat)


def audit(
    raw: RawReport,
    expected_seed: int,
    expected_maps: int,
    expected_rules: str = "classic_generic_2022",
) -> dict[str, object]:
    generator = raw["generator"]
    generator_defaults = {
        "land_density_per_million": 650_000,
        "starting_province_size": 5,
        "starting_money": 10,
        "tree_density_per_million": 150_000,
        "neutral_tower_density_per_million": 20_000,
        "neutral_capital_density_per_million": 10_000,
        "grave_density_per_million": 15_000,
    }
    if (
        any(generator[field] != value for field, value in generator_defaults.items())
        or generator["schema_version"] != 2
        or generator["width"] != 11
        or generator["height"] != 9
        or generator["players"] != 2
        or generator["seed"] != expected_seed
        or raw["rules"] != expected_rules
        or raw["candidate"] != "three-turn-search"
        or raw["baseline"] != "reply-search"
        or raw["search_nodes"] != 256
        or raw["baseline_search_nodes"] != 256
        or raw["candidate_reply_nodes"] != 64
        or raw["baseline_reply_nodes"] != 64
        or raw["candidate_followup_nodes"] != 32
        or raw["candidate_slate_size"] != 8
        or raw["action_limit"] != 2400
        or raw["maps"] != expected_maps
        or raw["games"] != expected_maps * 2
        or raw["baseline_reference_games"] != expected_maps
    ):
        raise ValueError("report does not match the frozen finite-horizon protocol")

    records = raw["records"]
    if len(records) != expected_maps * 2:
        raise ValueError("report does not cover every map and seat")
    for index, record in enumerate(records):
        if (
            record["seed"] != expected_seed + index // 2
            or record["candidate_seat"] != index % 2
        ):
            raise ValueError("map and seat ledger differs from the protocol")
        if index % 2 == 1 and record["baseline"] != records[index - 1]["baseline"]:
            raise ValueError("opposite seats disagree on the baseline reference")
        for episode in (record["candidate"], record["baseline"]):
            if (
                episode["winner"] not in (0, 1, None)
                or not 0 < episode["actions"] <= 2400
            ):
                raise ValueError("episode outcome differs from the arena")
            if episode["truncated"] and episode["actions"] != 2400:
                raise ValueError("nonterminal episode ended before the horizon")

    candidate_wins_by_seat = [
        sum(
            record["candidate"]["winner"] == seat
            for record in records
            if record["candidate_seat"] == seat
        )
        for seat in range(2)
    ]
    baseline_wins_by_seat = [
        sum(
            record["baseline"]["winner"] == seat
            for record in records
            if record["candidate_seat"] == seat
        )
        for seat in range(2)
    ]
    if (
        candidate_wins_by_seat != raw["candidate_wins_by_seat"]
        or baseline_wins_by_seat != raw["baseline_wins_by_seat"]
    ):
        raise ValueError("seat wins disagree with the game ledger")

    candidate_scores = np.asarray(
        [
            episode_score(record["candidate"], record["candidate_seat"])
            for record in records
        ]
    )
    baseline_scores = np.asarray(
        [
            episode_score(record["baseline"], record["candidate_seat"])
            for record in records
        ]
    )
    candidate_truncated = np.asarray(
        [record["candidate"]["truncated"] for record in records]
    )
    baseline_truncated = np.asarray(
        [record["baseline"]["truncated"] for record in records]
    )
    candidate_timeouts = int(candidate_truncated.sum())
    reference_timeouts = int(baseline_truncated.reshape(-1, 2)[:, 0].sum())
    if (
        candidate_timeouts != raw["candidate_truncations"]
        or reference_timeouts != raw["baseline_reference_truncations"]
    ):
        raise ValueError("timeout counts disagree with the game ledger")

    complete_maps = ~(candidate_truncated | baseline_truncated).reshape(-1, 2).any(
        axis=1
    )
    complete_candidate = candidate_scores.reshape(-1, 2)[complete_maps].reshape(-1)
    complete_baseline = baseline_scores.reshape(-1, 2)[complete_maps].reshape(-1)
    worst_candidate = np.where(candidate_truncated, 0.0, candidate_scores)
    worst_baseline = np.where(baseline_truncated, 1.0, baseline_scores)
    paired_maps = paired_map_comparison(candidate_scores, baseline_scores, 2, None)
    worst_maps = paired_map_comparison(worst_candidate, worst_baseline, 2, None)
    interval = paired_map_bootstrap_interval(
        candidate_scores, baseline_scores, 2, None, expected_seed
    )
    timeout_ledger = [
        {
            "seed": record["seed"],
            "candidate_seat": record["candidate_seat"],
            "candidate_truncated": record["candidate"]["truncated"],
            "baseline_truncated": record["baseline"]["truncated"],
            "candidate_adjudicated_winner": record["candidate"]["winner"],
            "baseline_adjudicated_winner": record["baseline"]["winner"],
        }
        for record in records
        if record["candidate"]["truncated"] or record["baseline"]["truncated"]
    ]
    candidate_mean = float(candidate_scores.mean())
    baseline_mean = float(baseline_scores.mean())
    gate = {
        "paired_map_sign": (
            paired_maps["candidate_better"] > paired_maps["baseline_better"]
            and paired_maps["exact_two_sided_sign_test_p"] < 0.05
        ),
        "bootstrap_lower_elo_positive": interval["baseline_adjusted_elo_delta"][0] > 0,
        "both_seats_nonnegative": all(
            candidate >= baseline
            for candidate, baseline in zip(
                raw["candidate_wins_by_seat"],
                raw["baseline_wins_by_seat"],
                strict=True,
            )
        ),
        "candidate_timeout_rate_at_most_one_percent": candidate_timeouts
        <= expected_maps * 0.02,
        "reference_timeout_rate_at_most_one_percent": reference_timeouts
        <= expected_maps * 0.01,
        "worst_case_timeout_sign": (
            worst_maps["candidate_better"] > worst_maps["baseline_better"]
            and worst_maps["exact_two_sided_sign_test_p"] < 0.05
        ),
    }
    return {
        "kind": "three_turn_finite_horizon_fresh_matched_audit",
        "protocol": "benchmarks/protocols/2026-09-24-duel-three-turn-finite-horizon-v1.json",
        "rules": raw["rules"],
        "seed_first": expected_seed,
        "maps": expected_maps,
        "seat_games": expected_maps * 2,
        "candidate_wins_by_seat": raw["candidate_wins_by_seat"],
        "baseline_reference_wins_by_seat": raw["baseline_wins_by_seat"],
        "candidate_timeouts": candidate_timeouts,
        "baseline_reference_timeouts": reference_timeouts,
        "timeout_ledger": timeout_ledger,
        "candidate_finite_horizon_score": candidate_mean,
        "baseline_reference_finite_horizon_score": baseline_mean,
        "pool_relative_finite_horizon_elo": baseline_adjusted_elo_delta(
            candidate_mean, baseline_mean, expected_maps * 2
        ),
        "map_bootstrap_95": interval,
        "paired_seat_games": paired_method_comparison(
            candidate_scores, baseline_scores
        ),
        "paired_independent_maps": paired_maps,
        "worst_case_timeout_independent_maps": worst_maps,
        "fully_terminal_maps": int(complete_maps.sum()),
        "fully_terminal_independent_maps": (
            paired_map_comparison(complete_candidate, complete_baseline, 2, None)
            if complete_maps.any()
            else None
        ),
        "gate": gate,
        "primary_gate_without_runtime": all(gate.values()),
        "profile_gate": (
            paired_maps["candidate_better"] >= paired_maps["baseline_better"]
            and gate["both_seats_nonnegative"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6_410_000)
    parser.add_argument("--maps", type=int, default=256)
    parser.add_argument("--rules", default="classic_generic_2022")
    arguments = parser.parse_args()
    with arguments.input.open(encoding="utf-8") as stream:
        raw = cast(RawReport, json.load(stream))
    print(
        json.dumps(
            audit(raw, arguments.seed, arguments.maps, arguments.rules), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
