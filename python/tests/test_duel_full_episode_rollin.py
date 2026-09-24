from copy import deepcopy

import numpy as np
import pytest

from python.audit_duel_full_episode_rollin import CHECKPOINTS, FIRST_SEED, audit
from python.evaluate import paired_map_comparison, winner_score


def report() -> dict[str, object]:
    winners = [0, 1, 0, 1]
    seats = [0, 1, 0, 1]
    source_winners = [0, 1]
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
    return {
        "checkpoint": CHECKPOINTS["full_episode"],
        "baseline_checkpoint": "routed-v6.pt",
        "baseline": "policy",
        "model_agent": "policy",
        "profile": "classic_generic_2022",
        "generator": "procedural_v2",
        "route_generator": "procedural_v1",
        "players": 2,
        "action_limit": 2400,
        "games": 4,
        "seed": FIRST_SEED,
        "pairing": {
            "first_seed": FIRST_SEED,
            "last_seed": FIRST_SEED + 1,
            "unique_seeds": 2,
            "scheme": "adjacent_same_seed_opposite_seat_v1",
        },
        "generator_config": {
            "schema_version": 2,
            "land_density_per_million": 650000,
            "starting_province_size": 5,
            "starting_money": 10,
            "tree_density_per_million": 150000,
            "neutral_tower_density_per_million": 20000,
            "neutral_capital_density_per_million": 10000,
            "grave_density_per_million": 15000,
        },
        "domain_descriptor": {
            "width": 11,
            "height": 9,
            "players": 2,
            "action_limit": 2400,
            "fog": False,
            "diplomacy": False,
        },
        "game_seeds": [FIRST_SEED, FIRST_SEED, FIRST_SEED + 1, FIRST_SEED + 1],
        "model_seats": seats,
        "winners": winners,
        "game_truncated": [False] * 4,
        "wins": 4,
        "draws": 0,
        "losses": 0,
        "truncations": 0,
        "baseline_self_play": {
            "games": 2,
            "winners": source_winners,
            "game_truncated": [False] * 2,
            "truncations": 0,
        },
        "paired_map_comparison": paired_map_comparison(
            candidate_scores, source_scores, 2, None
        ),
    }


def test_full_episode_audit_checks_both_seats_and_independent_maps() -> None:
    result = audit(report(), "full_episode", maps=2)

    assert result["candidate_wins_by_seat"] == [2, 2]
    assert result["source_reference_wins_by_seat"] == [1, 1]
    assert result["paired_independent_maps"]["candidate_better"] == 2
    assert result["gate"]["both_seats_improve"] is True
    assert result["direct_source_gate_passed"] is False


def test_full_episode_audit_rejects_bad_map_pairing_or_censor_accounting() -> None:
    wrong_seeds = deepcopy(report())
    wrong_seeds["game_seeds"] = [FIRST_SEED + 1] * 4
    with pytest.raises(ValueError, match="ledger"):
        audit(wrong_seeds, "full_episode", maps=2)

    wrong_truncations = deepcopy(report())
    wrong_truncations["game_truncated"] = [True, False, False, False]
    with pytest.raises(ValueError, match="totals"):
        audit(wrong_truncations, "full_episode", maps=2)
