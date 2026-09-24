from copy import deepcopy

import numpy as np
import pytest

from python.audit_duel_selective_replan import FIRST_SEED, GAMES, MAPS, audit
from python.evaluate import paired_map_comparison, winner_score


def report() -> dict[str, object]:
    winners = [index % 2 for index in range(GAMES)]
    seats = [index % 2 for index in range(GAMES)]
    reference_winners = [0] * MAPS
    candidate_scores = np.asarray(
        [
            winner_score(winner, seat)
            for winner, seat in zip(winners, seats, strict=True)
        ]
    )
    reference_scores = np.asarray([winner_score(0, seat) for seat in seats])
    return {
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
        "pairing": {
            "first_seed": FIRST_SEED,
            "last_seed": FIRST_SEED + MAPS - 1,
            "unique_seeds": MAPS,
            "scheme": "adjacent_same_seed_opposite_seat_v1",
        },
        "game_seeds": [FIRST_SEED + index // 2 for index in range(GAMES)],
        "model_seats": seats,
        "winners": winners,
        "game_truncated": [False] * GAMES,
        "wins": GAMES,
        "truncations": 0,
        "baseline_self_play": {
            "winners": reference_winners,
            "game_truncated": [False] * MAPS,
        },
        "paired_map_comparison": paired_map_comparison(
            candidate_scores, reference_scores, 2, None
        ),
        "selective_reply_search": {
            "queries": 100,
            "candidate_decisions": 256,
            "query_fraction": 100 / 256,
        },
        "model_baseline_policy_actions": {
            "decisions": 256,
            "disagreements": 80,
            "disagreement_rate": 80 / 256,
        },
    }


def test_selective_audit_verifies_both_seats_and_query_gate() -> None:
    result = audit(report())

    assert result["candidate_direct_wins_by_seat"] == [128, 128]
    assert result["source_self_play_reference_wins_by_seat"] == [128, 0]
    assert result["final_actions_different_from_source"] == 80
    assert result["exploratory_gate_passed"] is False
    assert result["gate"]["positive_source_relative_wins_in_both_seats"] is False


def test_selective_audit_rejects_misaligned_seed_and_query_accounting() -> None:
    wrong_seed = report()
    wrong_seed["game_seeds"] = [FIRST_SEED + 1] * GAMES
    with pytest.raises(ValueError, match="seeds"):
        audit(wrong_seed)

    wrong_queries = deepcopy(report())
    wrong_queries["selective_reply_search"]["query_fraction"] = 0.7
    with pytest.raises(ValueError, match="query accounting"):
        audit(wrong_queries)
