from copy import deepcopy

import pytest

from python.audit_duel_selective_teacher_confirmation import (
    FIRST_SEED,
    GAMES,
    MAPS,
    audit,
)


def report(agent: str, winning_maps: int) -> dict[str, object]:
    seats = [index % 2 for index in range(GAMES)]
    winners = [
        seat if index // 2 < winning_maps else 1 - seat
        for index, seat in enumerate(seats)
    ]
    return {
        "checkpoint": (
            "replan-markov-student-6500000.pt"
            if agent == "selective_reply_search"
            else "routed-v6.pt"
        ),
        "selective_source_checkpoint": "routed-v6.pt",
        "baseline": "reply_search",
        "baseline_checkpoint": None,
        "model_agent": agent,
        "profile": "classic_generic_2022",
        "generator": "procedural_v2",
        "route_generator": "procedural_v1",
        "generator_config": {"schema_version": 2},
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
        "game_seeds": [FIRST_SEED + index // 2 for index in range(GAMES)],
        "model_seats": seats,
        "winners": winners,
        "game_truncated": [False] * GAMES,
        "wins": winning_maps * 2,
        "truncations": 0,
        "baseline_self_play": {
            "winners": [0] * MAPS,
            "game_truncated": [False] * MAPS,
        },
        "selective_reply_search": {
            "queries": 100,
            "candidate_decisions": 400,
            "query_fraction": 0.25,
            "final_actions_different_from_source": 75,
        },
    }


def test_selective_confirmation_checks_paired_maps_and_both_seats() -> None:
    result = audit(report("selective_reply_search", 40), report("policy", 20))

    assert result["candidate_wins_by_seat"] == [40, 40]
    assert result["source_wins_by_seat"] == [20, 20]
    assert result["paired_independent_maps"]["candidate_better"] == 20
    assert result["confirmation_gate_passed"] is True


def test_selective_confirmation_rejects_misaligned_or_censored_games() -> None:
    candidate = report("selective_reply_search", 40)
    source = report("policy", 20)
    wrong_seed = deepcopy(source)
    wrong_seed["game_seeds"][0] += 1
    with pytest.raises(ValueError, match="mispaired"):
        audit(candidate, wrong_seed)

    censored = deepcopy(candidate)
    censored["game_truncated"][0] = True
    censored["truncations"] = 1
    censored_result = audit(censored, source)
    assert censored_result["candidate_nonterminal_games"] == 1
    assert censored_result["gate"]["all_256_games_terminal"] is False
    assert censored_result["confirmation_gate_passed"] is False

    wrong_reference = deepcopy(source)
    wrong_reference["baseline_self_play"]["winners"][0] = 1
    with pytest.raises(ValueError, match="self-play"):
        audit(candidate, wrong_reference)

    wrong_queries = deepcopy(candidate)
    wrong_queries["selective_reply_search"]["query_fraction"] = 0.4
    with pytest.raises(ValueError, match="query accounting"):
        audit(wrong_queries, source)
