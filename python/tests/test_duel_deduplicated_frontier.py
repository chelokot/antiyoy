from __future__ import annotations

from copy import deepcopy

from python.audit_duel_deduplicated_frontier import summarize


def record(seed: int, seat: int, winner: int) -> dict[str, object]:
    return {
        "seed": seed,
        "candidate_seat": seat,
        "outcome": {
            "terminal": True,
            "truncated": False,
            "winner": winner,
            "adjudicated_winner": None,
            "actions_after_intervention": 8,
        },
        "actions": 8,
        "candidate_turns": [[0.01, 0.01]],
        "baseline_turns": [[0.01, 0.01]],
    }


def test_summary_requires_rotated_games_and_withholds_elo_when_censored() -> None:
    records = [
        record(1, 0, 0),
        record(1, 1, 1),
        record(2, 0, 1),
        record(2, 1, 1),
    ]
    summary = summarize(records)
    assert summary["candidate_wins_by_seat"] == [1, 2]
    assert summary["paired_map_signs"]["candidate_better"] == 1
    assert summary["paired_map_signs"]["same"] == 1
    assert summary["fixed_pool_head_to_head_elo_candidate_over_baseline"] is not None

    censored = deepcopy(records)
    censored[0]["outcome"] = {
        "terminal": False,
        "truncated": True,
        "winner": None,
        "adjudicated_winner": 0,
        "actions_after_intervention": 2400,
    }
    censored_summary = summarize(censored)
    assert censored_summary["action_limit_censored_games"] == 1
    assert (
        censored_summary["fixed_pool_head_to_head_elo_candidate_over_baseline"] is None
    )
    assert censored_summary["map_bootstrap_95_fixed_pool_elo"] is None
    assert censored_summary["advance_gate_passed"] is False
