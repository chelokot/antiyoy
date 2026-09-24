from __future__ import annotations

from python.audit_duel_native_replan_equivalence import summarize


def test_equivalence_summary_counts_games_actions_and_full_turns() -> None:
    records: list[dict[str, object]] = []
    for seat in (0, 1):
        records.append(
            {
                "seed": 3,
                "replanned_seat": seat,
                "outcome": {
                    "winner": seat,
                    "adjudicated_winner": None,
                    "terminal": True,
                    "truncated": False,
                    "actions_after_intervention": 12,
                },
                "actions_compared": 12,
                "fork_turn_milliseconds": [2.0, 4.0],
                "native_turn_milliseconds": [1.0, 3.0],
            }
        )

    result = summarize(records)

    assert result["games"] == 2
    assert result["independent_maps"] == 1
    assert result["actions_compared"] == 24
    assert result["terminal_games"] == 2
    assert result["fork_full_turn_milliseconds"]["turns"] == 4
    assert result["native_full_turn_milliseconds"]["median"] == 2.0
