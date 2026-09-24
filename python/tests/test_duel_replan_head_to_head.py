from __future__ import annotations

from python.audit_duel_replan_head_to_head import summarize


def test_head_to_head_summary_rotates_seats_and_groups_independent_maps() -> None:
    records: list[dict[str, object]] = []
    for seed in (1, 2):
        for replanned_seat in (0, 1):
            records.append(
                {
                    "seed": seed,
                    "replanned_seat": replanned_seat,
                    "outcome": {
                        "winner": replanned_seat if seed == 1 else 1 - replanned_seat,
                        "adjudicated_winner": None,
                        "terminal": True,
                        "truncated": False,
                        "actions_after_intervention": 20,
                    },
                    "actions_by_seat": [10, 10],
                    "turn_milliseconds_by_seat": [[2.0], [4.0]],
                }
            )

    result = summarize(records)

    assert result["games"] == 4
    assert result["terminal_games"] == 4
    assert result["replanned_wins_by_seat"] == [1, 1]
    assert result["cached_wins_by_seat"] == [1, 1]
    assert result["replanned_vs_cached_paired_maps"]["candidate_better"] == 1
    assert result["replanned_vs_cached_paired_maps"]["baseline_better"] == 1
    assert result["turn_timing"]["replanned"]["turns"] == 4
