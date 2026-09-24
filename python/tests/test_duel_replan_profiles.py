from __future__ import annotations

from python.audit_duel_replan_profiles import worst_case_comparison


def test_profile_timeout_sensitivity_counts_replanned_truncation_as_loss() -> None:
    records: list[dict[str, object]] = []
    for seat in (0, 1):
        records.append(
            {
                "seed": 2,
                "replanned_seat": seat,
                "outcome": {
                    "winner": None,
                    "adjudicated_winner": seat,
                    "terminal": False,
                    "truncated": True,
                    "actions_after_intervention": 2400,
                },
            }
        )

    result = worst_case_comparison(records)

    assert result["candidate_better"] == 0
    assert result["baseline_better"] == 1
