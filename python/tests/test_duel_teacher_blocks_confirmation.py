from __future__ import annotations

from python.audit_duel_teacher_blocks import summarize
from python.audit_duel_teacher_blocks_confirmation import ARMS, confirmation_teacher


def test_spaced_schedule_uses_exactly_four_nonconsecutive_whole_turns() -> None:
    assert [confirmation_teacher("spaced_4", 6480000, 0, turn) for turn in range(8)] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert [confirmation_teacher("first_4", 6480000, 0, turn) for turn in range(8)] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_confirmation_summary_keeps_predeclared_arms_only() -> None:
    records: list[dict[str, object]] = []
    for seed in (1, 2):
        for seat in (0, 1):
            for arm in ARMS:
                records.append(
                    {
                        "seed": seed,
                        "root_seat": seat,
                        "arm": arm,
                        "root_turns": 8,
                        "teacher_turns": 4 if arm in ("first_4", "spaced_4") else 0,
                        "outcome": {
                            "winner": seat if seed == 1 and arm == "first_4" else 1 - seat,
                            "adjudicated_winner": None,
                            "terminal": True,
                            "truncated": False,
                            "actions_after_intervention": 12,
                        },
                    }
                )

    summary = summarize(records, ARMS)
    assert [row["arm"] for row in summary["arms"]] == list(ARMS)
    assert summary["arms"][2]["versus_first_1"]["finite_horizon_maps"]["candidate_better"] == 1
