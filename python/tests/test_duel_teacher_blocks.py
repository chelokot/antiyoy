from __future__ import annotations

from python.audit_duel_teacher_blocks import ARMS, scheduled_teacher, summarize
from python.audit_duel_teacher_coverage import teacher_turn


def test_teacher_block_schedule_is_contiguous_and_distributed_arm_is_unchanged() -> None:
    assert [scheduled_teacher("first_2", 6470000, 1, turn) for turn in range(4)] == [
        True,
        True,
        False,
        False,
    ]
    assert scheduled_teacher("direct", 6470000, 1, 0) is False
    assert scheduled_teacher("full_teacher", 6470000, 1, 50) is True
    assert [scheduled_teacher("distributed_25", 6470000, 1, turn) for turn in range(20)] == [
        teacher_turn(6470000, 1, turn, 25) for turn in range(20)
    ]


def test_block_summary_groups_rotated_seats_by_independent_map() -> None:
    records: list[dict[str, object]] = []
    for seed in (1, 2):
        for seat in (0, 1):
            for arm in ARMS:
                winner = seat if seed == 1 and arm == "first_2" else 1 - seat
                records.append(
                    {
                        "seed": seed,
                        "root_seat": seat,
                        "arm": arm,
                        "root_turns": 3,
                        "teacher_turns": 2 if arm == "first_2" else 0,
                        "outcome": {
                            "winner": winner,
                            "adjudicated_winner": None,
                            "terminal": True,
                            "truncated": False,
                            "actions_after_intervention": 12,
                        },
                    }
                )

    result = summarize(records)
    block = next(row for row in result["arms"] if row["arm"] == "first_2")

    assert result["independent_maps"] == 2
    assert block["wins_by_root_seat"] == [1, 1]
    assert block["actual_teacher_turns"] == 8
    assert block["versus_direct"]["finite_horizon_maps"]["candidate_better"] == 1
    assert block["versus_first_1"]["fully_terminal_maps"]["same"] == 1
