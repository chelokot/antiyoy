from __future__ import annotations

import numpy as np

from python.audit_duel_first_regret import create_environment, native_teacher_action
from python.audit_duel_teacher_plan_cache import replanned_teacher_action, summarize
from python.audit_duel_three_turn_intervention import FOLLOWUP_NODES


def test_fresh_replan_agrees_on_first_action_without_mutating_live_state() -> None:
    environment = create_environment(6490000)
    before = environment.observe()
    cached = native_teacher_action(environment, FOLLOWUP_NODES)
    replanned = replanned_teacher_action(environment)

    np.testing.assert_array_equal(cached, replanned)
    after = environment.observe()
    for key in ("owners", "objects", "active_players", "action_offsets"):
        np.testing.assert_array_equal(before[key], after[key])


def test_plan_cache_summary_groups_both_root_seats_by_map() -> None:
    records: list[dict[str, object]] = []
    for seed in (1, 2):
        for seat in (0, 1):
            for arm in ("direct", "cached_teacher", "replanned_teacher"):
                records.append(
                    {
                        "seed": seed,
                        "root_seat": seat,
                        "arm": arm,
                        "root_turns": 2,
                        "root_actions": 4,
                        "cache_replan_comparisons_by_action_position": [2, 2, 0, 0],
                        "cache_replan_disagreements_by_action_position": [0, 1, 0, 0],
                        "outcome": {
                            "winner": seat if seed == 1 and arm == "cached_teacher" else 1 - seat,
                            "adjudicated_winner": None,
                            "terminal": True,
                            "truncated": False,
                            "actions_after_intervention": 12,
                        },
                    }
                )

    result = summarize(records)

    assert result["independent_maps"] == 2
    assert result["cached_vs_fresh_replan"]["compared_by_action_position"] == [8, 8, 0, 0]
    assert result["cached_vs_fresh_replan"]["disagreements_by_action_position"] == [0, 4, 0, 0]
    assert result["cached_vs_replanned_games"]["finite_horizon_maps"]["candidate_better"] == 1
