from __future__ import annotations

import numpy as np
from pathlib import Path
from pytest import MonkeyPatch

from python import audit_duel_teacher_plan_cache as cache_audit
from python.audit_duel_first_regret import create_environment, native_teacher_action
from python.audit_duel_teacher_plan_cache import (
    audit_window,
    replanned_teacher_action,
    summarize,
)
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
                            "winner": seat
                            if seed == 1 and arm == "cached_teacher"
                            else 1 - seat,
                            "adjudicated_winner": None,
                            "terminal": True,
                            "truncated": False,
                            "actions_after_intervention": 12,
                        },
                    }
                )

    result = summarize(records)

    assert result["independent_maps"] == 2
    assert result["cached_vs_fresh_replan"]["compared_by_action_position"] == [
        8,
        8,
        0,
        0,
    ]
    assert result["cached_vs_fresh_replan"]["disagreements_by_action_position"] == [
        0,
        4,
        0,
        0,
    ]
    assert (
        result["cached_vs_replanned_games"]["finite_horizon_maps"]["candidate_better"]
        == 1
    )


def test_audit_window_uses_the_predeclared_seed_range_and_both_seats(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_audit, "digest", lambda _: cache_audit.CHECKPOINT_SHA256)
    monkeypatch.setattr(
        cache_audit, "load_routed_policy", lambda _: (object(), ["seat0", "seat1"])
    )

    def fake_game(seed: int, seat: int, arm: str, _: object) -> dict[str, object]:
        return {
            "seed": seed,
            "root_seat": seat,
            "arm": arm,
            "root_turns": 1,
            "root_actions": 1,
            "cache_replan_comparisons_by_action_position": [1, 0, 0, 0],
            "cache_replan_disagreements_by_action_position": [0, 0, 0, 0],
            "outcome": {
                "winner": seat,
                "adjudicated_winner": None,
                "terminal": True,
                "truncated": False,
                "actions_after_intervention": 2,
            },
        }

    monkeypatch.setattr(cache_audit, "play_game", fake_game)
    result = audit_window(Path("unused"), 50, 2, "frozen-protocol", "frozen-kind")

    assert result["protocol"] == "frozen-protocol"
    assert result["kind"] == "frozen-kind"
    assert result["seed_first"] == 50
    assert result["maps"] == 2
    assert [
        (row["seed"], row["root_seat"], row["arm"]) for row in result["records"]
    ] == [
        (seed, seat, arm)
        for seed in (50, 51)
        for seat in (0, 1)
        for arm in cache_audit.ARMS
    ]
