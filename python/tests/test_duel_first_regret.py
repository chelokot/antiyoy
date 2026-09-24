from __future__ import annotations

import numpy as np

from python.audit_duel_first_regret import (
    create_environment,
    first_strict_regret,
    force_completed_turn,
    outcome,
    score,
    summarize,
)


def test_first_strict_regret_ignores_native_reply_score_ties() -> None:
    record = {
        "native_reply_scores": [10, 10, 5],
        "autonomous_selected_index": 1,
        "native_selected_index": 0,
    }

    assert first_strict_regret(record) is None
    record["autonomous_selected_index"] = 2
    assert first_strict_regret(record) == (2, 0, 5)


def test_forced_complete_turn_preserves_original_state() -> None:
    original = create_environment(6389000)
    original_state = original.observe()
    root = int(original_state["active_players"][0])
    plans, _ = original.search_turn_plans(node_budget=64, slate_size=4)
    branch = original.fork(np.asarray([0], dtype=np.uint64))

    result = force_completed_turn(branch, root, plans[0][0])

    assert not original.done()[0]
    np.testing.assert_array_equal(original.observe()["owners"], original_state["owners"])
    if result is None:
        assert int(branch.observe()["active_players"][0]) != root
    else:
        assert branch.done()[0]


def test_action_limit_adjudication_remains_censored() -> None:
    branch = outcome(
        {
            "terminal": np.asarray([0], dtype=np.uint8),
            "truncated": np.asarray([1], dtype=np.uint8),
            "winners": np.asarray([255], dtype=np.uint8),
            "adjudicated_winners": np.asarray([0], dtype=np.uint8),
        },
        12,
    )

    assert branch["adjudicated_winner"] == 0
    assert branch["winner"] is None
    assert score(branch, 0) is None


def test_summary_groups_outcomes_by_independent_map() -> None:
    def branch(winner: int) -> dict[str, object]:
        return {
            "terminal": True,
            "truncated": False,
            "winner": winner,
            "adjudicated_winner": None,
            "actions_after_intervention": 4,
        }

    samples = [
        {
            "seed": 10,
            "root_seat": 0,
            "outcomes": {"hybrid": branch(1), "native_best": branch(0)},
        },
        {
            "seed": 10,
            "root_seat": 1,
            "outcomes": {"hybrid": branch(1), "native_best": branch(0)},
        },
        {
            "seed": 11,
            "root_seat": 1,
            "outcomes": {"hybrid": branch(0), "native_best": branch(1)},
        },
        {"seed": 11, "root_seat": 0, "no_intervention_reason": "round_limit"},
    ]

    report = summarize(samples)

    assert report["positions_with_strict_regret"] == 3
    assert report["maps_with_both_seats_uncensored"] == 1
    assert report["map_net_outcome"] == {
        "native_better": 1,
        "hybrid_better": 0,
        "same": 1,
        "exact_two_sided_sign_test_p": 1.0,
    }
