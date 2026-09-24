from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from torch import Tensor

import python.audit_duel_three_turn_intervention as intervention
from antiyoy_rl import VectorEnv
from python.audit_duel_first_regret import create_environment, native_teacher_action
from python.audit_duel_three_turn_intervention import (
    FOLLOWUP_NODES,
    complete_turn,
    sample_first_disagreement,
    summarize,
)


def test_three_turn_teacher_completes_only_the_forked_turn() -> None:
    original = create_environment(6439000)
    before = original.observe()
    root = int(before["active_players"][0])
    branch = original.fork(np.asarray([0], dtype=np.uint64))

    plan, result = complete_turn(
        branch,
        root,
        lambda environment: native_teacher_action(environment, FOLLOWUP_NODES),
    )

    assert plan
    assert int(native_teacher_action(original, FOLLOWUP_NODES)[0]) == plan[0]
    np.testing.assert_array_equal(original.observe()["owners"], before["owners"])
    assert not original.done()[0]
    assert branch.done()[0] or int(branch.observe()["active_players"][0]) != root
    assert not (bool(result["terminal"][0]) and bool(result["truncated"][0]))


def test_later_round_window_skips_earlier_teacher_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EndTurnPolicy:
        def actions(
            self, observation: Mapping[str, np.ndarray], rules: Tensor
        ) -> np.ndarray:
            return np.asarray(
                [np.flatnonzero(observation["action_kinds"] == 0)[0]],
                dtype=np.uint64,
            )

    queried_rounds: list[int] = []

    def matching_teacher(environment: VectorEnv, followup_nodes: int) -> np.ndarray:
        assert followup_nodes == FOLLOWUP_NODES
        observation = environment.observe()
        queried_rounds.append(int(observation["rounds"][0]))
        return EndTurnPolicy().actions(observation, Tensor())

    monkeypatch.setattr(intervention, "native_teacher_action", matching_teacher)

    sample = sample_first_disagreement(
        6439001,
        0,
        EndTurnPolicy(),
        maximum_round=3,
        minimum_round=3,
    )

    assert queried_rounds == [3]
    assert sample["observed_root_turns"] == 1
    assert sample["no_intervention_reason"] == "round_limit"


def test_intervention_summary_censors_adjudication_and_groups_maps() -> None:
    def branch(winner: int, truncated: bool = False) -> dict[str, object]:
        return {
            "terminal": not truncated,
            "truncated": truncated,
            "winner": None if truncated else winner,
            "adjudicated_winner": winner if truncated else None,
            "actions_after_intervention": 7,
        }

    samples = [
        {
            "seed": 10,
            "root_seat": 0,
            "outcomes": {"direct": branch(1), "teacher": branch(0)},
        },
        {
            "seed": 10,
            "root_seat": 1,
            "outcomes": {"direct": branch(1), "teacher": branch(0)},
        },
        {
            "seed": 11,
            "root_seat": 1,
            "outcomes": {"direct": branch(0), "teacher": branch(1)},
        },
        {
            "seed": 12,
            "root_seat": 0,
            "outcomes": {
                "direct": branch(0),
                "teacher": branch(1, truncated=True),
            },
        },
        {"seed": 11, "root_seat": 0, "no_intervention_reason": "round_limit"},
    ]

    report = summarize(samples)

    assert report["sampled_positions"] == 4
    assert report["no_intervention_reasons"] == {"round_limit": 1}
    assert report["by_root_seat"][0]["teacher_better"] == 1
    assert report["by_root_seat"][0]["censored"] == 1
    assert report["maps_with_both_seats_uncensored"] == 1
    assert report["map_net_terminal_outcome"] == {
        "teacher_better": 1,
        "direct_better": 0,
        "same": 1,
        "exact_two_sided_sign_test_p": 1.0,
    }
