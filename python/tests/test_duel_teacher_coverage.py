from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from torch import Tensor

import python.audit_duel_teacher_coverage as coverage
from antiyoy_rl import ProceduralConfig, VectorEnv


def test_teacher_turn_hash_is_reproducible_and_nested() -> None:
    for turn in range(200):
        choices = [
            coverage.teacher_turn(6440000, 1, turn, rate)
            for rate in (0, 25, 50, 75, 100)
        ]
        assert choices[0] is False
        assert choices[-1] is True
        assert choices == sorted(choices)
        assert coverage.teacher_turn(6440000, 1, turn, 50) == choices[2]
    with pytest.raises(ValueError, match="between zero and 100"):
        coverage.teacher_turn(6440000, 1, 0, 101)


def test_whole_turn_coverage_preserves_the_direct_policy_outcome(
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

    def small_environment(seed: int) -> VectorEnv:
        return VectorEnv.procedural(
            1,
            ProceduralConfig(width=7, height=5, players=2, seed=seed),
            action_limit=12,
            profile="classic_generic_2022",
        )

    def matching_teacher(environment: VectorEnv, followup_nodes: int) -> np.ndarray:
        assert followup_nodes == coverage.FOLLOWUP_NODES
        return EndTurnPolicy().actions(environment.observe(), Tensor())

    monkeypatch.setattr(coverage, "create_environment", small_environment)
    monkeypatch.setattr(coverage, "native_teacher_action", matching_teacher)

    observed_turns: list[int] = []
    direct = coverage.play_game(
        6449000,
        0,
        0,
        EndTurnPolicy(),
        on_root_turn=lambda observation, turn: observed_turns.append(turn),
    )
    searched = coverage.play_game(6449000, 0, 100, EndTurnPolicy())

    assert direct["outcome"] == searched["outcome"]
    assert observed_turns == list(range(direct["root_turns"]))
    assert direct["teacher_turns"] == 0
    assert searched["teacher_turns"] == searched["root_turns"]
    assert searched["teacher_actions"] == searched["teacher_turns"]
    assert searched["outcome"]["truncated"] is True


def test_coverage_summary_groups_rotated_seats_by_map() -> None:
    records = []
    for seed in (1, 2):
        for seat in (0, 1):
            for percentage in coverage.TEACHER_PERCENTAGES:
                winner = seat if percentage == 100 and seed == 1 else 1 - seat
                records.append(
                    {
                        "seed": seed,
                        "root_seat": seat,
                        "teacher_percentage": percentage,
                        "teacher_turns": int(percentage > 0),
                        "root_turns": 1,
                        "outcome": {
                            "winner": winner,
                            "adjudicated_winner": None,
                            "terminal": True,
                            "truncated": False,
                            "actions_after_intervention": 5,
                        },
                    }
                )

    report = coverage.summarize(records)
    direct = report["arms"][0]
    searched = report["arms"][-1]

    assert direct["paired_finite_horizon_maps"]["same"] == 2
    assert searched["wins_by_root_seat"] == [1, 1]
    assert searched["paired_finite_horizon_maps"]["candidate_better"] == 1
    assert searched["paired_finite_horizon_maps"]["same"] == 1
    assert searched["fully_terminal_maps"] == 2
