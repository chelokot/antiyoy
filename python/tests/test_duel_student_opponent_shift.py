import numpy as np
import pytest
import torch

from python import audit_duel_student_opponent_shift as audit


class FakeEnvironment:
    def __init__(self) -> None:
        self.active = 0
        self.root_turns = 0
        self.finished = False
        self.actions: list[tuple[int, int]] = []

    def done(self) -> np.ndarray:
        return np.asarray([self.finished])

    def observe(self) -> dict[str, np.ndarray]:
        return {"active_players": np.asarray([self.active])}

    def rules_jsons(self) -> list[str]:
        return ["{}"]

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        self.actions.append((self.active, int(action[0])))
        if self.active == 0:
            self.root_turns += 1
            self.finished = self.root_turns == 4
            self.active = 1
        else:
            self.active = 0
        return {
            "terminal": np.asarray([self.finished]),
            "truncated": np.asarray([False]),
            "winners": np.asarray([0]),
            "adjudicated_winners": np.asarray([0]),
        }


class FakePolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def actions(
        self, observation: dict[str, np.ndarray], rules: torch.Tensor
    ) -> np.ndarray:
        return np.asarray([self.action], dtype=np.uint64)


@pytest.mark.parametrize(
    ("arm", "opponent_action"),
    [("student_selfplay", 8), ("student_vs_source", 7)],
)
def test_root_is_student_and_opponent_changes_at_fixed_checkpoints(
    monkeypatch: pytest.MonkeyPatch, arm: str, opponent_action: int
) -> None:
    environment = FakeEnvironment()
    monkeypatch.setattr(audit, "create_environment", lambda seed: environment)
    monkeypatch.setattr(
        audit, "encode_rules_batch", lambda rules, device: torch.empty(0)
    )
    monkeypatch.setattr(
        audit,
        "native_teacher_action",
        lambda game, followup_nodes, replan: np.asarray([9], dtype=np.uint64),
    )
    monkeypatch.setattr(
        audit,
        "turn_state",
        lambda observation, root, turn_index: {"turn": turn_index + 1},
    )

    result = audit.play_game(6540000, 0, arm, FakePolicy(8), FakePolicy(7))

    assert [action for seat, action in environment.actions if seat == 0] == [8] * 4
    assert [action for seat, action in environment.actions if seat == 1] == [
        opponent_action
    ] * 3
    assert result["root_turns"] == 4
    assert set(result["checkpoints"]) == {"1", "4"}
    assert not result["checkpoints"]["1"]["student_matches_teacher"]
    assert result["outcome"]["terminal"]


def test_summary_uses_only_paired_surviving_states() -> None:
    records = [
        {
            "seed": 1,
            "root_seat": 0,
            "arm": arm,
            "checkpoints": {
                "1": {
                    "student_matches_teacher": matched,
                    "state": {
                        "owned_cells": lead,
                        "province_money": 0,
                        "province_profit": 0,
                        "unit_strength": 0,
                    },
                }
            },
            "outcome": {"terminal": True},
        }
        for arm, matched, lead in (
            ("student_selfplay", True, 2),
            ("student_vs_source", False, -1),
        )
    ]
    records.extend(
        {
            "seed": 1,
            "root_seat": 1,
            "arm": arm,
            "checkpoints": {},
            "outcome": {"terminal": True},
        }
        for arm in audit.ARMS
    )

    summary = audit.summarize(records)
    first = summary["checkpoints"][0]

    assert first["paired_seed_seat_games"] == 1
    assert first["excluded_seed_seat_games"] == 1
    assert first["by_root_seat"] == [
        {
            "seat": 0,
            "pairs": 1,
            "selfplay_teacher_matches": 1,
            "source_opponent_teacher_matches": 0,
        },
        {
            "seat": 1,
            "pairs": 0,
            "selfplay_teacher_matches": 0,
            "source_opponent_teacher_matches": 0,
        },
    ]
    assert first["median_source_opponent_minus_selfplay_lead"]["owned_cells"] == -3
