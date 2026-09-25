import numpy as np
import pytest
import torch

from python import audit_duel_four_turn_replanned_prefix as audit


class FakeBranch:
    def __init__(self, root_actions: int) -> None:
        self.root_actions = root_actions
        self.active = 0
        self.completed_root_actions = 0
        self.selected: list[tuple[int, int]] = []
        self.finished = False

    def done(self) -> np.ndarray:
        return np.asarray([self.finished])

    def observe(self) -> dict[str, np.ndarray]:
        return {"active_players": np.asarray([self.active])}

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        self.selected.append((self.active, int(action[0])))
        if self.active == 0:
            self.completed_root_actions += 1
            if self.completed_root_actions == self.root_actions:
                self.finished = True
            else:
                self.active = 1
        else:
            self.active = 0
        return {
            "terminal": np.asarray([self.finished]),
            "truncated": np.asarray([False]),
            "winners": np.asarray([0]),
            "adjudicated_winners": np.asarray([0]),
        }


class FakeSource:
    def actions(
        self, observation: dict[str, np.ndarray], rules: torch.Tensor
    ) -> np.ndarray:
        return np.asarray([7], dtype=np.uint64)


@pytest.mark.parametrize(
    ("prefix", "expected_root_actions"),
    [("source", [7, 7, 7, 7, 8]), ("teacher", [8, 8, 8, 8, 8])],
)
def test_four_turn_prefix_switches_only_after_four_completed_root_turns(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
    expected_root_actions: list[int],
) -> None:
    monkeypatch.setattr(
        audit,
        "native_teacher_action",
        lambda branch, followup_nodes, replan: np.asarray([8], dtype=np.uint64),
    )
    branch = FakeBranch(root_actions=5)

    outcome, completed, prefix_actions = audit.run_branch(
        branch,
        0,
        FakeSource(),
        torch.empty(0),
        expected_root_actions[0],
        prefix,
    )

    assert [
        action for seat, action in branch.selected if seat == 0
    ] == expected_root_actions
    assert [action for seat, action in branch.selected if seat == 1] == [7] * 4
    assert completed == 4
    assert prefix_actions == 4
    assert outcome["terminal"]
    assert outcome["winner"] == 0
    assert outcome["actions_after_intervention"] == 9


def test_four_turn_prefix_reports_early_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "native_teacher_action",
        lambda branch, followup_nodes, replan: np.asarray([8], dtype=np.uint64),
    )
    branch = FakeBranch(root_actions=2)

    _, completed, prefix_actions = audit.run_branch(
        branch, 0, FakeSource(), torch.empty(0), 7, "source"
    )

    assert completed == 2
    assert prefix_actions == 2
