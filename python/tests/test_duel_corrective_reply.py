from __future__ import annotations

import numpy as np
import torch

from python.collect_duel_corrective_reply import collect_candidate


class OneTurnEnvironment:
    def __init__(self) -> None:
        self.chosen: int | None = None

    def done(self) -> np.ndarray:
        return np.asarray([False])

    def observe(self) -> dict[str, np.ndarray]:
        return {
            "active_players": np.asarray([1 if self.chosen is None else 0]),
            "action_kinds": np.asarray([0, 1]),
        }

    def search_actions_replanned(self, node_budget: int) -> np.ndarray:
        assert node_budget == 64
        return np.asarray([1])

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        self.chosen = int(actions[0])
        return {"truncated": np.asarray([False])}


class FirstActionPolicy:
    def __call__(
        self, observation: dict[str, np.ndarray], rules: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return torch.tensor([2.0, 1.0]), None


class CensoredTurnEnvironment(OneTurnEnvironment):
    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        self.chosen = int(actions[0])
        return {"truncated": np.asarray([True])}


def test_corrective_label_does_not_replace_student_rollout_action() -> None:
    environment = OneTurnEnvironment()

    states, labels, chosen, censored = collect_candidate(
        environment, FirstActionPolicy(), torch.zeros(1)
    )

    assert labels == [1]
    assert chosen == [0]
    assert environment.chosen == 0
    assert len(states) == 1
    assert not censored


def test_corrective_trace_keeps_local_legal_action_indices() -> None:
    environment = OneTurnEnvironment()

    states, labels, chosen, _ = collect_candidate(
        environment, FirstActionPolicy(), torch.zeros(1)
    )

    assert all(
        0 <= index < len(state["action_kinds"])
        for state, index in zip(states, labels, strict=True)
    )
    assert all(
        0 <= index < len(state["action_kinds"])
        for state, index in zip(states, chosen, strict=True)
    )


def test_corrective_trace_marks_action_limit_censoring() -> None:
    environment = CensoredTurnEnvironment()

    states, labels, chosen, censored = collect_candidate(
        environment, FirstActionPolicy(), torch.zeros(1)
    )

    assert len(states) == len(labels) == len(chosen) == 1
    assert censored
