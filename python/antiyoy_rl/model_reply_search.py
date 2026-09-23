from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch

from ._native import VectorEnv
from .routed import RoutedPolicy


class ModelReplySearch:
    def __init__(
        self,
        node_budget: int = 256,
        slate_size: int = 8,
        beam_width: int = 32,
        branch_width: int = 48,
        maximum_actions_per_turn: int = 24,
    ) -> None:
        self.node_budget = node_budget
        self.slate_size = slate_size
        self.beam_width = beam_width
        self.branch_width = branch_width
        self.maximum_actions_per_turn = maximum_actions_per_turn
        self.pending: dict[int, tuple[int, deque[int]]] = {}
        self.root_turns = 0
        self.candidate_replies = 0
        self.opponent_actions = 0
        self.decision_times_seconds: list[float] = []

    def actions(
        self,
        environment: VectorEnv,
        policy: RoutedPolicy,
        rule_features: torch.Tensor,
        active: np.ndarray,
    ) -> np.ndarray:
        observation = environment.observe()
        active_players = observation["active_players"]
        if active.shape != active_players.shape:
            raise ValueError("active mask must match the environment batch")
        if any(environment.done()[index] for index in np.flatnonzero(active)):
            raise ValueError("model reply search cannot act in a finished game")

        needs_plan = active.copy()
        for index in self.pending:
            needs_plan[index] = False
        if bool(needs_plan.any()):
            search_started = time.perf_counter()
            plans, static_scores = environment.search_turn_plans(
                node_budget=self.node_budget,
                slate_size=self.slate_size,
                beam_width=self.beam_width,
                branch_width=self.branch_width,
                maximum_actions_per_turn=self.maximum_actions_per_turn,
                active_mask=needs_plan.astype(np.uint8),
            )
            search_seconds_per_root = (
                time.perf_counter() - search_started
            ) / int(needs_plan.sum())
            for index in np.flatnonzero(needs_plan):
                started = time.perf_counter()
                root = int(active_players[index])
                candidate_plans = plans[index]
                if not candidate_plans:
                    raise RuntimeError("native search returned no completed root turn")
                predicted = [
                    self._reply_score(
                        environment,
                        policy,
                        rule_features[index : index + 1],
                        int(index),
                        root,
                        plan,
                        static_score,
                    )
                    for plan, static_score in zip(
                        candidate_plans, static_scores[index]
                    )
                ]
                selected = max(
                    range(len(candidate_plans)),
                    key=lambda candidate: (
                        predicted[candidate],
                        static_scores[index][candidate],
                        -candidate,
                    ),
                )
                self.pending[int(index)] = (
                    root, deque(candidate_plans[selected])
                )
                self.root_turns += 1
                self.decision_times_seconds.append(
                    search_seconds_per_root + time.perf_counter() - started
                )

        selected_actions = np.zeros(len(active), dtype=np.uint64)
        action_offsets = observation["action_offsets"]
        for index in np.flatnonzero(active):
            root, plan = self.pending[int(index)]
            if int(active_players[index]) != root:
                raise RuntimeError("cached root turn belongs to another player")
            action = plan.popleft()
            if action >= int(action_offsets[index + 1] - action_offsets[index]):
                raise RuntimeError("cached root turn contains an illegal action index")
            selected_actions[index] = action
            if not plan:
                del self.pending[int(index)]
        return selected_actions

    def _reply_score(
        self,
        environment: VectorEnv,
        policy: RoutedPolicy,
        rule_features: torch.Tensor,
        index: int,
        root: int,
        plan: list[int],
        static_score: int,
    ) -> int:
        branch = environment.fork(np.asarray([index], dtype=np.uint64))
        for action in plan:
            result = branch.step(np.asarray([action], dtype=np.uint64))
            if bool(result["truncated"][0]):
                raise RuntimeError("root candidate reached the action limit")
        if int(branch.position_scores(root)[0]) != static_score:
            raise RuntimeError("indexed root plan changed its native static score")
        self.candidate_replies += 1
        if branch.done()[0]:
            return static_score
        opponent = int(branch.observe()["active_players"][0])
        if opponent == root:
            raise RuntimeError("root candidate did not complete its turn")
        for _ in range(self.maximum_actions_per_turn):
            action = int(policy.actions(branch.observe(), rule_features)[0])
            result = branch.step(np.asarray([action], dtype=np.uint64))
            self.opponent_actions += 1
            if bool(result["truncated"][0]):
                raise RuntimeError("simulated opponent reply reached the action limit")
            if branch.done()[0]:
                return int(branch.position_scores(root)[0])
            if int(branch.observe()["active_players"][0]) != opponent:
                return int(branch.position_scores(root)[0])
        raise RuntimeError("simulated opponent reply exceeded the turn depth limit")
