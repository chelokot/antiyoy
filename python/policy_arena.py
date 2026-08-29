from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import action_distribution, encode_rules_batch

try:
    from .evaluate import load_policy
except ImportError:
    from evaluate import load_policy


class PolicyArena:
    def __init__(
        self,
        checkpoint: Path,
        profile: str,
        width: int,
        height: int,
        seed: int,
        action_limit: int,
        device_name: str,
    ) -> None:
        self.checkpoint = checkpoint
        self.profile = profile
        self.width = width
        self.height = height
        self.action_limit = action_limit
        self.device = torch.device(device_name)
        self.model, _ = load_policy(checkpoint, self.device)
        self.environment = VectorEnv(
            1,
            width=width,
            height=height,
            seed=seed,
            action_limit=action_limit,
            profile=profile,
        )
        self.rules = encode_rules_batch(self.environment.rules_jsons(), self.device)
        self.seed = seed
        self.revision = 0
        self.winner: int | None = None
        self.terminal = False
        self.truncated = False
        self.lock = threading.Lock()

    def reset(self, seed: int) -> dict[str, object]:
        with self.lock:
            self.environment.reset(0, seed)
            self.seed = seed
            self.revision += 1
            self.winner = None
            self.terminal = False
            self.truncated = False
            return self._state()

    def state(self) -> dict[str, object]:
        with self.lock:
            return self._state()

    def act(self, action_index: int, expected_revision: int) -> dict[str, object]:
        with self.lock:
            if expected_revision != self.revision:
                raise ValueError(
                    f"stale revision {expected_revision}, current revision is {self.revision}"
                )
            if self.terminal or self.truncated:
                raise ValueError("the match is already finished")
            observation = self.environment.observe()
            if int(observation["active_players"][0]) != 0:
                raise ValueError("the neural policy is still moving")
            action_count = int(
                observation["action_offsets"][1] - observation["action_offsets"][0]
            )
            if action_index < 0 or action_index >= action_count:
                raise ValueError(
                    f"action index {action_index} is outside {action_count} legal actions"
                )
            self._step(action_index)
            while not self.terminal and not self.truncated:
                observation = self.environment.observe()
                if int(observation["active_players"][0]) == 0:
                    break
                self._step(self._policy_action(observation))
            return self._state()

    def _policy_action(self, observation: dict[str, np.ndarray]) -> int:
        with torch.inference_mode():
            logits, _ = self.model(observation, self.rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            return int(distribution.logits.argmax(dim=1)[0].item())

    def _step(self, action_index: int) -> None:
        result = self.environment.step(np.array([action_index], dtype=np.uint64))
        self.revision += 1
        self.terminal = bool(result["terminal"][0])
        self.truncated = bool(result["truncated"][0])
        winner = int(result["winners"][0])
        self.winner = None if winner == 255 else winner

    def _state(self) -> dict[str, object]:
        observation = self.environment.observe()
        cell_count = self.width * self.height
        province_count = int(observation["province_offsets"][1])
        action_start = int(observation["action_offsets"][0])
        action_end = int(observation["action_offsets"][1])
        cells = [
            {
                "id": index,
                "playable": bool(observation["playable"][index]),
                "visible": bool(observation["visible"][index]),
                "owner": None
                if int(observation["owners"][index]) == 255
                else int(observation["owners"][index]),
                "object": int(observation["objects"][index]),
                "unit": int(observation["unit_strengths"][index]),
                "ready": bool(observation["ready"][index]),
                "defense": int(observation["defenses"][index]),
                "province": None
                if int(observation["province_ids"][index]) == 65535
                else int(observation["province_ids"][index]),
            }
            for index in range(cell_count)
        ]
        provinces = [
            {
                "id": index,
                "owner": int(observation["province_owners"][index]),
                "money": int(observation["province_money"][index]),
                "profit": int(observation["province_profit"][index]),
                "capital": int(observation["province_capitals"][index]),
                "size": int(observation["province_sizes"][index]),
            }
            for index in range(province_count)
        ]
        actions = [
            {
                "index": local_index,
                "kind": int(observation["action_kinds"][global_index]),
                "source": self._optional_hex(
                    int(observation["action_sources"][global_index])
                ),
                "target": self._optional_hex(
                    int(observation["action_targets"][global_index])
                ),
                "parameter": int(observation["action_parameters"][global_index]),
            }
            for local_index, global_index in enumerate(range(action_start, action_end))
        ]
        return {
            "revision": self.revision,
            "checkpoint": self.checkpoint.name,
            "profile": self.profile,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "round": int(observation["rounds"][0]),
            "active_player": int(observation["active_players"][0]),
            "terminal": self.terminal,
            "truncated": self.truncated,
            "winner": self.winner,
            "cells": cells,
            "provinces": provinces,
            "actions": actions,
        }

    @staticmethod
    def _optional_hex(value: int) -> int | None:
        return None if value == 65535 else value
