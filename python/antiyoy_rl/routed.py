from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .model import UniversalPolicy, action_distribution, select_environments


class RoutedPolicy:
    def __init__(
        self,
        models: Mapping[str, UniversalPolicy],
        seat_experts: Sequence[str],
    ) -> None:
        if not seat_experts:
            raise ValueError("routed policy requires at least one seat")
        missing = set(seat_experts).difference(models)
        if missing:
            raise ValueError(
                f"routed policy is missing experts: {', '.join(sorted(missing))}"
            )
        self.models = dict(models)
        self.seat_experts = tuple(seat_experts)

    def __call__(
        self,
        observation: Mapping[str, np.ndarray],
        rule_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        environments_by_expert: dict[str, list[int]] = {}
        for environment, active_player in enumerate(observation["active_players"]):
            player = int(active_player)
            if player >= len(self.seat_experts):
                raise ValueError("active player has no routed expert")
            expert = self.seat_experts[player]
            environments_by_expert.setdefault(expert, []).append(environment)
        action_logits: dict[int, Tensor] = {}
        values: dict[int, Tensor] = {}
        for expert, environments in environments_by_expert.items():
            selected = select_environments(observation, environments)
            selected_rules = (
                rule_features
                if rule_features.ndim == 1
                else rule_features[environments]
            )
            expert_logits, expert_values = self.models[expert](
                selected, selected_rules
            )
            offsets = np.asarray(selected["action_offsets"], dtype=np.int64)
            for selected_index, environment in enumerate(environments):
                start, end = offsets[selected_index : selected_index + 2]
                action_logits[environment] = expert_logits[start:end]
                values[environment] = expert_values[selected_index]
        environment_count = len(observation["widths"])
        return (
            torch.cat([action_logits[index] for index in range(environment_count)]),
            torch.stack([values[index] for index in range(environment_count)]),
        )

    def actions(
        self,
        observation: Mapping[str, np.ndarray],
        rule_features: Tensor,
    ) -> np.ndarray:
        with torch.no_grad():
            logits, _ = self(observation, rule_features)
            distribution = action_distribution(logits, observation["action_offsets"])
        return distribution.logits.argmax(dim=1).cpu().numpy().astype(np.uint64)
