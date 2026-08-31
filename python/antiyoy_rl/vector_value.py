from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn


MAX_PLAYERS = 8
VECTOR_VALUE_ARTIFACT_KIND = "relative_vector_value_head"
VECTOR_VALUE_ARTIFACT_VERSION = 1


class RelativeValueHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, MAX_PLAYERS),
        )

    def forward(self, value_features: Tensor) -> Tensor:
        return self.network(value_features)


def initialize_from_scalar_value_head(
    vector_head: RelativeValueHead, scalar_head: nn.Sequential
) -> None:
    scalar_input = scalar_head[0]
    scalar_output = scalar_head[2]
    if not isinstance(scalar_input, nn.Linear) or not isinstance(
        scalar_output, nn.Linear
    ):
        raise ValueError("scalar value head has an unsupported architecture")
    with torch.no_grad():
        vector_head.network[0].weight.copy_(scalar_input.weight)
        vector_head.network[0].bias.copy_(scalar_input.bias)
        vector_head.network[2].weight.copy_(
            scalar_output.weight.expand(MAX_PLAYERS, -1)
        )
        vector_head.network[2].bias.copy_(scalar_output.bias.expand(MAX_PLAYERS))


def load_relative_value_head(
    artifact: Mapping[str, object],
    hidden: int,
    device: torch.device,
) -> RelativeValueHead:
    if artifact.get("kind") != VECTOR_VALUE_ARTIFACT_KIND:
        raise ValueError("artifact is not a relative vector-value head")
    if artifact.get("artifact_version") != VECTOR_VALUE_ARTIFACT_VERSION:
        raise ValueError("vector-value artifact version is unsupported")
    architecture = artifact.get("architecture")
    if not isinstance(architecture, Mapping) or architecture.get("hidden") != hidden:
        raise ValueError("vector-value architecture does not match the policy")
    state = artifact.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("vector-value artifact has no model weights")
    head = RelativeValueHead(hidden).to(device)
    head.load_state_dict(state)
    head.eval()
    return head


def relative_to_absolute_utilities(
    relative_utilities: Tensor,
    active_players: np.ndarray,
    player_counts: np.ndarray,
) -> Tensor:
    active = np.asarray(active_players, dtype=np.int64)
    counts = np.asarray(player_counts, dtype=np.int64)
    if active.shape != counts.shape:
        raise ValueError("active players and player counts must have equal shapes")
    if relative_utilities.shape != (counts.size, MAX_PLAYERS):
        raise ValueError("relative utilities must contain eight values per leaf")
    if counts.size == 0 or np.any(counts < 2) or np.any(counts > MAX_PLAYERS):
        raise ValueError("player counts must be between two and eight")
    relative_indices = torch.as_tensor(
        np.concatenate(
            [
                np.remainder(np.arange(count) - active_player, count)
                for active_player, count in zip(active, counts, strict=True)
            ]
        ),
        dtype=torch.long,
        device=relative_utilities.device,
    )
    rows = torch.arange(
        counts.size, device=relative_utilities.device
    ).repeat_interleave(
        torch.as_tensor(counts, dtype=torch.long, device=relative_utilities.device)
    )
    return relative_utilities[rows, relative_indices]
