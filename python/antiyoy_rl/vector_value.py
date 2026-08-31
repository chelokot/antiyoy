from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


MAX_PLAYERS = 8


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
                np.remainder(np.arange(count), count) - active_player
                for active_player, count in zip(active, counts, strict=True)
            ]
        ),
        dtype=torch.long,
        device=relative_utilities.device,
    )
    relative_indices.remainder_(
        torch.as_tensor(
            np.repeat(counts, counts),
            dtype=torch.long,
            device=relative_utilities.device,
        )
    )
    rows = torch.arange(
        counts.size, device=relative_utilities.device
    ).repeat_interleave(
        torch.as_tensor(counts, dtype=torch.long, device=relative_utilities.device)
    )
    return relative_utilities[rows, relative_indices]
