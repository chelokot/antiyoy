from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import UniversalPolicy


class BrowserPolicy(nn.Module):
    def __init__(
        self,
        policy: UniversalPolicy,
        rule_features: Tensor,
        width: int,
        height: int,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.width = width
        self.height = height
        self.register_buffer("rule_features", rule_features)

    def forward(
        self,
        playable: Tensor,
        visible: Tensor,
        owners: Tensor,
        objects: Tensor,
        unit_strengths: Tensor,
        ready: Tensor,
        defenses: Tensor,
        province_features: Tensor,
        active_player: Tensor,
        player_count: Tensor,
        round_number: Tensor,
        action_sources: Tensor,
        action_targets: Tensor,
        action_kinds: Tensor,
        action_parameters: Tensor,
    ) -> tuple[Tensor, Tensor]:
        owner_relation = torch.where(
            owners == 255,
            torch.zeros_like(owners),
            torch.where(owners == active_player, 1, 2),
        )
        turn_distance = torch.where(
            owners == 255,
            torch.zeros_like(owners),
            torch.remainder(owners - active_player, player_count) + 1,
        )
        relation_codes = torch.full_like(owners, 4)
        cells = (
            self.policy.owner_embedding(owner_relation)
            + self.policy.turn_distance_embedding(turn_distance)
            + self.policy.cell_relation_embedding(relation_codes)
            + self.policy.object_embedding(objects)
            + self.policy.unit_embedding(unit_strengths)
            + self.policy.ready_embedding(ready)
            + self.policy.defense_embedding(defenses)
            + self.policy.visibility_embedding(visible)
            + self.policy.province_projection(province_features)
        )
        grid_mask = playable.reshape(1, 1, self.height, self.width)
        grid = cells.reshape(1, self.height, self.width, self.policy.hidden).permute(
            0, 3, 1, 2
        )
        for block in self.policy.blocks:
            grid = block(grid, grid_mask)
        denominator = grid_mask.sum().clamp_min(1)
        pooled = (grid * grid_mask).sum(dim=(2, 3)) / denominator
        rules = self.policy.rule_projection(self.rule_features).reshape(
            1, self.policy.hidden
        )
        rounds = torch.log1p(round_number.to(dtype=torch.float32)).reshape(1, 1)
        context = (
            rules
            + self.policy.player_count_embedding(player_count).reshape(
                1, self.policy.hidden
            )
            + self.policy.round_projection(rounds)
        )
        global_features = pooled + context
        flat_cells = grid.permute(0, 2, 3, 1).reshape(
            self.width * self.height, self.policy.hidden
        )
        source_indices = action_sources.clamp(0, self.width * self.height - 1)
        source_present = (action_sources != 65535).reshape(-1, 1)
        source_features = torch.where(
            source_present,
            flat_cells[source_indices],
            self.policy.missing_source.reshape(1, -1),
        )
        target_indices = action_targets.clamp(0, self.width * self.height - 1)
        target_present = (action_targets != 65535).reshape(-1, 1)
        target_features = torch.where(
            target_present,
            flat_cells[target_indices],
            self.policy.missing_source.reshape(1, -1),
        )
        action_count = action_kinds.shape[0]
        repeated_global = global_features.expand(action_count, -1)
        action_features = torch.cat(
            (
                source_features,
                target_features,
                repeated_global,
                self.policy.action_kind_embedding(action_kinds),
                self.policy.action_parameter_embedding(action_parameters),
            ),
            dim=1,
        )
        logits = self.policy.action_head(action_features).squeeze(1)
        value = self.policy.value_head(torch.cat((pooled, context), dim=1)).squeeze(1)
        return logits, value
