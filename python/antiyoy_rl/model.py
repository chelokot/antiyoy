from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional


RULE_FEATURES = 42


def encode_rules(serialized: str, device: torch.device) -> Tensor:
    rules = json.loads(serialized)
    economy = rules["economy"]
    combat = rules["combat"]
    vegetation = rules["vegetation"]
    lifecycle = rules["lifecycle"]
    values = [
        rules["minimum_province_size"],
        economy["starting_money"],
        economy["clear_hex_income"],
        economy["farm_hex_income"],
        economy["unit_price_per_level"],
        *economy["unit_upkeep"],
        economy["farm_base_price"],
        economy["farm_price_increment"],
        economy["tower_price"],
        economy["strong_tower_price"],
        economy["tower_upkeep"],
        economy["strong_tower_upkeep"],
        economy["planted_tree_price"],
        economy["tree_cut_reward"],
        combat["maximum_unit_strength"],
        combat["movement_range"],
        int(combat["strongest_unit_ignores_defense"]),
        int(combat["farms_enabled"]),
        int(combat["towers_enabled"]),
        int(combat["strong_towers_enabled"]),
        int(combat["tree_planting_enabled"]),
        int(combat["recruited_units_ready_on_owned_empty"]),
        int(combat["recruited_merge_preserves_readiness"]),
        int(combat["foreign_recruit_requires_economic_neighbour"]),
        int(vegetation["enabled"]),
        vegetation["pine_minimum_neighbours"],
        vegetation["pine_spread_per_million"] / 1_000_000,
        vegetation["palm_spread_per_million"] / 1_000_000,
        int(vegetation["target_based_spread"]),
        vegetation["target_spread_per_million"] / 1_000_000,
        int(vegetation["charge_player_zero_per_spawn"]),
        int(vegetation["grave_tree_skips_next_cycle"]),
        int(lifecycle["split_money_follows_capital_then_farms"]),
        int(lifecycle["merge_capital_prefers_farm_support"]),
        int(lifecycle["singleton_buildings_persist"]),
        int(lifecycle["eliminate_singleton_units_after_capture"]),
        int(lifecycle["skip_first_round_income"]),
        int(lifecycle["income_before_grave_conversion"]),
    ]
    if len(values) != RULE_FEATURES:
        raise ValueError(f"expected {RULE_FEATURES} rule features, received {len(values)}")
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sign(tensor) * torch.log1p(torch.abs(tensor))


class HexBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, hidden, 3, 3))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.projection = nn.Conv2d(hidden, hidden, 1)
        self.normalization = nn.GroupNorm(1, hidden)
        mask = torch.tensor(
            [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        self.register_buffer("mask", mask)
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, cells: Tensor, playable: Tensor) -> Tensor:
        mixed = functional.conv2d(cells, self.weight * self.mask, self.bias, padding=1)
        mixed = self.projection(functional.gelu(mixed))
        return functional.gelu(self.normalization(cells + mixed)) * playable


class UniversalPolicy(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.owner_embedding = nn.Embedding(3, hidden)
        self.object_embedding = nn.Embedding(8, hidden)
        self.unit_embedding = nn.Embedding(5, hidden)
        self.ready_embedding = nn.Embedding(2, hidden)
        self.defense_embedding = nn.Embedding(5, hidden)
        self.province_projection = nn.Linear(3, hidden)
        self.rule_projection = nn.Sequential(
            nn.Linear(RULE_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(HexBlock(hidden) for _ in range(layers))
        self.action_kind_embedding = nn.Embedding(5, hidden)
        self.action_parameter_embedding = nn.Embedding(5, hidden)
        self.missing_source = nn.Parameter(torch.zeros(hidden))
        self.action_head = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        observation: Mapping[str, np.ndarray],
        rule_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        device = self.missing_source.device
        widths = torch.as_tensor(observation["widths"], dtype=torch.long, device=device)
        heights = torch.as_tensor(observation["heights"], dtype=torch.long, device=device)
        if not torch.all(widths == widths[0]) or not torch.all(heights == heights[0]):
            raise ValueError("one policy forward pass requires uniform map dimensions")
        environments = widths.numel()
        width = int(widths[0].item())
        height = int(heights[0].item())
        cell_count = width * height
        active = torch.as_tensor(
            observation["active_players"], dtype=torch.long, device=device
        ).repeat_interleave(cell_count)
        owners = torch.as_tensor(observation["owners"], dtype=torch.long, device=device)
        owner_relation = torch.where(owners == 255, 0, torch.where(owners == active, 1, 2))
        objects = torch.as_tensor(observation["objects"], dtype=torch.long, device=device)
        units = torch.as_tensor(
            observation["unit_strengths"], dtype=torch.long, device=device
        )
        ready = torch.as_tensor(observation["ready"], dtype=torch.long, device=device)
        defenses = torch.as_tensor(observation["defenses"], dtype=torch.long, device=device)
        playable = torch.as_tensor(
            observation["playable"], dtype=torch.float32, device=device
        )
        cells = (
            self.owner_embedding(owner_relation)
            + self.object_embedding(objects)
            + self.unit_embedding(units)
            + self.ready_embedding(ready)
            + self.defense_embedding(defenses)
        )
        cells = cells + self._province_features(observation, environments, cell_count, device)
        grid_mask = playable.reshape(environments, 1, height, width)
        grid = cells.reshape(environments, height, width, self.hidden).permute(0, 3, 1, 2)
        for block in self.blocks:
            grid = block(grid, grid_mask)
        denominator = grid_mask.sum(dim=(2, 3)).clamp_min(1)
        pooled = (grid * grid_mask).sum(dim=(2, 3)) / denominator
        rules = self.rule_projection(rule_features).reshape(1, self.hidden).expand(environments, -1)
        global_features = pooled + rules
        flat_cells = grid.permute(0, 2, 3, 1).reshape(environments * cell_count, self.hidden)
        logits = self._action_logits(observation, flat_cells, global_features, cell_count, device)
        values = self.value_head(torch.cat((pooled, rules), dim=1)).squeeze(1)
        return logits, values

    def _province_features(
        self,
        observation: Mapping[str, np.ndarray],
        environments: int,
        cell_count: int,
        device: torch.device,
    ) -> Tensor:
        province_ids = torch.as_tensor(
            observation["province_ids"], dtype=torch.long, device=device
        )
        province_offsets = torch.as_tensor(
            observation["province_offsets"], dtype=torch.long, device=device
        )
        cell_environments = (
            torch.arange(environments, device=device)
            .reshape(-1, 1)
            .expand(-1, cell_count)
            .reshape(-1)
        )
        valid = province_ids != 65535
        local_ids = torch.where(valid, province_ids, torch.zeros_like(province_ids))
        global_ids = province_offsets[cell_environments] + local_ids
        money = torch.as_tensor(
            observation["province_money"], dtype=torch.float32, device=device
        )
        profit = torch.as_tensor(
            observation["province_profit"], dtype=torch.float32, device=device
        )
        sizes = torch.as_tensor(
            observation["province_sizes"], dtype=torch.float32, device=device
        )
        numeric = torch.zeros((environments * cell_count, 3), device=device)
        selected = global_ids[valid]
        values = torch.stack((money[selected], profit[selected], sizes[selected]), dim=1)
        numeric[valid] = torch.sign(values) * torch.log1p(torch.abs(values))
        return self.province_projection(numeric)

    def _action_logits(
        self,
        observation: Mapping[str, np.ndarray],
        cells: Tensor,
        global_features: Tensor,
        cell_count: int,
        device: torch.device,
    ) -> Tensor:
        offsets_numpy = np.asarray(observation["action_offsets"], dtype=np.int64)
        counts_numpy = np.diff(offsets_numpy)
        action_environments_numpy = np.repeat(
            np.arange(global_features.shape[0], dtype=np.int64), counts_numpy
        )
        action_environments = torch.as_tensor(
            action_environments_numpy, dtype=torch.long, device=device
        )
        sources = torch.as_tensor(
            observation["action_sources"], dtype=torch.long, device=device
        )
        targets = torch.as_tensor(
            observation["action_targets"], dtype=torch.long, device=device
        )
        source_features = self.missing_source.reshape(1, -1).expand(sources.numel(), -1).clone()
        valid_sources = sources != 65535
        source_indices = action_environments[valid_sources] * cell_count + sources[valid_sources]
        source_features[valid_sources] = cells[source_indices]
        target_features = self.missing_source.reshape(1, -1).expand(targets.numel(), -1).clone()
        valid_targets = targets != 65535
        target_indices = action_environments[valid_targets] * cell_count + targets[valid_targets]
        target_features[valid_targets] = cells[target_indices]
        kinds = torch.as_tensor(
            observation["action_kinds"], dtype=torch.long, device=device
        )
        parameters = torch.as_tensor(
            observation["action_parameters"], dtype=torch.long, device=device
        )
        features = torch.cat(
            (
                source_features,
                target_features,
                global_features[action_environments],
                self.action_kind_embedding(kinds),
                self.action_parameter_embedding(parameters),
            ),
            dim=1,
        )
        return self.action_head(features).squeeze(1)


def action_distribution(logits: Tensor, offsets: np.ndarray) -> torch.distributions.Categorical:
    device = logits.device
    boundaries_numpy = np.asarray(offsets, dtype=np.int64)
    counts_numpy = np.diff(boundaries_numpy)
    environments = counts_numpy.size
    maximum = int(counts_numpy.max())
    action_environments_numpy = np.repeat(
        np.arange(environments, dtype=np.int64), counts_numpy
    )
    positions_numpy = np.arange(logits.numel(), dtype=np.int64) - np.repeat(
        boundaries_numpy[:-1], counts_numpy
    )
    action_environments = torch.as_tensor(
        action_environments_numpy, dtype=torch.long, device=device
    )
    positions = torch.as_tensor(positions_numpy, dtype=torch.long, device=device)
    padded = torch.full((environments, maximum), -torch.inf, device=device)
    padded[action_environments, positions] = logits
    return torch.distributions.Categorical(logits=padded)
