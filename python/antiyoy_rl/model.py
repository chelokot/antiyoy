from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional


RULE_FEATURES = 45
ACTION_KIND_NAMES = (
    "end_turn",
    "move",
    "recruit",
    "build",
    "plant_tree",
    "diplomacy",
)
CELL_FEATURES = (
    "playable",
    "visible",
    "owners",
    "objects",
    "unit_strengths",
    "ready",
    "defenses",
    "province_ids",
)


def domain_key(generator: str, descriptor: Mapping[str, object]) -> str:
    if "seed" in descriptor:
        raise ValueError("domain descriptors must not contain a seed")
    payload = json.dumps(
        {"generator": generator, "descriptor": descriptor},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def rotate_observation_180(
    observation: Mapping[str, np.ndarray],
    rotation_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    environments = len(observation["widths"])
    selected = (
        np.ones(environments, dtype=np.bool_)
        if rotation_mask is None
        else np.asarray(rotation_mask, dtype=np.bool_)
    )
    if selected.shape != (environments,):
        raise ValueError("rotation mask must contain one value per environment")
    rotated = dict(observation)
    cell_offsets = np.asarray(observation["cell_offsets"], dtype=np.int64)
    action_offsets = np.asarray(observation["action_offsets"], dtype=np.int64)
    province_offsets = np.asarray(observation["province_offsets"], dtype=np.int64)
    for key in CELL_FEATURES:
        values = np.asarray(observation[key])
        transformed = values.copy()
        for environment in np.flatnonzero(selected):
            start, end = cell_offsets[environment : environment + 2]
            transformed[start:end] = values[start:end][::-1]
        rotated[key] = transformed
    sources = np.asarray(observation["action_sources"])
    targets = np.asarray(observation["action_targets"])
    kinds = np.asarray(observation["action_kinds"])
    rotated_sources = sources.copy()
    rotated_targets = targets.copy()
    capitals = np.asarray(observation["province_capitals"])
    rotated_capitals = capitals.copy()
    for environment in np.flatnonzero(selected):
        cell_count = int(cell_offsets[environment + 1] - cell_offsets[environment])
        action_start, action_end = action_offsets[environment : environment + 2]
        action_slice = slice(action_start, action_end)
        source_values = sources[action_slice]
        valid_sources = source_values != 65535
        source_indices = action_start + np.flatnonzero(valid_sources)
        rotated_sources[source_indices] = cell_count - 1 - source_values[valid_sources]
        target_values = targets[action_slice]
        valid_targets = np.logical_and(
            target_values != 65535,
            kinds[action_slice] != 5,
        )
        target_indices = action_start + np.flatnonzero(valid_targets)
        rotated_targets[target_indices] = cell_count - 1 - target_values[valid_targets]
        province_start, province_end = province_offsets[environment : environment + 2]
        province_slice = slice(province_start, province_end)
        capital_values = capitals[province_slice]
        valid_capitals = capital_values != 65535
        capital_indices = province_start + np.flatnonzero(valid_capitals)
        rotated_capitals[capital_indices] = (
            cell_count - 1 - capital_values[valid_capitals]
        )
    rotated["action_sources"] = rotated_sources
    rotated["action_targets"] = rotated_targets
    rotated["province_capitals"] = rotated_capitals
    return rotated


def encode_rules(serialized: str, device: torch.device) -> Tensor:
    return encode_rules_batch([serialized], device)[0]


def encode_rules_batch(serialized: list[str], device: torch.device) -> Tensor:
    values = [_rule_values(value) for value in serialized]
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sign(tensor) * torch.log1p(torch.abs(tensor))


def _rule_values(serialized: str) -> list[float]:
    rules = json.loads(serialized)
    economy = rules["economy"]
    combat = rules["combat"]
    vegetation = rules["vegetation"]
    lifecycle = rules["lifecycle"]
    diplomacy = rules["diplomacy"]
    relation_codes = {"War": 0, "Neutral": 1, "Friend": 2, "Alliance": 3}
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
        int(diplomacy["enabled"]),
        relation_codes[diplomacy["initial_relation"]],
        int(diplomacy["alliances_share_wars"]),
    ]
    if len(values) != RULE_FEATURES:
        raise ValueError(
            f"expected {RULE_FEATURES} rule features, received {len(values)}"
        )
    return values


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
        self.visibility_embedding = nn.Embedding(2, hidden)
        self.province_projection = nn.Linear(3, hidden)
        self.rule_projection = nn.Sequential(
            nn.Linear(RULE_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(HexBlock(hidden) for _ in range(layers))
        self.action_kind_embedding = nn.Embedding(len(ACTION_KIND_NAMES), hidden)
        self.action_parameter_embedding = nn.Embedding(6, hidden)
        self.relation_embedding = nn.Embedding(4, hidden)
        self.proposal_embedding = nn.Embedding(5, hidden)
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
        widths = np.asarray(observation["widths"], dtype=np.int64)
        heights = np.asarray(observation["heights"], dtype=np.int64)
        dimensions: dict[tuple[int, int], list[int]] = {}
        for environment, size in enumerate(zip(widths, heights, strict=True)):
            dimensions.setdefault((int(size[0]), int(size[1])), []).append(environment)
        if len(dimensions) == 1:
            return self._forward_uniform(observation, rule_features)
        if rule_features.ndim != 1 and rule_features.shape[0] != widths.size:
            raise ValueError("rule feature rows must match the environment count")
        action_logits: dict[int, Tensor] = {}
        values: dict[int, Tensor] = {}
        for environments in dimensions.values():
            selected = select_environments(observation, environments)
            selected_rules = (
                rule_features
                if rule_features.ndim == 1
                else rule_features[environments]
            )
            group_logits, group_values = self._forward_uniform(selected, selected_rules)
            group_offsets = np.asarray(selected["action_offsets"], dtype=np.int64)
            for group_index, environment in enumerate(environments):
                start, end = group_offsets[group_index : group_index + 2]
                action_logits[environment] = group_logits[start:end]
                values[environment] = group_values[group_index]
        return (
            torch.cat([action_logits[index] for index in range(widths.size)]),
            torch.stack([values[index] for index in range(widths.size)]),
        )

    def _forward_uniform(
        self,
        observation: Mapping[str, np.ndarray],
        rule_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        device = self.missing_source.device
        widths = torch.as_tensor(observation["widths"], dtype=torch.long, device=device)
        heights = torch.as_tensor(
            observation["heights"], dtype=torch.long, device=device
        )
        environments = widths.numel()
        width = int(widths[0].item())
        height = int(heights[0].item())
        cell_count = width * height
        active = torch.as_tensor(
            observation["active_players"], dtype=torch.long, device=device
        ).repeat_interleave(cell_count)
        owners = torch.as_tensor(observation["owners"], dtype=torch.long, device=device)
        owner_relation = torch.where(
            owners == 255, 0, torch.where(owners == active, 1, 2)
        )
        objects = torch.as_tensor(
            observation["objects"], dtype=torch.long, device=device
        )
        units = torch.as_tensor(
            observation["unit_strengths"], dtype=torch.long, device=device
        )
        ready = torch.as_tensor(observation["ready"], dtype=torch.long, device=device)
        defenses = torch.as_tensor(
            observation["defenses"], dtype=torch.long, device=device
        )
        visible = torch.as_tensor(
            observation["visible"], dtype=torch.long, device=device
        )
        playable = torch.as_tensor(
            observation["playable"], dtype=torch.float32, device=device
        )
        cells = (
            self.owner_embedding(owner_relation)
            + self.object_embedding(objects)
            + self.unit_embedding(units)
            + self.ready_embedding(ready)
            + self.defense_embedding(defenses)
            + self.visibility_embedding(visible)
        )
        cells = cells + self._province_features(
            observation, environments, cell_count, device
        )
        grid_mask = playable.reshape(environments, 1, height, width)
        grid = cells.reshape(environments, height, width, self.hidden).permute(
            0, 3, 1, 2
        )
        for block in self.blocks:
            grid = block(grid, grid_mask)
        denominator = grid_mask.sum(dim=(2, 3)).clamp_min(1)
        pooled = (grid * grid_mask).sum(dim=(2, 3)) / denominator
        rules = self.rule_projection(rule_features)
        if rules.ndim == 1:
            rules = rules.reshape(1, self.hidden).expand(environments, -1)
        elif rules.shape[0] != environments:
            raise ValueError("rule feature rows must match the environment count")
        diplomacy = self._diplomacy_context(observation, rule_features, device)
        context = rules + diplomacy
        global_features = pooled + context
        flat_cells = grid.permute(0, 2, 3, 1).reshape(
            environments * cell_count, self.hidden
        )
        logits = self._action_logits(
            observation, flat_cells, global_features, cell_count, device
        )
        values = self.value_head(torch.cat((pooled, context), dim=1)).squeeze(1)
        return logits, values

    def _diplomacy_context(
        self,
        observation: Mapping[str, np.ndarray],
        rule_features: Tensor,
        device: torch.device,
    ) -> Tensor:
        relation_offsets = np.asarray(observation["relation_offsets"], dtype=np.int64)
        player_counts = np.asarray(observation["player_counts"], dtype=np.int64)
        active_players = np.asarray(observation["active_players"], dtype=np.int64)
        enabled_features = (
            rule_features[42].expand(player_counts.size)
            if rule_features.ndim == 1
            else rule_features[:, 42]
        )
        enabled = enabled_features > 0
        if not bool(torch.any(enabled).item()):
            return torch.zeros((player_counts.size, self.hidden), device=device)
        relations = torch.as_tensor(
            observation["relations"], dtype=torch.long, device=device
        )
        proposals = torch.as_tensor(
            observation["proposals"], dtype=torch.long, device=device
        )
        present = np.flatnonzero(np.diff(relation_offsets) > 0)
        environment_indices_numpy = np.repeat(
            present.astype(np.int64), player_counts[present]
        )
        outgoing_numpy = np.concatenate(
            [
                int(relation_offsets[environment])
                + int(active) * int(players)
                + np.arange(int(players), dtype=np.int64)
                for environment, players, active in zip(
                    present,
                    player_counts[present],
                    active_players[present],
                    strict=True,
                )
            ]
        )
        incoming_numpy = np.concatenate(
            [
                int(relation_offsets[environment])
                + np.arange(int(players), dtype=np.int64) * int(players)
                + int(active)
                for environment, players, active in zip(
                    present,
                    player_counts[present],
                    active_players[present],
                    strict=True,
                )
            ]
        )
        environment_indices = torch.as_tensor(
            environment_indices_numpy, dtype=torch.long, device=device
        )
        present_indices = torch.as_tensor(present, dtype=torch.long, device=device)
        outgoing = torch.as_tensor(outgoing_numpy, dtype=torch.long, device=device)
        incoming = torch.as_tensor(incoming_numpy, dtype=torch.long, device=device)
        outgoing_proposals = torch.where(
            proposals[outgoing] == 255, 4, proposals[outgoing]
        )
        incoming_proposals = torch.where(
            proposals[incoming] == 255, 4, proposals[incoming]
        )
        vectors = (
            self.relation_embedding(relations[outgoing])
            + self.proposal_embedding(outgoing_proposals)
            + self.proposal_embedding(incoming_proposals)
        )
        contexts = torch.zeros((player_counts.size, self.hidden), device=device)
        contexts.index_add_(0, environment_indices, vectors)
        divisors = torch.ones((player_counts.size, 1), device=device)
        divisors[present_indices] = torch.as_tensor(
            player_counts[present], dtype=torch.float32, device=device
        ).reshape(-1, 1)
        contexts /= divisors
        return contexts * enabled.to(dtype=torch.float32).reshape(-1, 1)

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
        values = torch.stack(
            (money[selected], profit[selected], sizes[selected]), dim=1
        )
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
        kinds = torch.as_tensor(
            observation["action_kinds"], dtype=torch.long, device=device
        )
        source_features = (
            self.missing_source.reshape(1, -1).expand(sources.numel(), -1).clone()
        )
        valid_sources = sources != 65535
        source_indices = (
            action_environments[valid_sources] * cell_count + sources[valid_sources]
        )
        source_features[valid_sources] = cells[source_indices]
        target_features = (
            self.missing_source.reshape(1, -1).expand(targets.numel(), -1).clone()
        )
        diplomacy_actions = kinds == 5
        valid_targets = torch.logical_and(targets != 65535, ~diplomacy_actions)
        target_indices = (
            action_environments[valid_targets] * cell_count + targets[valid_targets]
        )
        target_features[valid_targets] = cells[target_indices]
        if torch.any(diplomacy_actions):
            target_features[diplomacy_actions] = self._diplomacy_action_targets(
                observation,
                action_environments[diplomacy_actions],
                targets[diplomacy_actions],
                device,
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

    def _diplomacy_action_targets(
        self,
        observation: Mapping[str, np.ndarray],
        environments: Tensor,
        targets: Tensor,
        device: torch.device,
    ) -> Tensor:
        relations = torch.as_tensor(
            observation["relations"], dtype=torch.long, device=device
        )
        proposals = torch.as_tensor(
            observation["proposals"], dtype=torch.long, device=device
        )
        offsets = torch.as_tensor(
            observation["relation_offsets"], dtype=torch.long, device=device
        )
        player_counts = torch.as_tensor(
            observation["player_counts"], dtype=torch.long, device=device
        )[environments]
        active = torch.as_tensor(
            observation["active_players"], dtype=torch.long, device=device
        )[environments]
        bases = offsets[environments]
        outgoing = bases + active * player_counts + targets
        incoming = bases + targets * player_counts + active
        outgoing_proposals = torch.where(
            proposals[outgoing] == 255, 4, proposals[outgoing]
        )
        incoming_proposals = torch.where(
            proposals[incoming] == 255, 4, proposals[incoming]
        )
        return (
            self.relation_embedding(relations[outgoing])
            + self.proposal_embedding(outgoing_proposals)
            + self.proposal_embedding(incoming_proposals)
        )


def load_policy_state(model: UniversalPolicy, state: Mapping[str, Tensor]) -> None:
    migrated = dict(state)
    current = model.state_dict()
    for key in ("action_kind_embedding.weight", "action_parameter_embedding.weight"):
        if migrated[key].shape != current[key].shape:
            expanded = torch.zeros_like(current[key])
            expanded[: migrated[key].shape[0]] = migrated[key]
            migrated[key] = expanded
    rule_key = "rule_projection.0.weight"
    if migrated[rule_key].shape != current[rule_key].shape:
        expanded = torch.zeros_like(current[rule_key])
        expanded[:, : migrated[rule_key].shape[1]] = migrated[rule_key]
        migrated[rule_key] = expanded
    for key in ("relation_embedding.weight", "proposal_embedding.weight"):
        if key not in migrated:
            migrated[key] = current[key]
    model.load_state_dict(migrated)


def select_environments(
    observation: Mapping[str, np.ndarray],
    environments: list[int],
) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    offset_groups = (
        (
            "cell_offsets",
            (
                "playable",
                "visible",
                "owners",
                "objects",
                "unit_strengths",
                "ready",
                "defenses",
                "province_ids",
            ),
        ),
        (
            "province_offsets",
            (
                "province_owners",
                "province_money",
                "province_profit",
                "province_capitals",
                "province_sizes",
            ),
        ),
        (
            "action_offsets",
            (
                "action_kinds",
                "action_sources",
                "action_targets",
                "action_parameters",
            ),
        ),
        ("relation_offsets", ("relations", "proposals")),
    )
    for offsets_key, value_keys in offset_groups:
        offsets = np.asarray(observation[offsets_key], dtype=np.int64)
        slices = [slice(offsets[index], offsets[index + 1]) for index in environments]
        counts = [value.stop - value.start for value in slices]
        selected[offsets_key] = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        )
        for key in value_keys:
            values = np.asarray(observation[key])
            selected[key] = np.concatenate([values[value] for value in slices])
    for key in ("widths", "heights", "active_players", "player_counts", "rounds"):
        selected[key] = np.asarray(observation[key])[environments]
    return selected


def concatenate_observations(
    observations: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not observations:
        raise ValueError("at least one observation is required")
    concatenated: dict[str, np.ndarray] = {}
    offset_groups = (
        (
            "cell_offsets",
            (
                "playable",
                "visible",
                "owners",
                "objects",
                "unit_strengths",
                "ready",
                "defenses",
                "province_ids",
            ),
        ),
        (
            "province_offsets",
            (
                "province_owners",
                "province_money",
                "province_profit",
                "province_capitals",
                "province_sizes",
            ),
        ),
        (
            "action_offsets",
            (
                "action_kinds",
                "action_sources",
                "action_targets",
                "action_parameters",
            ),
        ),
        ("relation_offsets", ("relations", "proposals")),
    )
    for offsets_key, value_keys in offset_groups:
        counts = np.concatenate(
            [
                np.diff(np.asarray(observation[offsets_key], dtype=np.int64))
                for observation in observations
            ]
        )
        concatenated[offsets_key] = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        )
        for key in value_keys:
            concatenated[key] = np.concatenate(
                [np.asarray(observation[key]) for observation in observations]
            )
    for key in ("widths", "heights", "active_players", "player_counts", "rounds"):
        concatenated[key] = np.concatenate(
            [np.asarray(observation[key]) for observation in observations]
        )
    return concatenated


def action_distribution(
    logits: Tensor, offsets: np.ndarray
) -> torch.distributions.Categorical:
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
