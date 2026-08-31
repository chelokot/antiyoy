from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

import numpy as np
import torch
from torch import Tensor

from ._native import VectorEnv
from .model import action_distribution, concatenate_observations, select_environments


class PolicySearchMetrics(TypedDict):
    leaf_batches: int
    evaluated_leaves: int
    nodes: np.ndarray
    completed_simulations: np.ndarray
    maximum_depth: np.ndarray
    root_visits: np.ndarray
    root_action_offsets: NotRequired[np.ndarray]
    root_probabilities: NotRequired[np.ndarray]


ValuePerspective = Literal["active", "root"]
OpponentHorizon = Literal["search", "leaf"]
SearchObjective = Literal["scalar", "maxn"]
MAXN_PLAYERS = 8


@dataclass(frozen=True)
class PolicySearchConfig:
    node_budget: int = 256
    exploration: float = 1.5
    virtual_loss: float = 1.0
    maximum_depth: int = 128
    root_value_weight: float | None = None
    leaf_batch_size: int = 512
    value_perspective: ValuePerspective = "active"
    opponent_horizon: OpponentHorizon = "search"
    objective: SearchObjective = "scalar"


PolicyEvaluator = Callable[[Mapping[str, np.ndarray], Tensor], tuple[Tensor, Tensor]]
MaxNEvaluator = Callable[[Mapping[str, np.ndarray], Tensor], tuple[Tensor, Tensor]]


def root_perspective_leaf_values(
    evaluator: PolicyEvaluator,
    observation: Mapping[str, np.ndarray],
    rule_features: Tensor,
    active_values: Tensor,
    root_players: np.ndarray,
) -> Tensor:
    active_players = np.asarray(observation["active_players"])
    roots = np.asarray(root_players, dtype=active_players.dtype)
    if roots.shape != active_players.shape:
        raise ValueError("root players must contain one value per search leaf")
    if np.array_equal(active_players, roots):
        return active_values
    root_observation = dict(observation)
    root_observation["active_players"] = roots
    with torch.no_grad():
        _, root_values = evaluator(root_observation, rule_features)
    native_signs = torch.as_tensor(
        np.where(active_players == roots, 1.0, -1.0),
        dtype=root_values.dtype,
        device=root_values.device,
    )
    return root_values * native_signs


def maxn_leaf_utilities(
    evaluator: PolicyEvaluator,
    observation: Mapping[str, np.ndarray],
    rule_features: Tensor,
    active_values: Tensor,
) -> Tensor:
    active_players = np.asarray(observation["active_players"], dtype=np.int64)
    player_counts = np.asarray(observation["player_counts"], dtype=np.int64)
    if active_players.shape != player_counts.shape:
        raise ValueError("active players and player counts must have equal shapes")
    if player_counts.size == 0 or np.any(player_counts < 2):
        raise ValueError("MaxN requires at least two players per leaf")
    if np.any(player_counts > MAXN_PLAYERS):
        raise ValueError(f"MaxN supports at most {MAXN_PLAYERS} players")
    if active_values.shape != torch.Size((player_counts.size,)):
        raise ValueError("active values must contain one value per search leaf")
    utility_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(player_counts, dtype=np.int64))
    )
    utilities = torch.empty(
        int(utility_offsets[-1]),
        dtype=active_values.dtype,
        device=active_values.device,
    )
    active_indices = torch.as_tensor(
        utility_offsets[:-1] + active_players,
        dtype=torch.long,
        device=active_values.device,
    )
    utilities[active_indices] = active_values
    perspective_observations: list[dict[str, np.ndarray]] = []
    perspective_rules: list[Tensor] = []
    perspective_targets: list[np.ndarray] = []
    for player in range(int(player_counts.max())):
        environments = np.flatnonzero(
            np.logical_and(player_counts > player, active_players != player)
        )
        if environments.size == 0:
            continue
        selected = select_environments(observation, environments.tolist())
        selected["active_players"] = np.full(environments.size, player, dtype=np.uint8)
        perspective_observations.append(selected)
        perspective_rules.append(
            rule_features.reshape(1, -1).expand(environments.size, -1)
            if rule_features.ndim == 1
            else rule_features[environments]
        )
        perspective_targets.append(utility_offsets[environments] + player)
    combined_observation = concatenate_observations(perspective_observations)
    combined_rules = torch.cat(perspective_rules)
    with torch.no_grad():
        _, perspective_values = evaluator(combined_observation, combined_rules)
    targets = torch.as_tensor(
        np.concatenate(perspective_targets),
        dtype=torch.long,
        device=active_values.device,
    )
    utilities[targets] = perspective_values
    return utilities


def policy_search_actions(
    environment: VectorEnv,
    evaluator: PolicyEvaluator,
    rule_features: Tensor,
    active_mask: np.ndarray,
    config: PolicySearchConfig,
    include_root_targets: bool = False,
    maxn_evaluator: MaxNEvaluator | None = None,
) -> tuple[np.ndarray, PolicySearchMetrics]:
    active = np.asarray(active_mask, dtype=np.uint8)
    if active.shape != (environment.environments,):
        raise ValueError("active mask must contain one value per environment")
    if config.value_perspective not in ("active", "root"):
        raise ValueError("PUCT value perspective must be active or root")
    if config.opponent_horizon not in ("search", "leaf"):
        raise ValueError("PUCT opponent horizon must be search or leaf")
    if config.objective not in ("scalar", "maxn"):
        raise ValueError("PUCT objective must be scalar or maxn")
    if config.objective == "maxn" and config.value_perspective != "active":
        raise ValueError(
            "MaxN evaluates every player and requires active perspective mode"
        )
    root_players = np.asarray(environment.observe()["active_players"], dtype=np.uint8)
    search = environment.policy_search(
        node_budget=config.node_budget,
        exploration=config.exploration,
        virtual_loss=config.virtual_loss,
        maximum_depth=config.maximum_depth,
        root_value_weight=config.root_value_weight,
        search_opponent_turns=config.opponent_horizon == "search",
        maxn=config.objective == "maxn",
        active_mask=active,
    )
    leaf_batches = 0
    evaluated_leaves = 0
    while not search.is_complete():
        observation = search.select_leaves(config.leaf_batch_size)
        search_environments = np.asarray(
            observation["search_environments"], dtype=np.int64
        )
        if search_environments.size == 0:
            continue
        selected_rules = rule_features[search_environments]
        with torch.no_grad():
            if config.objective == "maxn" and maxn_evaluator is not None:
                logits, bounded_utilities = maxn_evaluator(observation, selected_rules)
                bounded_utilities = bounded_utilities.clamp(-1, 1)
                values = None
            else:
                logits, values = evaluator(observation, selected_rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            action_counts = np.diff(observation["action_offsets"])
            flat_priors = torch.cat(
                [
                    distribution.probs[index, : int(count)]
                    for index, count in enumerate(action_counts)
                ]
            )
            if config.objective == "maxn" and maxn_evaluator is None:
                assert values is not None
                bounded_utilities = maxn_leaf_utilities(
                    evaluator, observation, selected_rules, values
                ).clamp(-1, 1)
            elif config.objective == "scalar":
                assert values is not None
                leaf_values = (
                    root_perspective_leaf_values(
                        evaluator,
                        observation,
                        selected_rules,
                        values,
                        root_players[search_environments],
                    )
                    if config.value_perspective == "root"
                    else values
                )
                bounded_values = leaf_values.clamp(-1, 1)
        cpu_priors = flat_priors.to(device="cpu", dtype=torch.float32).numpy()
        if config.objective == "maxn":
            search.complete_maxn_leaves(
                cpu_priors,
                bounded_utilities.to(device="cpu", dtype=torch.float32).numpy(),
            )
        else:
            search.complete_leaves(
                cpu_priors,
                bounded_values.to(device="cpu", dtype=torch.float32).numpy(),
            )
        leaf_batches += 1
        evaluated_leaves += len(search_environments)
    stats = search.stats()
    metrics: PolicySearchMetrics = {
        "leaf_batches": leaf_batches,
        "evaluated_leaves": evaluated_leaves,
        "nodes": np.asarray(stats["nodes"], dtype=np.uint64),
        "completed_simulations": np.asarray(
            stats["completed_simulations"], dtype=np.uint64
        ),
        "maximum_depth": np.asarray(stats["maximum_depth"], dtype=np.uint64),
        "root_visits": np.asarray(stats["root_visits"], dtype=np.uint64),
    }
    if include_root_targets:
        root_targets = search.root_targets()
        metrics["root_action_offsets"] = np.asarray(
            root_targets["offsets"], dtype=np.uint64
        )
        metrics["root_probabilities"] = np.asarray(
            root_targets["probabilities"], dtype=np.float32
        )
    return np.asarray(search.action_indices(), dtype=np.uint64), metrics
