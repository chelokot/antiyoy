from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor

from ._native import VectorEnv
from .model import action_distribution


class PolicySearchMetrics(TypedDict):
    leaf_batches: int
    evaluated_leaves: int
    nodes: np.ndarray
    completed_simulations: np.ndarray
    maximum_depth: np.ndarray
    root_visits: np.ndarray


@dataclass(frozen=True)
class PolicySearchConfig:
    node_budget: int = 256
    exploration: float = 1.5
    virtual_loss: float = 1.0
    maximum_depth: int = 128
    leaf_batch_size: int = 512


PolicyEvaluator = Callable[[Mapping[str, np.ndarray], Tensor], tuple[Tensor, Tensor]]


def policy_search_actions(
    environment: VectorEnv,
    evaluator: PolicyEvaluator,
    rule_features: Tensor,
    active_mask: np.ndarray,
    config: PolicySearchConfig,
) -> tuple[np.ndarray, PolicySearchMetrics]:
    active = np.asarray(active_mask, dtype=np.uint8)
    if active.shape != (environment.environments,):
        raise ValueError("active mask must contain one value per environment")
    search = environment.policy_search(
        node_budget=config.node_budget,
        exploration=config.exploration,
        virtual_loss=config.virtual_loss,
        maximum_depth=config.maximum_depth,
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
            logits, values = evaluator(observation, selected_rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            action_counts = np.diff(observation["action_offsets"])
            flat_priors = torch.cat(
                [
                    distribution.probs[index, : int(count)]
                    for index, count in enumerate(action_counts)
                ]
            )
            bounded_values = values.clamp(-1, 1)
        search.complete_leaves(
            flat_priors.to(device="cpu", dtype=torch.float32).numpy(),
            bounded_values.to(device="cpu", dtype=torch.float32).numpy(),
        )
        leaf_batches += 1
        evaluated_leaves += len(search_environments)
    stats = search.stats()
    return np.asarray(search.action_indices(), dtype=np.uint64), {
        "leaf_batches": leaf_batches,
        "evaluated_leaves": evaluated_leaves,
        "nodes": np.asarray(stats["nodes"], dtype=np.uint64),
        "completed_simulations": np.asarray(
            stats["completed_simulations"], dtype=np.uint64
        ),
        "maximum_depth": np.asarray(stats["maximum_depth"], dtype=np.uint64),
        "root_visits": np.asarray(stats["root_visits"], dtype=np.uint64),
    }
