from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from torch import Tensor

from ._native import VectorEnv


class PolicyActor(Protocol):
    def actions(
        self, observation: Mapping[str, np.ndarray], rule_features: Tensor
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class CandidateOutcome:
    action: int
    winner: int | None
    terminal: bool
    truncated: bool
    censored: bool
    territory: int
    rollout_steps: int


def rollout_candidates(
    environment: VectorEnv,
    source_index: int,
    action_indices: np.ndarray,
    policy: PolicyActor,
    rule_features: Tensor,
    horizon: int | None = None,
) -> list[CandidateOutcome]:
    actions = np.asarray(action_indices, dtype=np.uint64)
    if actions.ndim != 1 or actions.size == 0:
        raise ValueError("counterfactual actions must be a nonempty vector")
    if horizon is not None and horizon < 1:
        raise ValueError("counterfactual horizon must be positive")
    root_player = int(environment.observe()["active_players"][source_index])
    branches = environment.fork(np.full(actions.size, source_index, dtype=np.uint64))
    branch_rules = rule_features[source_index : source_index + 1].expand(
        actions.size, -1
    )
    branch_ids = np.arange(actions.size, dtype=np.int64)
    outcomes: list[CandidateOutcome | None] = [None] * actions.size
    rollout_steps = np.ones(actions.size, dtype=np.int32)
    selected = actions
    while True:
        result = branches.step(selected)
        done = np.logical_or(result["terminal"], result["truncated"])
        censored = np.logical_and(
            np.logical_not(done),
            rollout_steps[branch_ids] >= horizon if horizon is not None else False,
        )
        finished = np.logical_or(done, censored)
        if bool(finished.any()):
            observation = branches.observe()
            cell_offsets = np.asarray(observation["cell_offsets"], dtype=np.int64)
            owners = np.asarray(observation["owners"], dtype=np.uint8)
            for local_index in np.flatnonzero(finished):
                original_index = int(branch_ids[local_index])
                start, end = cell_offsets[local_index : local_index + 2]
                truncated = bool(result["truncated"][local_index])
                winner_key = "adjudicated_winners" if truncated else "winners"
                outcomes[original_index] = CandidateOutcome(
                    action=int(actions[original_index]),
                    winner=(
                        None
                        if censored[local_index]
                        else int(result[winner_key][local_index])
                    ),
                    terminal=bool(result["terminal"][local_index]),
                    truncated=truncated,
                    censored=bool(censored[local_index]),
                    territory=int(np.count_nonzero(owners[start:end] == root_player)),
                    rollout_steps=int(rollout_steps[original_index]),
                )
        if bool(finished.all()):
            return [outcome for outcome in outcomes if outcome is not None]
        remaining = np.flatnonzero(np.logical_not(finished))
        branches = branches.fork(remaining.astype(np.uint64))
        branch_ids = branch_ids[remaining]
        branch_rules = branch_rules[remaining]
        rollout_steps[branch_ids] += 1
        selected = policy.actions(branches.observe(), branch_rules)
