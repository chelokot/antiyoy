from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl.counterfactual import CandidateOutcome, rollout_candidates
from antiyoy_rl.model import action_distribution, encode_rules_batch

try:
    from .build_bundle import digest
    from .collect_action_q import (
        add_collection_arguments,
        collection_config,
        load_collection_policy,
    )
    from .distill_puct import (
        PuctDistillationConfig,
        create_environment,
        domain_descriptor,
        validate_config,
    )
    from .evaluate import load_policy_checkpoint, winner_score
except ImportError:
    from build_bundle import digest
    from collect_action_q import (
        add_collection_arguments,
        collection_config,
        load_collection_policy,
    )
    from distill_puct import (
        PuctDistillationConfig,
        create_environment,
        domain_descriptor,
        validate_config,
    )
    from evaluate import load_policy_checkpoint, winner_score


def outcome_record(outcome: CandidateOutcome, root_player: int) -> dict[str, object]:
    return {
        **asdict(outcome),
        "root_score": (
            None
            if outcome.winner is None
            else winner_score(outcome.winner, root_player)
        ),
    }


def benchmark(
    checkpoint_path: Path,
    config: PuctDistillationConfig,
    label_stride: int,
    candidate_count: int,
    training_seat: int | None,
    rollout_horizon: int | None,
) -> dict[str, object]:
    validate_config(config)
    if label_stride < 1 or candidate_count < 2:
        raise ValueError(
            "label stride must be positive and candidate count at least two"
        )
    if rollout_horizon is not None and rollout_horizon < 1:
        raise ValueError("rollout horizon must be positive")
    if training_seat is not None and not 0 <= training_seat < config.players:
        raise ValueError("training seat is outside the player range")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    descriptor = domain_descriptor(config, dict(checkpoint["config"]))
    policy, selected_experts, _, _ = load_collection_policy(
        checkpoint, config, descriptor, device
    )
    environment = create_environment(config, dict(checkpoint["config"]))
    rules = encode_rules_batch(environment.rules_jsons(), device)
    episode_seeds = np.arange(
        config.seed, config.seed + config.environments, dtype=np.int64
    )
    episode_actions: list[list[int]] = [[] for _ in range(config.environments)]
    samples: list[dict[str, object]] = []
    next_seed = config.seed + config.environments
    completed_games = 0
    started = time.perf_counter()
    for update in range(config.updates):
        observation = environment.observe()
        with torch.no_grad():
            logits, _ = policy(observation, rules)
            distribution = action_distribution(logits, observation["action_offsets"])
        probabilities = distribution.probs.cpu().numpy()
        direct = distribution.logits.argmax(dim=1).cpu().numpy().astype(np.uint64)
        if update % label_stride == 0:
            action_offsets = np.asarray(observation["action_offsets"], dtype=np.int64)
            for index, active_player in enumerate(observation["active_players"]):
                root_player = int(active_player)
                if training_seat is not None and root_player != training_seat:
                    continue
                action_count = int(action_offsets[index + 1] - action_offsets[index])
                if action_count < 2:
                    continue
                candidates = np.argsort(
                    -probabilities[index, :action_count], kind="stable"
                )[:candidate_count].astype(np.uint64)
                if candidates[0] != direct[index]:
                    raise RuntimeError(
                        "candidate order disagrees with the direct policy"
                    )
                outcomes = rollout_candidates(
                    environment, index, candidates, policy, rules, rollout_horizon
                )
                cell_start, cell_end = observation["cell_offsets"][index : index + 2]
                root_territory = int(
                    np.count_nonzero(
                        observation["owners"][cell_start:cell_end] == root_player
                    )
                )
                samples.append(
                    {
                        "episode_seed": int(episode_seeds[index]),
                        "episode_step": len(episode_actions[index]),
                        "replay_actions": episode_actions[index].copy(),
                        "seat": root_player,
                        "round": int(observation["rounds"][index]),
                        "root_territory": root_territory,
                        "candidates": [
                            {
                                **outcome_record(outcome, root_player),
                                "prior": float(probabilities[index, outcome.action]),
                                "action_kind": int(
                                    observation["action_kinds"][
                                        action_offsets[index] + outcome.action
                                    ]
                                ),
                            }
                            for outcome in outcomes
                        ],
                    }
                )
        result = environment.step(direct)
        done = np.logical_or(result["terminal"], result["truncated"])
        for index, action in enumerate(direct):
            episode_actions[index].append(int(action))
        for index in np.flatnonzero(done):
            completed_games += 1
            environment.reset(int(index), next_seed)
            episode_seeds[index] = next_seed
            episode_actions[index] = []
            next_seed += 1
    preferred_better = 0
    alternative_better = 0
    different_winner = 0
    different_territory = 0
    alternative_territory_better = 0
    completed_samples = 0
    for sample in samples:
        candidates = sample["candidates"]
        preferred = candidates[0]
        alternative_territory_better += int(
            max(candidate["territory"] for candidate in candidates[1:])
            > preferred["territory"]
        )
        if all(candidate["root_score"] is not None for candidate in candidates):
            completed_samples += 1
            alternative_scores = [
                candidate["root_score"] for candidate in candidates[1:]
            ]
            preferred_better += int(preferred["root_score"] > max(alternative_scores))
            alternative_better += int(max(alternative_scores) > preferred["root_score"])
            different_winner += int(
                any(
                    candidate["winner"] != preferred["winner"]
                    for candidate in candidates[1:]
                )
            )
        different_territory += int(
            any(
                candidate["territory"] != preferred["territory"]
                for candidate in candidates[1:]
            )
        )
    return {
        "schema_version": 1,
        "kind": "frozen_policy_counterfactual_rollout_scout",
        "source": {
            "path": str(checkpoint_path),
            "sha256": digest(checkpoint_path),
            "seat_experts": selected_experts,
        },
        "config": {
            "domain": asdict(config),
            "descriptor": descriptor,
            "label_stride": label_stride,
            "candidate_count": candidate_count,
            "training_seat": training_seat,
            "rollout_horizon": rollout_horizon,
        },
        "sample_count": len(samples),
        "completed_samples": completed_samples,
        "completed_rollin_games": completed_games,
        "preferred_better_states": preferred_better,
        "alternative_better_states": alternative_better,
        "different_winner_states": different_winner,
        "different_territory_states": different_territory,
        "alternative_territory_better_states": alternative_territory_better,
        "seconds": time.perf_counter() - started,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    add_collection_arguments(parser)
    parser.add_argument("--label-stride", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--training-seat", type=int)
    parser.add_argument("--rollout-horizon", type=int)
    arguments = parser.parse_args()
    report = benchmark(
        arguments.checkpoint,
        collection_config(arguments),
        arguments.label_stride,
        arguments.candidate_count,
        arguments.training_seat,
        arguments.rollout_horizon,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
