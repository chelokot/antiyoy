from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl.model import action_distribution, encode_rules_batch
from antiyoy_rl.puct import PolicySearchConfig, policy_search_actions

try:
    from .build_bundle import digest
    from .collect_action_q import (
        add_collection_arguments,
        collection_config,
        load_collection_policy,
        observation_fingerprint,
        save_dataset,
        verify_shared_action_representation,
    )
    from .distill_puct import (
        PuctDistillationConfig,
        create_environment,
        domain_descriptor,
        validate_config,
    )
    from .evaluate import load_policy_checkpoint
except ImportError:
    from build_bundle import digest
    from collect_action_q import (
        add_collection_arguments,
        collection_config,
        load_collection_policy,
        observation_fingerprint,
        save_dataset,
        verify_shared_action_representation,
    )
    from distill_puct import (
        PuctDistillationConfig,
        create_environment,
        domain_descriptor,
        validate_config,
    )
    from evaluate import load_policy_checkpoint


DATASET_SCHEMA_VERSION = 1
DATASET_KIND = "puct_action_q_slate_dataset"


def replay_slate_state(dataset: dict[str, object], state: int) -> str:
    config = dataset["config"]
    descriptor = config["descriptor"]
    states = dataset["states"]
    replay = dataset["replay"]
    episode_seed = int(states["episode_seeds"][state])
    episode_step = int(states["episode_steps"][state])
    replay_seeds = replay["episode_seeds"].tolist()
    try:
        episode = replay_seeds.index(episode_seed)
    except ValueError as error:
        raise ValueError("slate replay is missing its episode") from error
    start = int(replay["action_offsets"][episode])
    end = start + episode_step
    if end > int(replay["action_offsets"][episode + 1]):
        raise ValueError("slate replay episode is shorter than the labeled state")
    from antiyoy_rl import ProceduralConfig, VectorEnv

    environment_arguments = {
        "action_limit": int(descriptor["action_limit"]),
        "profile": str(config["profile"]),
        "fog": bool(descriptor["fog"]),
        "diplomacy": bool(descriptor["diplomacy"]),
        "initial_relation": str(descriptor["initial_relation"]),
    }
    if config["generator"] == "procedural_v1":
        generator = ProceduralConfig(
            width=int(descriptor["width"]),
            height=int(descriptor["height"]),
            players=int(descriptor["players"]),
            seed=episode_seed,
            land_density_per_million=int(descriptor["land_density_per_million"]),
            starting_province_size=int(descriptor["starting_province_size"]),
            starting_money=int(descriptor["starting_money"]),
            tree_density_per_million=int(descriptor["tree_density_per_million"]),
            neutral_tower_density_per_million=int(
                descriptor["neutral_tower_density_per_million"]
            ),
            neutral_capital_density_per_million=int(
                descriptor["neutral_capital_density_per_million"]
            ),
            grave_density_per_million=int(descriptor["grave_density_per_million"]),
        )
        environment = VectorEnv.procedural(1, generator, **environment_arguments)
    else:
        environment = VectorEnv(
            1,
            width=int(descriptor["width"]),
            height=int(descriptor["height"]),
            seed=episode_seed,
            **environment_arguments,
        )
    environment.reset(0, episode_seed)
    for action in replay["actions"][start:end]:
        result = environment.step(np.asarray([int(action)], dtype=np.uint64))
        if bool(result["terminal"][0]) or bool(result["truncated"][0]):
            raise ValueError("slate replay terminates before its labeled state")
    return observation_fingerprint(environment.observe(), 0)


def informative_states(
    offsets: torch.Tensor, values: torch.Tensor, visits: torch.Tensor
) -> int:
    informative = 0
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        selected = visits[int(start) : int(end)] > 0
        measured = values[int(start) : int(end)][selected]
        informative += int(
            measured.numel() > 1 and float(measured.max() - measured.min()) > 1e-6
        )
    return informative


def collect_action_slates(
    checkpoint_path: Path,
    output_path: Path,
    config: PuctDistillationConfig,
) -> dict[str, object]:
    validate_config(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    checkpoint_config = dict(checkpoint["config"])
    descriptor = domain_descriptor(config, checkpoint_config)
    policy, selected_experts, hidden, layers = load_collection_policy(
        checkpoint, config, descriptor, device
    )
    feature_expert = verify_shared_action_representation(policy.models)
    feature_model = policy.models[feature_expert]
    environment = create_environment(config, checkpoint_config)
    episode_seeds = np.arange(
        config.seed, config.seed + config.environments, dtype=np.int64
    )
    episode_actions: dict[int, list[int]] = {
        int(episode_seed): [] for episode_seed in episode_seeds
    }
    for environment_index, episode_seed in enumerate(episode_seeds):
        environment.reset(environment_index, int(episode_seed))
    rules = encode_rules_batch(environment.rules_jsons(), device)
    search_config = PolicySearchConfig(
        node_budget=config.puct_nodes,
        exploration=config.puct_exploration,
        virtual_loss=config.puct_virtual_loss,
        maximum_depth=config.puct_maximum_depth,
        root_value_weight=config.puct_root_value_weight,
        leaf_batch_size=config.puct_leaf_batch_size,
        value_perspective=config.puct_value_perspective,
        opponent_horizon=config.puct_opponent_horizon,
        objective=config.puct_objective,
    )
    active_mask = np.ones(config.environments, dtype=np.uint8)
    action_features: list[torch.Tensor] = []
    baseline_logits: list[torch.Tensor] = []
    root_probabilities: list[torch.Tensor] = []
    root_values: list[torch.Tensor] = []
    root_visits: list[torch.Tensor] = []
    slate_offsets = [0]
    sample_seeds: list[int] = []
    sample_steps: list[int] = []
    sample_seats: list[int] = []
    sample_rounds: list[int] = []
    search_actions: list[int] = []
    direct_actions: list[int] = []
    state_fingerprints: list[str] = []
    next_reset_seed = config.seed + config.environments
    completed_games = 0
    truncations = 0
    evaluated_leaves = 0
    leaf_batches = 0
    started = time.perf_counter()
    feature_model.eval()
    for _ in range(config.updates):
        observation = environment.observe()
        selected_actions, metrics = policy_search_actions(
            environment,
            policy,
            rules,
            active_mask,
            search_config,
            include_root_targets=True,
        )
        root_offsets = np.asarray(metrics["root_action_offsets"], dtype=np.int64)
        observation_offsets = np.asarray(observation["action_offsets"], dtype=np.int64)
        if not np.array_equal(root_offsets, observation_offsets):
            raise RuntimeError("PUCT root actions do not match policy action features")
        with torch.no_grad():
            logits, _, features = feature_model.forward_with_action_features(
                observation, rules
            )
            distribution = action_distribution(logits, observation["action_offsets"])
            direct = distribution.logits.argmax(dim=1)
        action_features.append(features.detach().cpu().to(torch.float16))
        baseline_logits.append(logits.detach().cpu().to(torch.float16))
        root_probabilities.append(
            torch.as_tensor(metrics["root_probabilities"], dtype=torch.float32)
        )
        root_values.append(torch.as_tensor(metrics["root_values"], dtype=torch.float32))
        root_visits.append(
            torch.as_tensor(metrics["root_action_visits"], dtype=torch.int32)
        )
        for environment_index, (start, end) in enumerate(
            zip(root_offsets[:-1], root_offsets[1:], strict=True)
        ):
            slate_offsets.append(slate_offsets[-1] + int(end - start))
            episode_seed = int(episode_seeds[environment_index])
            sample_seeds.append(episode_seed)
            sample_steps.append(len(episode_actions[episode_seed]))
            sample_seats.append(int(observation["active_players"][environment_index]))
            sample_rounds.append(int(observation["rounds"][environment_index]))
            search_actions.append(int(selected_actions[environment_index]))
            direct_actions.append(int(direct[environment_index].item()))
            state_fingerprints.append(
                observation_fingerprint(observation, environment_index)
            )
        rollin_actions = (
            selected_actions
            if config.rollin == "teacher"
            else direct.cpu().numpy().astype(np.uint64)
        )
        result = environment.step(rollin_actions)
        done = np.logical_or(result["terminal"], result["truncated"])
        for environment_index, action in enumerate(rollin_actions):
            episode_actions[int(episode_seeds[environment_index])].append(int(action))
        for environment_index in np.flatnonzero(done):
            truncations += int(result["truncated"][environment_index])
            completed_games += 1
            environment.reset(int(environment_index), next_reset_seed)
            episode_seeds[environment_index] = next_reset_seed
            episode_actions[next_reset_seed] = []
            next_reset_seed += 1
        evaluated_leaves += int(metrics["evaluated_leaves"])
        leaf_batches += int(metrics["leaf_batches"])
    features = torch.cat(action_features)
    logits = torch.cat(baseline_logits).to(torch.float32)
    probabilities = torch.cat(root_probabilities)
    values = torch.cat(root_values)
    visits = torch.cat(root_visits)
    offsets = torch.tensor(slate_offsets, dtype=torch.int64)
    if not (
        features.shape[0]
        == logits.shape[0]
        == probabilities.shape[0]
        == values.shape[0]
        == visits.shape[0]
        == int(offsets[-1])
    ):
        raise RuntimeError("action slate arrays have inconsistent lengths")
    replay_episode_seeds = sorted(set(sample_seeds))
    replay_actions: list[int] = []
    replay_offsets = [0]
    for episode_seed in replay_episode_seeds:
        replay_actions.extend(episode_actions[episode_seed])
        replay_offsets.append(len(replay_actions))
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "source": {
            "path": str(checkpoint_path),
            "sha256": digest(checkpoint_path),
            "seat_experts": selected_experts,
            "feature_expert": feature_expert,
        },
        "model": {
            "hidden": hidden,
            "layers": layers,
            "feature_width": hidden * 5,
        },
        "config": {
            "profile": config.profile,
            "generator": config.generator,
            "players": config.players,
            "descriptor": descriptor,
            "seed": config.seed,
            "environments": config.environments,
            "updates": config.updates,
            "rollin": config.rollin,
            "puct": {
                "node_budget": config.puct_nodes,
                "exploration": config.puct_exploration,
                "virtual_loss": config.puct_virtual_loss,
                "maximum_depth": config.puct_maximum_depth,
                "root_value_weight": config.puct_root_value_weight,
                "leaf_batch_size": config.puct_leaf_batch_size,
                "value_perspective": config.puct_value_perspective,
                "opponent_horizon": config.puct_opponent_horizon,
                "objective": config.puct_objective,
            },
        },
        "actions": {
            "offsets": offsets,
            "features": features,
            "baseline_logits": logits,
            "root_probabilities": probabilities,
            "root_values": values,
            "root_visits": visits,
        },
        "states": {
            "episode_seeds": torch.tensor(sample_seeds, dtype=torch.int64),
            "episode_steps": torch.tensor(sample_steps, dtype=torch.int32),
            "seats": torch.tensor(sample_seats, dtype=torch.uint8),
            "rounds": torch.tensor(sample_rounds, dtype=torch.int32),
            "search_actions": torch.tensor(search_actions, dtype=torch.int32),
            "direct_actions": torch.tensor(direct_actions, dtype=torch.int32),
            "fingerprints": state_fingerprints,
        },
        "replay": {
            "episode_seeds": torch.tensor(replay_episode_seeds, dtype=torch.int64),
            "action_offsets": torch.tensor(replay_offsets, dtype=torch.int64),
            "actions": torch.tensor(replay_actions, dtype=torch.int32),
        },
    }
    verification_indices = np.linspace(
        0, len(sample_seeds) - 1, min(16, len(sample_seeds)), dtype=np.int64
    )
    for state in verification_indices:
        actual = replay_slate_state(dataset, int(state))
        if actual != state_fingerprints[int(state)]:
            raise RuntimeError(f"action slate replay mismatch at state {int(state)}")
    save_dataset(dataset, output_path)
    state_count = len(sample_seeds)
    report: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "output": {
            "path": str(output_path),
            "sha256": digest(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "source": dataset["source"],
        "config": dataset["config"],
        "states": state_count,
        "actions": int(features.shape[0]),
        "mean_actions_per_state": float(features.shape[0] / state_count),
        "visited_actions": int((visits > 0).sum()),
        "informative_states": informative_states(offsets, values, visits),
        "search_disagreements": int(
            (dataset["states"]["search_actions"] != dataset["states"]["direct_actions"])
            .sum()
            .item()
        ),
        "seats": torch.bincount(
            dataset["states"]["seats"].to(torch.long), minlength=config.players
        ).tolist(),
        "unique_episode_seeds": len(replay_episode_seeds),
        "replay_actions": len(replay_actions),
        "verified_replays": len(verification_indices),
        "completed_games": completed_games,
        "truncations": truncations,
        "evaluated_leaves": evaluated_leaves,
        "leaf_batches": leaf_batches,
        "seconds": time.perf_counter() - started,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    add_collection_arguments(parser)
    arguments = parser.parse_args()
    report = collect_action_slates(
        arguments.checkpoint,
        arguments.output,
        collection_config(arguments),
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
