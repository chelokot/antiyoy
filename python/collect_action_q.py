from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import ProceduralConfig, VectorEnv
from antiyoy_rl.model import (
    action_distribution,
    domain_key,
    encode_rules_batch,
    select_environments,
)
from antiyoy_rl.puct import PolicySearchConfig, policy_search_actions
from antiyoy_rl.routed import RoutedPolicy

try:
    from .build_bundle import digest
    from .distill_puct import (
        PuctDistillationConfig,
        create_environment,
        domain_descriptor,
        validate_config,
    )
    from .evaluate import (
        instantiate_policy,
        load_policy_checkpoint,
        select_policy_state,
    )
except ImportError:
    from build_bundle import digest
    from distill_puct import (
        PuctDistillationConfig,
        create_environment,
        domain_descriptor,
        validate_config,
    )
    from evaluate import instantiate_policy, load_policy_checkpoint, select_policy_state


DATASET_SCHEMA_VERSION = 1
DATASET_KIND = "puct_action_q_pair_dataset"


def observation_fingerprint(
    observation: Mapping[str, np.ndarray], environment: int
) -> str:
    selected = select_environments(observation, [environment])
    value = hashlib.sha256()
    for name in sorted(selected):
        array = np.asarray(selected[name])
        value.update(name.encode())
        value.update(array.dtype.str.encode())
        value.update(np.asarray(array.shape, dtype=np.uint64).tobytes())
        value.update(array.tobytes())
    return value.hexdigest()


def verify_shared_action_representation(
    models: Mapping[str, torch.nn.Module],
) -> str:
    experts = list(models)
    reference_expert = experts[0]
    reference = models[reference_expert].state_dict()
    for expert in experts[1:]:
        candidate = models[expert].state_dict()
        for name, expected in reference.items():
            if name.startswith("value_head."):
                continue
            if not torch.equal(candidate[name], expected):
                raise ValueError(
                    "action-Q collection requires experts that differ only in value heads"
                )
    return reference_expert


def load_collection_policy(
    checkpoint: dict[str, object],
    config: PuctDistillationConfig,
    descriptor: dict[str, object],
    device: torch.device,
) -> tuple[RoutedPolicy, list[str], int, int]:
    evaluation_domain = domain_key(config.generator, descriptor)
    models = {}
    experts = []
    hidden = 0
    layers = 0
    for seat in range(config.players):
        state, selected_config = select_policy_state(
            checkpoint,
            config.profile,
            config.generator,
            config.players,
            seat,
            evaluation_domain,
        )
        expert = str(selected_config["selected_expert"])
        if expert not in models:
            model = instantiate_policy(state, selected_config, device)
            model.requires_grad_(False)
            models[expert] = model
        experts.append(expert)
        selected_hidden = int(selected_config["hidden"])
        selected_layers = int(selected_config["layers"])
        if hidden not in (0, selected_hidden) or layers not in (0, selected_layers):
            raise ValueError("routed experts have incompatible architectures")
        hidden = selected_hidden
        layers = selected_layers
    return RoutedPolicy(models, experts), experts, hidden, layers


def save_dataset(dataset: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(dataset, temporary)
    temporary.replace(output_path)


def replay_dataset_example(dataset: dict[str, object], example: int) -> str:
    config = dataset["config"]
    descriptor = config["descriptor"]
    examples = dataset["examples"]
    episode_seed = int(examples["episode_seeds"][example])
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
    start = int(examples["replay_offsets"][example])
    end = int(examples["replay_offsets"][example + 1])
    for action in examples["replay_actions"][start:end]:
        result = environment.step(np.asarray([int(action)], dtype=np.uint64))
        if bool(result["terminal"][0]) or bool(result["truncated"][0]):
            raise ValueError("action-Q replay terminates before its labeled state")
    return observation_fingerprint(environment.observe(), 0)


def collect_action_q(
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
    histories: list[list[int]] = [[] for _ in range(config.environments)]
    search_features: list[torch.Tensor] = []
    direct_features: list[torch.Tensor] = []
    search_values: list[torch.Tensor] = []
    direct_values: list[torch.Tensor] = []
    baseline_margins: list[torch.Tensor] = []
    sample_seeds: list[int] = []
    sample_seats: list[int] = []
    sample_rounds: list[int] = []
    state_fingerprints: list[str] = []
    replay_actions: list[int] = []
    replay_offsets = [0]
    next_reset_seed = config.seed + config.environments
    completed_games = 0
    truncations = 0
    evaluated_leaves = 0
    leaf_batches = 0
    started = time.perf_counter()
    feature_model.eval()
    for _ in range(config.updates):
        observation = environment.observe()
        search_actions, metrics = policy_search_actions(
            environment,
            policy,
            rules,
            active_mask,
            search_config,
            include_root_targets=True,
        )
        root_offsets = np.asarray(metrics["root_action_offsets"], dtype=np.int64)
        root_values = torch.as_tensor(
            metrics["root_values"], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            logits, _, features = feature_model.forward_with_action_features(
                observation, rules
            )
            distribution = action_distribution(logits, observation["action_offsets"])
            direct_actions_tensor = distribution.logits.argmax(dim=1)
        search_actions_tensor = torch.as_tensor(
            search_actions, dtype=torch.long, device=device
        )
        root_starts = torch.as_tensor(
            root_offsets[:-1], dtype=torch.long, device=device
        )
        search_indices = root_starts + search_actions_tensor
        direct_indices = root_starts + direct_actions_tensor
        disagreement = search_actions_tensor != direct_actions_tensor
        for environment_index in torch.nonzero(disagreement).flatten().tolist():
            search_index = int(search_indices[environment_index].item())
            direct_index = int(direct_indices[environment_index].item())
            search_features.append(
                features[search_index].detach().cpu().to(torch.float16)
            )
            direct_features.append(
                features[direct_index].detach().cpu().to(torch.float16)
            )
            search_values.append(root_values[search_index].detach().cpu())
            direct_values.append(root_values[direct_index].detach().cpu())
            baseline_margins.append(
                (logits[search_index] - logits[direct_index]).detach().cpu()
            )
            sample_seeds.append(int(episode_seeds[environment_index]))
            sample_seats.append(int(observation["active_players"][environment_index]))
            sample_rounds.append(int(observation["rounds"][environment_index]))
            state_fingerprints.append(
                observation_fingerprint(observation, environment_index)
            )
            replay_actions.extend(histories[environment_index])
            replay_offsets.append(len(replay_actions))
        rollin_actions = (
            search_actions
            if config.rollin == "teacher"
            else direct_actions_tensor.cpu().numpy().astype(np.uint64)
        )
        result = environment.step(rollin_actions)
        done = np.logical_or(result["terminal"], result["truncated"])
        for environment_index, action in enumerate(rollin_actions):
            histories[environment_index].append(int(action))
        for environment_index in np.flatnonzero(done):
            truncations += int(result["truncated"][environment_index])
            completed_games += 1
            environment.reset(int(environment_index), next_reset_seed)
            episode_seeds[environment_index] = next_reset_seed
            next_reset_seed += 1
            histories[environment_index].clear()
        evaluated_leaves += int(metrics["evaluated_leaves"])
        leaf_batches += int(metrics["leaf_batches"])
    if not search_features:
        raise RuntimeError("PUCT collection produced no policy disagreements")
    stacked_search_values = torch.stack(search_values).to(torch.float32)
    stacked_direct_values = torch.stack(direct_values).to(torch.float32)
    regrets = stacked_search_values - stacked_direct_values
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
        "examples": {
            "search_features": torch.stack(search_features),
            "direct_features": torch.stack(direct_features),
            "search_values": stacked_search_values,
            "direct_values": stacked_direct_values,
            "regrets": regrets,
            "baseline_margins": torch.stack(baseline_margins).to(torch.float32),
            "episode_seeds": torch.tensor(sample_seeds, dtype=torch.int64),
            "seats": torch.tensor(sample_seats, dtype=torch.uint8),
            "rounds": torch.tensor(sample_rounds, dtype=torch.int32),
            "state_fingerprints": state_fingerprints,
            "replay_offsets": torch.tensor(replay_offsets, dtype=torch.int64),
            "replay_actions": torch.tensor(replay_actions, dtype=torch.int32),
        },
    }
    verification_indices = np.linspace(
        0, len(search_features) - 1, min(16, len(search_features)), dtype=np.int64
    )
    for example in verification_indices:
        actual = replay_dataset_example(dataset, int(example))
        if actual != state_fingerprints[int(example)]:
            raise RuntimeError(f"action-Q replay mismatch at example {int(example)}")
    save_dataset(dataset, output_path)
    positive = regrets > 0
    positive_regret_count = int(positive.sum().item())
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
        "examples": len(search_features),
        "positive_regrets": positive_regret_count,
        "positive_regret_rate": float(positive.float().mean().item()),
        "mean_regret": float(regrets.mean().item()),
        "mean_positive_regret": (
            float(regrets[positive].mean().item())
            if positive_regret_count > 0
            else None
        ),
        "seats": torch.bincount(
            dataset["examples"]["seats"].to(torch.long), minlength=config.players
        ).tolist(),
        "unique_episode_seeds": len(set(sample_seeds)),
        "replay_actions": len(replay_actions),
        "verified_replays": len(verification_indices),
        "completed_games": completed_games,
        "truncations": truncations,
        "evaluated_leaves": evaluated_leaves,
        "leaf_batches": leaf_batches,
        "seconds": time.perf_counter() - started,
    }
    return report


def add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="classic_generic_2022")
    parser.add_argument(
        "--generator",
        choices=("symmetric_duel_v1", "procedural_v1"),
        default="procedural_v1",
    )
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1_300_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--width", type=int, default=19)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--action-limit", type=int, default=2_400)
    parser.add_argument("--land-density-per-million", type=int, default=650_000)
    parser.add_argument("--starting-province-size", type=int, default=5)
    parser.add_argument("--starting-money", type=int, default=10)
    parser.add_argument("--tree-density-per-million", type=int, default=150_000)
    parser.add_argument("--neutral-tower-density-per-million", type=int, default=20_000)
    parser.add_argument(
        "--neutral-capital-density-per-million", type=int, default=10_000
    )
    parser.add_argument("--grave-density-per-million", type=int, default=15_000)
    parser.add_argument("--rollin", choices=("teacher", "student"), default="student")
    parser.add_argument("--puct-nodes", type=int, default=8)
    parser.add_argument("--puct-exploration", type=float, default=1.5)
    parser.add_argument("--puct-virtual-loss", type=float, default=1.0)
    parser.add_argument("--puct-maximum-depth", type=int, default=128)
    parser.add_argument("--puct-root-value-weight", type=float, default=1.0)
    parser.add_argument("--puct-leaf-batch-size", type=int, default=512)


def collection_config(arguments: argparse.Namespace) -> PuctDistillationConfig:
    return PuctDistillationConfig(
        profile=arguments.profile,
        generator=arguments.generator,
        players=arguments.players,
        environments=arguments.environments,
        updates=arguments.updates,
        seed=arguments.seed,
        device=arguments.device,
        width=arguments.width,
        height=arguments.height,
        action_limit=arguments.action_limit,
        land_density_per_million=arguments.land_density_per_million,
        starting_province_size=arguments.starting_province_size,
        starting_money=arguments.starting_money,
        tree_density_per_million=arguments.tree_density_per_million,
        neutral_tower_density_per_million=(arguments.neutral_tower_density_per_million),
        neutral_capital_density_per_million=(
            arguments.neutral_capital_density_per_million
        ),
        grave_density_per_million=arguments.grave_density_per_million,
        rollin=arguments.rollin,
        puct_nodes=arguments.puct_nodes,
        puct_exploration=arguments.puct_exploration,
        puct_virtual_loss=arguments.puct_virtual_loss,
        puct_maximum_depth=arguments.puct_maximum_depth,
        puct_root_value_weight=arguments.puct_root_value_weight,
        puct_leaf_batch_size=arguments.puct_leaf_batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    add_collection_arguments(parser)
    arguments = parser.parse_args()
    report = collect_action_q(
        arguments.checkpoint,
        arguments.output,
        collection_config(arguments),
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
