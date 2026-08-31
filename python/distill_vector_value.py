from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from antiyoy_rl import ProceduralConfig, VectorEnv
from antiyoy_rl.model import (
    UniversalPolicy,
    action_distribution,
    domain_key,
    encode_rules_batch,
)
from antiyoy_rl.puct import maxn_leaf_utilities
from antiyoy_rl.routed import RoutedPolicy
from antiyoy_rl.vector_value import (
    MAX_PLAYERS,
    VECTOR_VALUE_ARTIFACT_KIND,
    VECTOR_VALUE_ARTIFACT_VERSION,
    RelativeValueHead,
    initialize_from_scalar_value_head,
)

try:
    from .build_bundle import digest
    from .evaluate import (
        instantiate_policy,
        load_policy_checkpoint,
        select_policy_state,
    )
except ImportError:
    from build_bundle import digest
    from evaluate import (
        instantiate_policy,
        load_policy_checkpoint,
        select_policy_state,
    )


@dataclass(frozen=True)
class VectorValueDistillationConfig:
    profile: str = "classic_generic_2022"
    players: int = 5
    environments: int = 64
    updates: int = 500
    validation_environments: int = 32
    validation_steps: int = 16
    seed: int = 1_080_000
    device: str = "cuda"
    width: int = 19
    height: int = 15
    action_limit: int = 1_000
    land_density_per_million: int = 650_000
    starting_province_size: int = 5
    starting_money: int = 10
    tree_density_per_million: int = 150_000
    neutral_tower_density_per_million: int = 20_000
    neutral_capital_density_per_million: int = 10_000
    grave_density_per_million: int = 15_000
    learning_rate: float = 3e-4


def validate_config(config: VectorValueDistillationConfig) -> None:
    if config.players < 2 or config.players > MAX_PLAYERS:
        raise ValueError("vector-value player count must be between two and eight")
    if (
        min(
            config.environments,
            config.updates,
            config.validation_environments,
            config.validation_steps,
        )
        < 1
    ):
        raise ValueError("vector-value batch and training sizes must be positive")
    if config.width < 3 or config.height < 3 or config.action_limit < 1:
        raise ValueError("vector-value arena dimensions are invalid")
    densities = (
        config.land_density_per_million,
        config.tree_density_per_million,
        config.neutral_tower_density_per_million,
        config.neutral_capital_density_per_million,
        config.grave_density_per_million,
    )
    if any(density < 0 or density > 1_000_000 for density in densities):
        raise ValueError("vector-value map density is invalid")
    if config.starting_province_size < 1 or config.starting_money < 0:
        raise ValueError("vector-value starting economy is invalid")
    if config.learning_rate <= 0:
        raise ValueError("vector-value learning rate must be positive")


def domain_descriptor(
    config: VectorValueDistillationConfig, checkpoint_config: dict[str, object]
) -> dict[str, object]:
    return {
        "width": config.width,
        "height": config.height,
        "players": config.players,
        "action_limit": config.action_limit,
        "fog": checkpoint_config["fog"],
        "diplomacy": checkpoint_config.get("diplomacy", False),
        "initial_relation": checkpoint_config.get("initial_relation", "neutral"),
        "land_density_per_million": config.land_density_per_million,
        "starting_province_size": config.starting_province_size,
        "starting_money": config.starting_money,
        "tree_density_per_million": config.tree_density_per_million,
        "neutral_tower_density_per_million": config.neutral_tower_density_per_million,
        "neutral_capital_density_per_million": (
            config.neutral_capital_density_per_million
        ),
        "grave_density_per_million": config.grave_density_per_million,
    }


def create_environment(
    config: VectorValueDistillationConfig,
    checkpoint_config: dict[str, object],
    environments: int,
    seed: int,
) -> VectorEnv:
    generator = ProceduralConfig(
        width=config.width,
        height=config.height,
        players=config.players,
        seed=seed,
        land_density_per_million=config.land_density_per_million,
        starting_province_size=config.starting_province_size,
        starting_money=config.starting_money,
        tree_density_per_million=config.tree_density_per_million,
        neutral_tower_density_per_million=config.neutral_tower_density_per_million,
        neutral_capital_density_per_million=(
            config.neutral_capital_density_per_million
        ),
        grave_density_per_million=config.grave_density_per_million,
    )
    return VectorEnv.procedural(
        environments,
        generator,
        action_limit=config.action_limit,
        profile=config.profile,
        fog=bool(checkpoint_config["fog"]),
        diplomacy=bool(checkpoint_config.get("diplomacy", False)),
        initial_relation=str(checkpoint_config.get("initial_relation", "neutral")),
    )


def load_routed_policy(
    checkpoint: dict[str, object],
    config: VectorValueDistillationConfig,
    descriptor: dict[str, object],
    device: torch.device,
) -> tuple[RoutedPolicy, list[str], int, int]:
    evaluation_domain = domain_key("procedural_v1", descriptor)
    models: dict[str, UniversalPolicy] = {}
    experts: list[str] = []
    hidden = 0
    layers = 0
    for seat in range(config.players):
        state, selected_config = select_policy_state(
            checkpoint,
            config.profile,
            "procedural_v1",
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


def teacher_utilities(
    policy: RoutedPolicy,
    observation: dict[str, np.ndarray],
    rules: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits, active_values = policy(observation, rules)
        utilities = maxn_leaf_utilities(
            policy, observation, rules, active_values
        ).clamp(-1, 1)
    return logits, utilities


def validation_metrics(
    policy: RoutedPolicy,
    head: RelativeValueHead,
    config: VectorValueDistillationConfig,
    checkpoint_config: dict[str, object],
    device: torch.device,
) -> dict[str, float | int]:
    environment = create_environment(
        config,
        checkpoint_config,
        config.validation_environments,
        config.seed + 10_000_000,
    )
    rules = encode_rules_batch(environment.rules_jsons(), device)
    squared_error = 0.0
    absolute_error = 0.0
    labels = 0
    next_reset_seed = config.seed + 10_000_000 + config.validation_environments
    head.eval()
    for _ in range(config.validation_steps):
        observation = environment.observe()
        logits, targets = teacher_utilities(policy, observation, rules)
        with torch.no_grad():
            _, predictions = policy.maxn(observation, rules, head)
        difference = predictions - targets
        squared_error += float(difference.square().sum().item())
        absolute_error += float(difference.abs().sum().item())
        labels += targets.numel()
        actions = (
            action_distribution(logits, observation["action_offsets"])
            .logits.argmax(dim=1)
            .cpu()
            .numpy()
            .astype(np.uint64)
        )
        result = environment.step(actions)
        done = np.logical_or(result["terminal"], result["truncated"])
        for environment_index in np.flatnonzero(done):
            environment.reset(int(environment_index), next_reset_seed)
            next_reset_seed += 1
    return {
        "labels": labels,
        "mse": squared_error / labels,
        "mae": absolute_error / labels,
    }


def distill_vector_value(
    checkpoint_path: Path,
    output_path: Path,
    config: VectorValueDistillationConfig,
) -> dict[str, object]:
    validate_config(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    checkpoint_config = dict(checkpoint["config"])
    descriptor = domain_descriptor(config, checkpoint_config)
    policy, experts, hidden, layers = load_routed_policy(
        checkpoint, config, descriptor, device
    )
    head = RelativeValueHead(hidden).to(device)
    initialize_from_scalar_value_head(head, policy.models[experts[0]].value_head)
    initial_metrics = validation_metrics(
        policy, head, config, checkpoint_config, device
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    environment = create_environment(
        config, checkpoint_config, config.environments, config.seed
    )
    rules = encode_rules_batch(environment.rules_jsons(), device)
    next_reset_seed = config.seed + config.environments
    completed_games = 0
    labels = 0
    loss_average = 0.0
    started = time.perf_counter()
    head.train()
    for update in range(1, config.updates + 1):
        observation = environment.observe()
        logits, targets = teacher_utilities(policy, observation, rules)
        _, predictions = policy.maxn(observation, rules, head)
        loss = functional.mse_loss(predictions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        labels += targets.numel()
        loss_average += (float(loss.item()) - loss_average) / update
        actions = (
            action_distribution(logits, observation["action_offsets"])
            .logits.argmax(dim=1)
            .cpu()
            .numpy()
            .astype(np.uint64)
        )
        result = environment.step(actions)
        done = np.logical_or(result["terminal"], result["truncated"])
        for environment_index in np.flatnonzero(done):
            completed_games += 1
            environment.reset(int(environment_index), next_reset_seed)
            next_reset_seed += 1
        if update == 1 or update % 100 == 0 or update == config.updates:
            print(
                json.dumps(
                    {
                        "stage": "vector_value_distillation",
                        "update": update,
                        "loss": float(loss.item()),
                        "labels": labels,
                        "completed_games": completed_games,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    seconds = time.perf_counter() - started
    final_metrics = validation_metrics(policy, head, config, checkpoint_config, device)
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "maxn_vector_value_distillation",
        "source": {
            "path": str(checkpoint_path),
            "sha256": digest(checkpoint_path),
            "seat_experts": experts,
        },
        "configuration": asdict(config),
        "domain": domain_key("procedural_v1", descriptor),
        "domain_descriptor": descriptor,
        "architecture": {"hidden": hidden, "layers": layers},
        "teacher": "repeated_scalar_player_perspectives",
        "student": "shared_encoder_relative_vector_head",
        "encoder_pass_reduction": config.players,
        "training": {
            "seconds": seconds,
            "updates": config.updates,
            "labels": labels,
            "labels_per_second": labels / seconds,
            "mean_mse": loss_average,
            "completed_games": completed_games,
        },
        "validation": {"initial": initial_metrics, "final": final_metrics},
    }
    artifact = {
        "kind": VECTOR_VALUE_ARTIFACT_KIND,
        "artifact_version": VECTOR_VALUE_ARTIFACT_VERSION,
        "source": report["source"],
        "architecture": {
            "hidden": hidden,
            "layers": layers,
            "maximum_players": MAX_PLAYERS,
            "perspective": "relative_to_active_player",
        },
        "model": {
            key: value.detach().cpu() for key, value in head.state_dict().items()
        },
        "summary": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(artifact, temporary)
    temporary.replace(output_path)
    report["output"] = {
        "path": str(output_path),
        "sha256": digest(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", default="classic_generic_2022")
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--validation-environments", type=int, default=32)
    parser.add_argument("--validation-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1_080_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--width", type=int, default=19)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--action-limit", type=int, default=1_000)
    parser.add_argument("--land-density-per-million", type=int, default=650_000)
    parser.add_argument("--starting-province-size", type=int, default=5)
    parser.add_argument("--starting-money", type=int, default=10)
    parser.add_argument("--tree-density-per-million", type=int, default=150_000)
    parser.add_argument("--neutral-tower-density-per-million", type=int, default=20_000)
    parser.add_argument(
        "--neutral-capital-density-per-million", type=int, default=10_000
    )
    parser.add_argument("--grave-density-per-million", type=int, default=15_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    arguments = parser.parse_args()
    configuration = vars(arguments).copy()
    checkpoint_path = configuration.pop("checkpoint")
    output_path = configuration.pop("output")
    config = VectorValueDistillationConfig(**configuration)
    report = distill_vector_value(checkpoint_path, output_path, config)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
