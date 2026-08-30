from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from antiyoy_rl import OBSERVATION_VERSION, ProceduralConfig, ScenarioObjective, VectorEnv
from antiyoy_rl.model import (
    RULE_FEATURES,
    UniversalPolicy,
    action_distribution,
    encode_rules_batch,
    load_policy_state,
    rotate_observation_180,
)


CHECKPOINT_VERSION = 5


@dataclass(frozen=True)
class TrainingConfig:
    environments: int
    updates: int
    procedural: bool
    width: int
    height: int
    players: int
    seed: int
    land_density_per_million: int
    starting_province_size: int
    starting_money: int
    tree_density_per_million: int
    neutral_tower_density_per_million: int
    neutral_capital_density_per_million: int
    grave_density_per_million: int
    objective_json: str | None
    action_limit: int
    hidden: int
    layers: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    rollout_steps: int
    epochs: int
    clip_ratio: float
    imitation_updates: int
    imitation_teacher: str
    imitation_rollin: str
    imitation_symmetry_augmentation: bool
    checkpoint_every: int
    search_nodes: int
    search_beam_width: int
    search_branch_width: int
    search_maximum_actions_per_turn: int
    entropy_weight: float
    value_weight: float
    territory_weight: float
    treasury_weight: float
    unit_weight: float
    profile: str | None
    profiles: list[str] | None
    fog: bool
    diplomacy: bool
    initial_relation: str
    device: str
    initialize: Path | None
    resume: Path | None
    checkpoint: Path | None


@dataclass
class Rollout:
    observations: list[dict[str, np.ndarray]]
    actions: list[Tensor]
    log_probabilities: Tensor
    values: Tensor
    rewards: Tensor
    perspectives: Tensor
    continuations: Tensor
    bootstrap: Tensor


def reward_tensor(result: dict[str, np.ndarray], config: TrainingConfig, device: torch.device) -> Tensor:
    outcome = torch.as_tensor(result["outcomes"], dtype=torch.float32, device=device)
    territory = torch.as_tensor(result["territory_delta"], dtype=torch.float32, device=device)
    treasury = torch.as_tensor(result["treasury_delta"], dtype=torch.float32, device=device)
    units = torch.as_tensor(result["unit_strength_delta"], dtype=torch.float32, device=device)
    return (
        outcome
        + config.territory_weight * territory
        + config.treasury_weight * treasury
        + config.unit_weight * units
    )


def make_environment(config: TrainingConfig) -> VectorEnv:
    generator = ProceduralConfig(
        width=config.width,
        height=config.height,
        players=config.players,
        seed=config.seed,
        land_density_per_million=config.land_density_per_million,
        starting_province_size=config.starting_province_size,
        starting_money=config.starting_money,
        tree_density_per_million=config.tree_density_per_million,
        neutral_tower_density_per_million=config.neutral_tower_density_per_million,
        neutral_capital_density_per_million=config.neutral_capital_density_per_million,
        grave_density_per_million=config.grave_density_per_million,
    )
    objective = (
        None
        if config.objective_json is None
        else ScenarioObjective.from_json(config.objective_json)
    )
    if config.profiles is None:
        if config.procedural:
            return VectorEnv.procedural(
                config.environments,
                generator,
                action_limit=config.action_limit,
                profile=cast(str, config.profile),
                fog=config.fog,
                diplomacy=config.diplomacy,
                initial_relation=config.initial_relation,
                objective=objective,
            )
        return VectorEnv(
            config.environments,
            width=config.width,
            height=config.height,
            seed=config.seed,
            action_limit=config.action_limit,
            profile=cast(str, config.profile),
            fog=config.fog,
            diplomacy=config.diplomacy,
            initial_relation=config.initial_relation,
            objective=objective,
        )
    schedule = [
        config.profiles[index % len(config.profiles)]
        for index in range(config.environments)
    ]
    if config.procedural:
        return VectorEnv.procedural_mixed(
            schedule,
            generator,
            action_limit=config.action_limit,
            fog=config.fog,
            diplomacy=config.diplomacy,
            initial_relation=config.initial_relation,
            objective=objective,
        )
    return VectorEnv.mixed(
        schedule,
        width=config.width,
        height=config.height,
        seed=config.seed,
        action_limit=config.action_limit,
        fog=config.fog,
        diplomacy=config.diplomacy,
        initial_relation=config.initial_relation,
        objective=objective,
    )


def validate_config(config: TrainingConfig) -> None:
    positive = {
        "environments": config.environments,
        "rollout_steps": config.rollout_steps,
        "epochs": config.epochs,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"positive training values required: {', '.join(invalid)}")
    if config.clip_ratio <= 0:
        raise ValueError("clip_ratio must be positive")
    if config.imitation_updates < 0:
        raise ValueError("imitation_updates must not be negative")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must not be negative")
    if config.checkpoint_every > 0 and config.checkpoint is None:
        raise ValueError("checkpoint_every requires a checkpoint path")
    if config.updates < 0:
        raise ValueError("updates must not be negative")
    if config.updates == 0 and config.imitation_updates == 0:
        raise ValueError("at least one PPO or imitation update is required")
    if config.imitation_teacher not in {"greedy", "search"}:
        raise ValueError("imitation_teacher must be greedy or search")
    if config.imitation_rollin not in {"teacher", "policy"}:
        raise ValueError("imitation_rollin must be teacher or policy")
    if config.search_nodes < 2:
        raise ValueError("search_nodes must be at least two")
    if config.search_beam_width < 1:
        raise ValueError("search_beam_width must be positive")
    if config.search_branch_width < 2:
        raise ValueError("search_branch_width must be at least two")
    if config.search_maximum_actions_per_turn < 1:
        raise ValueError("search_maximum_actions_per_turn must be positive")
    if config.profiles is not None and not config.profiles:
        raise ValueError("profiles must not be empty")
    if config.initialize is not None and config.resume is not None:
        raise ValueError("initialize and resume are mutually exclusive")
    if config.procedural and config.players < 2:
        raise ValueError("procedural maps require at least two players")
    densities = [
        config.land_density_per_million,
        config.tree_density_per_million,
        config.neutral_tower_density_per_million,
        config.neutral_capital_density_per_million,
        config.grave_density_per_million,
    ]
    if config.procedural and any(density < 0 or density > 1_000_000 for density in densities):
        raise ValueError("procedural densities must be between zero and one million")
    if config.procedural and sum(densities[1:]) > 1_000_000:
        raise ValueError("procedural neutral object densities exceed one million")


def collect_rollout(
    environment: VectorEnv,
    model: UniversalPolicy,
    rules: Tensor,
    config: TrainingConfig,
    device: torch.device,
    reset_seed: int,
) -> tuple[Rollout, int]:
    observations: list[dict[str, np.ndarray]] = []
    actions: list[Tensor] = []
    log_probabilities: list[Tensor] = []
    values: list[Tensor] = []
    rewards: list[Tensor] = []
    perspectives: list[Tensor] = []
    continuations: list[Tensor] = []
    observation = environment.observe()
    for _ in range(config.rollout_steps):
        with torch.no_grad():
            logits, value = model(observation, rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            action = distribution.sample()
            log_probability = distribution.log_prob(action)
        result = environment.step(action.cpu().numpy().astype(np.uint64))
        done = np.logical_or(result["terminal"], result["truncated"])
        for index in np.flatnonzero(done):
            environment.reset(int(index), reset_seed)
            reset_seed += 1
        next_observation = environment.observe()
        actors = torch.as_tensor(result["actors"], dtype=torch.long, device=device)
        next_players = torch.as_tensor(
            next_observation["active_players"], dtype=torch.long, device=device
        )
        observations.append(observation)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append(reward_tensor(result, config, device))
        perspectives.append(torch.where(actors == next_players, 1.0, -1.0))
        continuations.append(torch.as_tensor(~done, dtype=torch.float32, device=device))
        observation = next_observation
    with torch.no_grad():
        _, bootstrap = model(observation, rules)
    return (
        Rollout(
            observations=observations,
            actions=actions,
            log_probabilities=torch.stack(log_probabilities),
            values=torch.stack(values),
            rewards=torch.stack(rewards),
            perspectives=torch.stack(perspectives),
            continuations=torch.stack(continuations),
            bootstrap=bootstrap,
        ),
        reset_seed,
    )


def pretrain_teacher(
    environment: VectorEnv,
    model: UniversalPolicy,
    optimizer: torch.optim.Optimizer,
    rules: Tensor,
    config: TrainingConfig,
    reset_seed: int,
    device: torch.device,
    checkpoint_callback: Callable[[int, float, float], None] | None = None,
) -> tuple[int, float, float]:
    loss_average = 0.0
    accuracy_average = 0.0
    for update in range(1, config.imitation_updates + 1):
        observation = environment.observe()
        selected = (
            environment.greedy_actions()
            if config.imitation_teacher == "greedy"
            else environment.search_actions(
                node_budget=config.search_nodes,
                beam_width=config.search_beam_width,
                branch_width=config.search_branch_width,
                maximum_actions_per_turn=config.search_maximum_actions_per_turn,
            )
        )
        targets = torch.as_tensor(
            selected, dtype=torch.long, device=device
        )
        model_observation = observation
        if config.imitation_symmetry_augmentation:
            rotation_mask = (
                np.arange(config.environments, dtype=np.uint64) + update
            ) % 2 == 0
            model_observation = rotate_observation_180(observation, rotation_mask)
        logits, _ = model(model_observation, rules)
        distribution = action_distribution(logits, model_observation["action_offsets"])
        loss = -distribution.log_prob(targets).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        policy_actions = distribution.logits.argmax(dim=1)
        accuracy = float((policy_actions == targets).float().mean().item())
        loss_average += (float(loss.item()) - loss_average) / update
        accuracy_average += (accuracy - accuracy_average) / update
        rollin_actions = targets if config.imitation_rollin == "teacher" else policy_actions
        result = environment.step(rollin_actions.cpu().numpy().astype(np.uint64))
        done = np.logical_or(result["terminal"], result["truncated"])
        for index in np.flatnonzero(done):
            environment.reset(int(index), reset_seed)
            reset_seed += 1
        if checkpoint_callback is not None:
            checkpoint_callback(update, loss_average, accuracy_average)
        if update == 1 or update % 100 == 0 or update == config.imitation_updates:
            print(
                json.dumps(
                    {
                        "stage": f"{config.imitation_teacher}_distillation",
                        "update": update,
                        "loss": float(loss.item()),
                        "accuracy": accuracy,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return reset_seed, loss_average, accuracy_average


def rollout_targets(rollout: Rollout, config: TrainingConfig) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rollout.rewards)
    next_advantage = torch.zeros_like(rollout.bootstrap)
    next_value = rollout.bootstrap
    for step in range(config.rollout_steps - 1, -1, -1):
        continuation = rollout.continuations[step]
        perspective = rollout.perspectives[step]
        delta = (
            rollout.rewards[step]
            + config.gamma * perspective * next_value * continuation
            - rollout.values[step]
        )
        next_advantage = (
            delta
            + config.gamma
            * config.gae_lambda
            * perspective
            * continuation
            * next_advantage
        )
        advantages[step] = next_advantage
        next_value = rollout.values[step]
    returns = advantages + rollout.values
    advantages = (advantages - advantages.mean()) / advantages.std(correction=0).clamp_min(1e-6)
    return advantages, returns


def optimize_rollout(
    model: UniversalPolicy,
    optimizer: torch.optim.Optimizer,
    rules: Tensor,
    rollout: Rollout,
    advantages: Tensor,
    returns: Tensor,
    config: TrainingConfig,
) -> tuple[float, float]:
    loss_sum = 0.0
    entropy_sum = 0.0
    optimizer_steps = 0
    for _ in range(config.epochs):
        for step in torch.randperm(config.rollout_steps).tolist():
            observation = rollout.observations[step]
            logits, values = model(observation, rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            log_probability = distribution.log_prob(rollout.actions[step])
            ratio = torch.exp(log_probability - rollout.log_probabilities[step])
            unclipped = ratio * advantages[step]
            clipped = torch.clamp(
                ratio, 1 - config.clip_ratio, 1 + config.clip_ratio
            ) * advantages[step]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (returns[step] - values).square().mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_weight * value_loss
                - config.entropy_weight * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.item())
            entropy_sum += float(entropy.item())
    return loss_sum / optimizer_steps, entropy_sum / optimizer_steps


def restore_checkpoint(
    path: Path,
    model: UniversalPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    checkpoint = load_training_checkpoint(path, device, compatible=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def initialize_checkpoint(
    path: Path,
    model: UniversalPolicy,
    device: torch.device,
) -> None:
    checkpoint = load_training_checkpoint(path, device, compatible=True)
    load_policy_state(model, checkpoint["model"])


def load_training_checkpoint(
    path: Path,
    device: torch.device,
    compatible: bool,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    versions = (4, CHECKPOINT_VERSION) if compatible else (CHECKPOINT_VERSION,)
    observations = (6, OBSERVATION_VERSION) if compatible else (OBSERVATION_VERSION,)
    rule_widths = (42, RULE_FEATURES) if compatible else (RULE_FEATURES,)
    mode = "initialization" if compatible else "resume"
    if checkpoint["checkpoint_version"] not in versions:
        raise ValueError(f"{mode} checkpoint format does not match this trainer")
    if checkpoint["observation_version"] not in observations:
        raise ValueError(f"{mode} checkpoint observation contract does not match")
    if checkpoint["rule_features"] not in rule_widths:
        raise ValueError(f"{mode} checkpoint rule feature width does not match")
    return checkpoint


def recovery_checkpoint_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.latest{path.suffix}")


def save_training_checkpoint(
    path: Path,
    model: UniversalPolicy,
    optimizer: torch.optim.Optimizer,
    environment: VectorEnv,
    config: TrainingConfig,
    summary: dict[str, float | int | str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "checkpoint_version": CHECKPOINT_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "rule_features": RULE_FEATURES,
            "training_rules_jsons": environment.rules_jsons(),
            "training_generators_jsons": environment.generator_jsons(),
            "config": {
                **asdict(config),
                "initialize": (
                    str(config.initialize) if config.initialize is not None else None
                ),
                "resume": str(config.resume) if config.resume is not None else None,
                "checkpoint": str(config.checkpoint),
            },
            "summary": summary,
        },
        temporary,
    )
    temporary.replace(path)


def train(config: TrainingConfig) -> dict[str, float | int | str]:
    validate_config(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    environment = make_environment(config)
    model = UniversalPolicy(config.hidden, config.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.initialize is not None:
        initialize_checkpoint(config.initialize, model, device)
    elif config.resume is not None:
        restore_checkpoint(config.resume, model, optimizer, device)
    rules = encode_rules_batch(environment.rules_jsons(), device)
    reset_seed = config.seed + config.environments
    imitation_loss = 0.0
    imitation_accuracy = 0.0
    imitation_started = time.perf_counter()
    recovery_path = (
        recovery_checkpoint_path(config.checkpoint)
        if config.checkpoint is not None and config.checkpoint_every > 0
        else None
    )

    def save_imitation_recovery(update: int, loss: float, accuracy: float) -> None:
        if recovery_path is None or update % config.checkpoint_every != 0:
            return
        save_training_checkpoint(
            recovery_path,
            model,
            optimizer,
            environment,
            config,
            {
                "stage": "imitation",
                "imitation_update": update,
                "imitation_loss": loss,
                "imitation_accuracy": accuracy,
            },
        )

    if config.imitation_updates > 0:
        reset_seed, imitation_loss, imitation_accuracy = pretrain_teacher(
            environment,
            model,
            optimizer,
            rules,
            config,
            reset_seed,
            device,
            save_imitation_recovery,
        )
    imitation_seconds = time.perf_counter() - imitation_started
    imitation_transitions = config.imitation_updates * config.environments
    reward_average = 0.0
    loss_average = 0.0
    for update in range(1, config.updates + 1):
        rollout, reset_seed = collect_rollout(
            environment, model, rules, config, device, reset_seed
        )
        advantages, returns = rollout_targets(rollout, config)
        loss, entropy = optimize_rollout(
            model, optimizer, rules, rollout, advantages, returns, config
        )
        reward = float(rollout.rewards.mean().item())
        reward_average += (reward - reward_average) / update
        loss_average += (loss - loss_average) / update
        if update == 1 or update % 10 == 0 or update == config.updates:
            print(
                json.dumps(
                    {
                        "update": update,
                        "transitions": update
                        * config.rollout_steps
                        * config.environments,
                        "mean_reward": reward,
                        "mean_value": float(rollout.values.mean().item()),
                        "entropy": entropy,
                        "loss": loss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if recovery_path is not None and update % config.checkpoint_every == 0:
            save_training_checkpoint(
                recovery_path,
                model,
                optimizer,
                environment,
                config,
                {
                    "stage": "ppo",
                    "ppo_update": update,
                    "mean_reward": reward_average,
                    "mean_loss": loss_average,
                },
            )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    algorithm = "perspective_ppo_gae"
    if config.imitation_updates > 0:
        algorithm = f"{config.imitation_teacher}_distilled_{config.imitation_rollin}_rollin"
        if config.imitation_symmetry_augmentation:
            algorithm += "_rot180_augmented"
        if config.updates > 0:
            algorithm += "_perspective_ppo_gae"
    summary: dict[str, float | int | str] = {
        "algorithm": algorithm,
        "updates": config.updates,
        "environments": config.environments,
        "map_generator": "procedural_v1" if config.procedural else "symmetric_duel_v1",
        "parameters": parameters,
        "transitions": config.updates * config.rollout_steps * config.environments,
        "optimizer_steps": config.updates * config.rollout_steps * config.epochs,
        "imitation_updates": config.imitation_updates,
        "imitation_teacher": config.imitation_teacher,
        "imitation_rollin": config.imitation_rollin,
        "imitation_symmetry_augmentation": config.imitation_symmetry_augmentation,
        "imitation_transitions": imitation_transitions,
        "imitation_seconds": imitation_seconds,
        "imitation_transitions_per_second": (
            imitation_transitions / imitation_seconds
            if imitation_transitions > 0
            else 0.0
        ),
        "imitation_loss": imitation_loss,
        "imitation_accuracy": imitation_accuracy,
        "mean_reward": reward_average,
        "mean_loss": loss_average,
        "device": str(device),
        "initialized_from": str(config.initialize) if config.initialize is not None else "",
        "resumed_from": str(config.resume) if config.resume is not None else "",
    }
    if config.checkpoint is not None:
        save_training_checkpoint(
            config.checkpoint,
            model,
            optimizer,
            environment,
            config,
            summary,
        )
        if recovery_path is not None:
            recovery_path.unlink(missing_ok=True)
        summary["checkpoint"] = str(config.checkpoint)
    return summary


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--procedural", action="store_true")
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--land-density-per-million", type=int, default=650_000)
    parser.add_argument("--starting-province-size", type=int, default=5)
    parser.add_argument("--starting-money", type=int, default=10)
    parser.add_argument("--tree-density-per-million", type=int, default=150_000)
    parser.add_argument("--neutral-tower-density-per-million", type=int, default=20_000)
    parser.add_argument("--neutral-capital-density-per-million", type=int, default=10_000)
    parser.add_argument("--grave-density-per-million", type=int, default=15_000)
    parser.add_argument("--objective-json")
    parser.add_argument("--action-limit", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--imitation-updates", type=int, default=0)
    parser.add_argument(
        "--imitation-teacher", choices=("greedy", "search"), default="greedy"
    )
    parser.add_argument(
        "--imitation-rollin", choices=("teacher", "policy"), default="teacher"
    )
    parser.add_argument("--imitation-symmetry-augmentation", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--search-nodes", type=int, default=2048)
    parser.add_argument("--search-beam-width", type=int, default=32)
    parser.add_argument("--search-branch-width", type=int, default=48)
    parser.add_argument("--search-maximum-actions-per-turn", type=int, default=24)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--territory-weight", type=float, default=0.03)
    parser.add_argument("--treasury-weight", type=float, default=0.002)
    parser.add_argument("--unit-weight", type=float, default=0.01)
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--profile")
    profiles.add_argument("--profiles", nargs="+")
    parser.add_argument("--fog", action="store_true")
    parser.add_argument("--diplomacy", action="store_true")
    parser.add_argument(
        "--initial-relation",
        choices=("war", "neutral", "friend", "alliance"),
        default="neutral",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--initialize", type=Path)
    continuation.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    arguments = vars(parser.parse_args())
    if arguments["profile"] is None and arguments["profiles"] is None:
        arguments["profile"] = "classic_generic_2022"
    return TrainingConfig(**arguments)


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(asdict(configuration), default=str, sort_keys=True), flush=True)
    print(json.dumps(train(configuration), sort_keys=True), flush=True)
