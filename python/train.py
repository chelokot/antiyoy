from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from antiyoy_rl import OBSERVATION_VERSION, VectorEnv
from antiyoy_rl.model import (
    RULE_FEATURES,
    UniversalPolicy,
    action_distribution,
    encode_rules_batch,
)


CHECKPOINT_VERSION = 4


@dataclass(frozen=True)
class TrainingConfig:
    environments: int
    updates: int
    width: int
    height: int
    seed: int
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
    entropy_weight: float
    value_weight: float
    territory_weight: float
    treasury_weight: float
    unit_weight: float
    profile: str | None
    profiles: list[str] | None
    fog: bool
    device: str
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
    if config.profiles is None:
        return VectorEnv(
            config.environments,
            width=config.width,
            height=config.height,
            seed=config.seed,
            action_limit=config.action_limit,
            profile=cast(str, config.profile),
            fog=config.fog,
        )
    schedule = [
        config.profiles[index % len(config.profiles)]
        for index in range(config.environments)
    ]
    return VectorEnv.mixed(
        schedule,
        width=config.width,
        height=config.height,
        seed=config.seed,
        action_limit=config.action_limit,
        fog=config.fog,
    )


def validate_config(config: TrainingConfig) -> None:
    positive = {
        "environments": config.environments,
        "updates": config.updates,
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
    if config.profiles is not None and not config.profiles:
        raise ValueError("profiles must not be empty")


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


def pretrain_greedy(
    environment: VectorEnv,
    model: UniversalPolicy,
    optimizer: torch.optim.Optimizer,
    rules: Tensor,
    updates: int,
    reset_seed: int,
    device: torch.device,
) -> tuple[int, float, float]:
    loss_average = 0.0
    accuracy_average = 0.0
    for update in range(1, updates + 1):
        observation = environment.observe()
        targets = torch.as_tensor(
            environment.greedy_actions(), dtype=torch.long, device=device
        )
        logits, _ = model(observation, rules)
        distribution = action_distribution(logits, observation["action_offsets"])
        loss = -distribution.log_prob(targets).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        accuracy = float((distribution.logits.argmax(dim=1) == targets).float().mean().item())
        loss_average += (float(loss.item()) - loss_average) / update
        accuracy_average += (accuracy - accuracy_average) / update
        result = environment.step(targets.cpu().numpy().astype(np.uint64))
        done = np.logical_or(result["terminal"], result["truncated"])
        for index in np.flatnonzero(done):
            environment.reset(int(index), reset_seed)
            reset_seed += 1
        if update == 1 or update % 100 == 0 or update == updates:
            print(
                json.dumps(
                    {
                        "stage": "greedy_distillation",
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
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("resume checkpoint format does not match this trainer")
    if checkpoint["observation_version"] != OBSERVATION_VERSION:
        raise ValueError("resume checkpoint observation contract does not match")
    if checkpoint["rule_features"] != RULE_FEATURES:
        raise ValueError("resume checkpoint rule feature width does not match")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def train(config: TrainingConfig) -> dict[str, float | int | str]:
    validate_config(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    environment = make_environment(config)
    model = UniversalPolicy(config.hidden, config.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if config.resume is not None:
        restore_checkpoint(config.resume, model, optimizer, device)
    rules = encode_rules_batch(environment.rules_jsons(), device)
    reset_seed = config.seed + config.environments
    imitation_loss = 0.0
    imitation_accuracy = 0.0
    if config.imitation_updates > 0:
        reset_seed, imitation_loss, imitation_accuracy = pretrain_greedy(
            environment,
            model,
            optimizer,
            rules,
            config.imitation_updates,
            reset_seed,
            device,
        )
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
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary: dict[str, float | int | str] = {
        "algorithm": "greedy_distilled_perspective_ppo_gae"
        if config.imitation_updates > 0
        else "perspective_ppo_gae",
        "updates": config.updates,
        "environments": config.environments,
        "parameters": parameters,
        "transitions": config.updates * config.rollout_steps * config.environments,
        "optimizer_steps": config.updates * config.rollout_steps * config.epochs,
        "imitation_updates": config.imitation_updates,
        "imitation_loss": imitation_loss,
        "imitation_accuracy": imitation_accuracy,
        "mean_reward": reward_average,
        "mean_loss": loss_average,
        "device": str(device),
        "resumed_from": str(config.resume) if config.resume is not None else "",
    }
    if config.checkpoint is not None:
        config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "checkpoint_version": CHECKPOINT_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "rule_features": RULE_FEATURES,
                "training_rules_jsons": environment.rules_jsons(),
                "config": {
                    **asdict(config),
                    "resume": str(config.resume) if config.resume is not None else None,
                    "checkpoint": str(config.checkpoint),
                },
                "summary": summary,
            },
            config.checkpoint,
        )
        summary["checkpoint"] = str(config.checkpoint)
    return summary


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1)
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
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--territory-weight", type=float, default=0.03)
    parser.add_argument("--treasury-weight", type=float, default=0.002)
    parser.add_argument("--unit-weight", type=float, default=0.01)
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--profile")
    profiles.add_argument("--profiles", nargs="+")
    parser.add_argument("--fog", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    arguments = vars(parser.parse_args())
    if arguments["profile"] is None and arguments["profiles"] is None:
        arguments["profile"] = "classic_generic_2022"
    return TrainingConfig(**arguments)


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(asdict(configuration), default=str, sort_keys=True), flush=True)
    print(json.dumps(train(configuration), sort_keys=True), flush=True)
