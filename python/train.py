from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from antiyoy_rl import OBSERVATION_VERSION, VectorEnv
from antiyoy_rl.model import RULE_FEATURES, UniversalPolicy, action_distribution, encode_rules


CHECKPOINT_VERSION = 1


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
    entropy_weight: float
    value_weight: float
    territory_weight: float
    treasury_weight: float
    unit_weight: float
    profile: str
    device: str
    checkpoint: Path | None


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


def train(config: TrainingConfig) -> dict[str, float | int | str]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    environment = VectorEnv(
        config.environments,
        width=config.width,
        height=config.height,
        seed=config.seed,
        action_limit=config.action_limit,
        profile=config.profile,
    )
    model = UniversalPolicy(config.hidden, config.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    rules = encode_rules(environment.rules_json(), device)
    reset_seed = config.seed + config.environments
    reward_average = 0.0
    loss_average = 0.0
    for update in range(1, config.updates + 1):
        observation = environment.observe()
        logits, values = model(observation, rules)
        distribution = action_distribution(logits, observation["action_offsets"])
        actions = distribution.sample()
        log_probability = distribution.log_prob(actions)
        entropy = distribution.entropy()
        result = environment.step(actions.detach().cpu().numpy().astype(np.uint64))
        done = np.logical_or(result["terminal"], result["truncated"])
        for index in np.flatnonzero(done):
            environment.reset(int(index), reset_seed)
            reset_seed += 1
        next_observation = environment.observe()
        with torch.no_grad():
            _, next_values = model(next_observation, rules)
        rewards = reward_tensor(result, config, device)
        actors = torch.as_tensor(result["actors"], dtype=torch.long, device=device)
        next_players = torch.as_tensor(
            next_observation["active_players"], dtype=torch.long, device=device
        )
        perspective = torch.where(actors == next_players, 1.0, -1.0)
        continuation = torch.as_tensor(~done, dtype=torch.float32, device=device)
        target = rewards + config.gamma * perspective * next_values * continuation
        advantage = target - values
        policy_loss = -(log_probability * advantage.detach()).mean()
        value_loss = advantage.square().mean()
        loss = policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        reward_average += (float(rewards.mean().item()) - reward_average) / update
        loss_average += (float(loss.item()) - loss_average) / update
        if update == 1 or update % 100 == 0 or update == config.updates:
            print(
                json.dumps(
                    {
                        "update": update,
                        "mean_reward": float(rewards.mean().item()),
                        "mean_value": float(values.mean().item()),
                        "entropy": float(entropy.mean().item()),
                        "loss": float(loss.item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    summary: dict[str, float | int | str] = {
        "updates": config.updates,
        "environments": config.environments,
        "parameters": parameters,
        "mean_reward": reward_average,
        "mean_loss": loss_average,
        "device": str(device),
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
                "rules_json": environment.rules_json(),
                "config": {**asdict(config), "checkpoint": str(config.checkpoint)},
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
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--territory-weight", type=float, default=0.03)
    parser.add_argument("--treasury-weight", type=float, default=0.002)
    parser.add_argument("--unit-weight", type=float, default=0.01)
    parser.add_argument("--profile", default="classic_generic_2022")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    return TrainingConfig(**vars(arguments))


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(asdict(configuration), default=str, sort_keys=True), flush=True)
    print(json.dumps(train(configuration), sort_keys=True), flush=True)
