from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from antiyoy_rl.model import action_distribution, encode_rules_batch
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
    from .distill_vector_value import (
        VectorValueDistillationConfig,
        create_environment,
        domain_descriptor,
        load_routed_policy,
    )
    from .evaluate import load_policy_checkpoint
except ImportError:
    from build_bundle import digest
    from distill_vector_value import (
        VectorValueDistillationConfig,
        create_environment,
        domain_descriptor,
        load_routed_policy,
    )
    from evaluate import load_policy_checkpoint


@dataclass(frozen=True)
class VectorOutcomeSample:
    features: Tensor
    targets: Tensor


@dataclass(frozen=True)
class VectorOutcomeCalibrationConfig:
    profile: str = "classic_generic_2022"
    players: int = 5
    games: int = 320
    validation_games: int = 64
    seed: int = 1_110_000
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
    sample_stride: int = 4
    epochs: int = 16
    batch_size: int = 512
    learning_rate: float = 3e-4
    exploration_probability: float = 0.15
    exploration_top_k: int = 4
    target_mode: str = "binary"


def validate_config(config: VectorOutcomeCalibrationConfig) -> None:
    if config.players < 2 or config.players > MAX_PLAYERS:
        raise ValueError("vector outcome player count must be between two and eight")
    if config.games < 2 or not 0 < config.validation_games < config.games:
        raise ValueError("validation games must be a non-empty strict subset")
    if config.width < 3 or config.height < 3 or config.action_limit < 1:
        raise ValueError("vector outcome arena dimensions are invalid")
    if min(config.sample_stride, config.epochs, config.batch_size) < 1:
        raise ValueError("vector outcome training sizes must be positive")
    if config.learning_rate <= 0:
        raise ValueError("vector outcome learning rate must be positive")
    if not 0 <= config.exploration_probability <= 1:
        raise ValueError("vector outcome exploration probability is invalid")
    if config.exploration_top_k < 2:
        raise ValueError("vector outcome exploration requires at least two actions")
    if config.target_mode not in ("binary", "zero_sum"):
        raise ValueError("vector outcome target mode is unsupported")
    densities = (
        config.land_density_per_million,
        config.tree_density_per_million,
        config.neutral_tower_density_per_million,
        config.neutral_capital_density_per_million,
        config.grave_density_per_million,
    )
    if any(density < 0 or density > 1_000_000 for density in densities):
        raise ValueError("vector outcome map density is invalid")
    if config.starting_province_size < 1 or config.starting_money < 0:
        raise ValueError("vector outcome starting economy is invalid")


def distillation_config(
    config: VectorOutcomeCalibrationConfig,
) -> VectorValueDistillationConfig:
    return VectorValueDistillationConfig(
        profile=config.profile,
        players=config.players,
        environments=config.games,
        updates=1,
        validation_environments=config.validation_games,
        validation_steps=1,
        seed=config.seed,
        device=config.device,
        width=config.width,
        height=config.height,
        action_limit=config.action_limit,
        land_density_per_million=config.land_density_per_million,
        starting_province_size=config.starting_province_size,
        starting_money=config.starting_money,
        tree_density_per_million=config.tree_density_per_million,
        neutral_tower_density_per_million=config.neutral_tower_density_per_million,
        neutral_capital_density_per_million=(
            config.neutral_capital_density_per_million
        ),
        grave_density_per_million=config.grave_density_per_million,
        learning_rate=config.learning_rate,
    )


def relative_outcome_targets(
    winner: int,
    active_player: int,
    players: int,
    target_mode: str,
) -> Tensor:
    if winner == 255:
        return torch.zeros(players)
    loser_value = -1.0 if target_mode == "binary" else -1.0 / (players - 1)
    absolute = torch.full((players,), loser_value)
    absolute[winner] = 1.0
    indices = torch.remainder(torch.arange(players) + active_player, players)
    return absolute[indices]


def exploratory_actions(
    logits: Tensor,
    action_offsets: np.ndarray,
    random: np.random.Generator,
    probability: float,
    top_k: int,
    active: np.ndarray,
) -> tuple[np.ndarray, int]:
    distribution = action_distribution(logits, action_offsets)
    probabilities = distribution.probs.cpu().numpy()
    actions = distribution.logits.argmax(dim=1).cpu().numpy().astype(np.uint64)
    explore = np.logical_and(active, random.random(active.size) < probability)
    counts = np.diff(action_offsets)
    exploratory = 0
    for environment_index in np.flatnonzero(explore):
        legal_actions = int(counts[environment_index])
        candidate_count = min(top_k, legal_actions)
        if candidate_count < 2:
            continue
        ranked = np.argsort(
            -probabilities[environment_index, :legal_actions], kind="stable"
        )
        candidates = ranked[1:candidate_count]
        candidate_probabilities = probabilities[environment_index, candidates]
        mass = float(candidate_probabilities.sum())
        weights = (
            candidate_probabilities / mass
            if mass > 0
            else np.full(len(candidates), 1 / len(candidates))
        )
        actions[environment_index] = random.choice(candidates, p=weights)
        exploratory += 1
    return actions, exploratory


def collect_outcome_samples(
    policy: RoutedPolicy,
    config: VectorOutcomeCalibrationConfig,
    checkpoint_config: dict[str, object],
    device: torch.device,
) -> tuple[list[list[VectorOutcomeSample]], dict[str, object]]:
    runtime = distillation_config(config)
    environment = create_environment(
        runtime, checkpoint_config, config.games, config.seed
    )
    for environment_index in range(config.games):
        environment.reset(environment_index, config.seed + environment_index)
    rules = encode_rules_batch(environment.rules_jsons(), device)
    model = policy.models[policy.seat_experts[0]]
    finished = np.zeros(config.games, dtype=np.bool_)
    decisions = np.zeros(config.games, dtype=np.uint64)
    trajectories: list[list[tuple[Tensor, int]]] = [[] for _ in range(config.games)]
    samples: list[list[VectorOutcomeSample]] = [[] for _ in range(config.games)]
    winners = np.full(config.games, 255, dtype=np.uint8)
    truncations = 0
    transitions = 0
    exploratory = 0
    reset_seed = config.seed + config.games
    random = np.random.default_rng(config.seed ^ 0x51A7E)
    while not bool(finished.all()):
        observation = environment.observe()
        with torch.no_grad():
            logits, _, features = model.forward_with_value_features(observation, rules)
        active_players = np.asarray(observation["active_players"], dtype=np.uint8)
        for environment_index in np.flatnonzero(~finished):
            if decisions[environment_index] % config.sample_stride == 0:
                trajectories[environment_index].append(
                    (
                        features[environment_index].detach().cpu(),
                        int(active_players[environment_index]),
                    )
                )
            decisions[environment_index] += 1
        actions, selected_exploratory = exploratory_actions(
            logits,
            observation["action_offsets"],
            random,
            config.exploration_probability,
            config.exploration_top_k,
            np.logical_not(finished),
        )
        exploratory += selected_exploratory
        result = environment.step(actions)
        transitions += config.games
        done = np.logical_or(result["terminal"], result["truncated"])
        for environment_index in np.flatnonzero(done):
            if not finished[environment_index]:
                winner = int(result["winners"][environment_index])
                if bool(result["truncated"][environment_index]):
                    truncations += 1
                    winner = int(result["adjudicated_winners"][environment_index])
                winners[environment_index] = winner
                samples[environment_index] = [
                    VectorOutcomeSample(
                        features=sample_features,
                        targets=relative_outcome_targets(
                            winner,
                            active_player,
                            config.players,
                            config.target_mode,
                        ),
                    )
                    for sample_features, active_player in trajectories[
                        environment_index
                    ]
                ]
                trajectories[environment_index].clear()
                finished[environment_index] = True
            environment.reset(int(environment_index), reset_seed)
            reset_seed += 1
    winner_counts = np.bincount(winners[winners != 255], minlength=config.players)
    return samples, {
        "games": config.games,
        "samples": sum(len(game) for game in samples),
        "transitions": transitions,
        "exploratory_actions": exploratory,
        "truncations": truncations,
        "draws": int(np.count_nonzero(winners == 255)),
        "wins_by_seat": winner_counts.tolist(),
    }


def value_metrics(
    head: RelativeValueHead,
    samples: list[VectorOutcomeSample],
    players: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    squared_error = 0.0
    absolute_error = 0.0
    correct_sign = 0
    labels = 0
    head.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            features = torch.stack([sample.features for sample in batch]).to(device)
            targets = torch.stack([sample.targets for sample in batch]).to(device)
            predictions = head(features)[:, 1:players].clamp(-1, 1)
            expected = targets[:, 1:players]
            difference = predictions - expected
            squared_error += float(difference.square().sum().item())
            absolute_error += float(difference.abs().sum().item())
            correct_sign += int(
                (torch.sign(predictions) == torch.sign(expected)).sum().item()
            )
            labels += expected.numel()
    return {
        "samples": len(samples),
        "labels": labels,
        "mse": squared_error / labels,
        "mae": absolute_error / labels,
        "sign_accuracy": correct_sign / labels,
    }


def train_outcome_head(
    head: RelativeValueHead,
    training: list[VectorOutcomeSample],
    validation: list[VectorOutcomeSample],
    config: VectorOutcomeCalibrationConfig,
    device: torch.device,
) -> dict[str, object]:
    if not training or not validation:
        raise ValueError("vector outcome calibration requires non-empty samples")
    before_training = value_metrics(
        head, training, config.players, config.batch_size, device
    )
    before_validation = value_metrics(
        head, validation, config.players, config.batch_size, device
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    random = np.random.default_rng(config.seed)
    losses: list[float] = []
    head.train()
    for _ in range(config.epochs):
        indices = random.permutation(len(training))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            features = torch.stack(
                [training[int(index)].features for index in selected]
            ).to(device)
            targets = torch.stack(
                [training[int(index)].targets for index in selected]
            ).to(device)
            predictions = head(features)[:, 1 : config.players]
            loss = functional.mse_loss(predictions, targets[:, 1 : config.players])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            batches += 1
            epoch_loss += (float(loss.item()) - epoch_loss) / batches
        losses.append(epoch_loss)
    return {
        "training_before": before_training,
        "training_after": value_metrics(
            head, training, config.players, config.batch_size, device
        ),
        "validation_before": before_validation,
        "validation_after": value_metrics(
            head, validation, config.players, config.batch_size, device
        ),
        "epoch_losses": losses,
    }


def calibrate_vector_outcomes(
    checkpoint_path: Path,
    output_path: Path,
    config: VectorOutcomeCalibrationConfig,
) -> dict[str, object]:
    validate_config(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    checkpoint_config = dict(checkpoint["config"])
    runtime = distillation_config(config)
    descriptor = domain_descriptor(runtime, checkpoint_config)
    policy, experts, hidden, layers = load_routed_policy(
        checkpoint, runtime, descriptor, device
    )
    if len(set(experts)) != 1:
        raise ValueError("vector outcome calibration requires one shared expert")
    started = time.perf_counter()
    game_samples, collection = collect_outcome_samples(
        policy, config, checkpoint_config, device
    )
    training_games = config.games - config.validation_games
    training = [sample for game in game_samples[:training_games] for sample in game]
    validation = [sample for game in game_samples[training_games:] for sample in game]
    head = RelativeValueHead(hidden).to(device)
    initialize_from_scalar_value_head(head, policy.models[experts[0]].value_head)
    calibration = train_outcome_head(head, training, validation, config, device)
    seconds = time.perf_counter() - started
    source = {
        "path": str(checkpoint_path),
        "sha256": digest(checkpoint_path),
        "seat_experts": experts,
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "outcome_grounded_vector_value_calibration",
        "source": source,
        "configuration": asdict(config),
        "domain_descriptor": descriptor,
        "architecture": {"hidden": hidden, "layers": layers},
        "training_games": training_games,
        "validation_games": config.validation_games,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "collection": collection,
        "calibration": calibration,
        "seconds": seconds,
    }
    artifact = {
        "kind": VECTOR_VALUE_ARTIFACT_KIND,
        "artifact_version": VECTOR_VALUE_ARTIFACT_VERSION,
        "source": source,
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
    parser.add_argument("--games", type=int, default=320)
    parser.add_argument("--validation-games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1_110_000)
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
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--exploration-probability", type=float, default=0.15)
    parser.add_argument("--exploration-top-k", type=int, default=4)
    parser.add_argument(
        "--target-mode", choices=("binary", "zero_sum"), default="binary"
    )
    arguments = parser.parse_args()
    configuration = vars(arguments).copy()
    checkpoint_path = configuration.pop("checkpoint")
    output_path = configuration.pop("output")
    report = calibrate_vector_outcomes(
        checkpoint_path,
        output_path,
        VectorOutcomeCalibrationConfig(**configuration),
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
