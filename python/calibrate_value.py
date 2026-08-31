from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from antiyoy_rl import ProceduralConfig, VectorEnv
from antiyoy_rl.model import (
    UniversalPolicy,
    action_distribution,
    concatenate_observations,
    domain_key,
    encode_rules,
    encode_rules_batch,
    select_environments,
)
from antiyoy_rl.routed import RoutedPolicy

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
class ValueSample:
    observation: dict[str, np.ndarray]
    target: float


def outcome_target(winner: int, active_player: int) -> float:
    if winner == 255:
        return 0.0
    return 1.0 if winner == active_player else -1.0


def value_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 1 or predicted.size == 0:
        raise ValueError("value metrics require equal non-empty vectors")
    error = predicted - expected
    correlation = (
        float(np.corrcoef(predicted, expected)[0, 1])
        if np.std(predicted) > 0 and np.std(expected) > 0
        else None
    )
    return {
        "samples": int(expected.size),
        "mean_prediction": float(predicted.mean()),
        "mean_target": float(expected.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "sign_accuracy": float((np.sign(predicted) == np.sign(expected)).mean()),
        "correlation": correlation,
    }


def predict_values(
    model: UniversalPolicy,
    samples: list[ValueSample],
    rule_features: Tensor,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            observation = concatenate_observations(
                [sample.observation for sample in batch]
            )
            _, values = model(observation, rule_features)
            predictions.append(values.clamp(-1, 1).cpu().numpy())
    return np.concatenate(predictions)


def train_value_head(
    model: UniversalPolicy,
    training: list[ValueSample],
    validation: list[ValueSample],
    rule_features: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    if not training or not validation:
        raise ValueError("value calibration requires training and validation samples")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("value calibration hyperparameters must be positive")
    preserved = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("value_head.")
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.value_head.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    random = np.random.default_rng(seed)
    torch.manual_seed(seed)
    before_training = value_metrics(
        predict_values(model, training, rule_features, batch_size),
        np.asarray([sample.target for sample in training]),
    )
    before_validation = value_metrics(
        predict_values(model, validation, rule_features, batch_size),
        np.asarray([sample.target for sample in validation]),
    )
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        indices = random.permutation(len(training))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(indices), batch_size):
            selected_indices = indices[start : start + batch_size]
            selected = [training[int(index)] for index in selected_indices]
            observation = concatenate_observations(
                [sample.observation for sample in selected]
            )
            targets = torch.tensor(
                [sample.target for sample in selected],
                dtype=torch.float32,
                device=rule_features.device,
            )
            _, values = model(observation, rule_features)
            loss = torch.nn.functional.mse_loss(values, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batches += 1
            epoch_loss += (float(loss.item()) - epoch_loss) / batches
        losses.append(epoch_loss)
    model.eval()
    for key, value in model.state_dict().items():
        if key.startswith("value_head."):
            continue
        if not torch.equal(value.detach().cpu(), preserved[key]):
            raise RuntimeError(f"value calibration changed policy parameter: {key}")
    return {
        "training_before": before_training,
        "training_after": value_metrics(
            predict_values(model, training, rule_features, batch_size),
            np.asarray([sample.target for sample in training]),
        ),
        "validation_before": before_validation,
        "validation_after": value_metrics(
            predict_values(model, validation, rule_features, batch_size),
            np.asarray([sample.target for sample in validation]),
        ),
        "epoch_losses": losses,
    }


def collect_self_play_samples(
    environment: VectorEnv,
    policy: RoutedPolicy,
    rule_features: Tensor,
    seed: int,
    sample_stride: int,
    exploration_probability: float,
    exploration_top_k: int,
    players: int,
    training_seat: int | None,
) -> tuple[list[list[ValueSample]], dict[str, object]]:
    if sample_stride < 1:
        raise ValueError("sample stride must be positive")
    if not 0 <= exploration_probability <= 1 or exploration_top_k < 2:
        raise ValueError("exploration requires a probability and at least two actions")
    games = environment.environments
    for environment_index in range(games):
        environment.reset(environment_index, seed + environment_index)
    finished = np.zeros(games, dtype=np.bool_)
    decisions = np.zeros(games, dtype=np.uint64)
    trajectories: list[list[tuple[dict[str, np.ndarray], int]]] = [
        [] for _ in range(games)
    ]
    winners = np.full(games, 255, dtype=np.uint8)
    truncations = 0
    transitions = 0
    exploratory_actions = 0
    random = np.random.default_rng(seed ^ 0xC0FFEE)
    reset_seed = seed + games
    while not bool(finished.all()):
        observation = environment.observe()
        active_players = np.asarray(observation["active_players"], dtype=np.uint8)
        for environment_index in np.flatnonzero(~finished):
            active_player = int(active_players[environment_index])
            if decisions[environment_index] % sample_stride == 0 and (
                training_seat is None or active_player == training_seat
            ):
                trajectories[environment_index].append(
                    (
                        select_environments(observation, [int(environment_index)]),
                        active_player,
                    )
                )
            decisions[environment_index] += 1
        with torch.no_grad():
            logits, _ = policy(observation, rule_features)
            distribution = action_distribution(logits, observation["action_offsets"])
        probabilities = distribution.probs.cpu().numpy()
        actions = distribution.logits.argmax(dim=1).cpu().numpy().astype(np.uint64)
        explore = np.logical_and(
            ~finished,
            random.random(games) < exploration_probability,
        )
        counts = np.diff(observation["action_offsets"])
        for environment_index in np.flatnonzero(explore):
            legal_actions = int(counts[environment_index])
            candidate_count = min(exploration_top_k, legal_actions)
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
            exploratory_actions += 1
        result = environment.step(actions)
        transitions += games
        done = np.logical_or(result["terminal"], result["truncated"])
        for environment_index in np.flatnonzero(done):
            if not finished[environment_index]:
                winner = int(result["winners"][environment_index])
                if bool(result["truncated"][environment_index]):
                    truncations += 1
                    winner = int(result["adjudicated_winners"][environment_index])
                winners[environment_index] = winner
                finished[environment_index] = True
            environment.reset(int(environment_index), reset_seed)
            reset_seed += 1
    samples = [
        [
            ValueSample(
                observation,
                outcome_target(int(winners[environment_index]), active_player),
            )
            for observation, active_player in trajectory
        ]
        for environment_index, trajectory in enumerate(trajectories)
    ]
    winner_counts = np.bincount(winners[winners != 255], minlength=players)
    return samples, {
        "games": games,
        "samples": sum(len(trajectory) for trajectory in samples),
        "transitions": transitions,
        "exploratory_actions": exploratory_actions,
        "exploration_probability": exploration_probability,
        "exploration_top_k": exploration_top_k,
        "truncations": truncations,
        "draws": int(np.count_nonzero(winners == 255)),
        "wins_by_seat": winner_counts.tolist(),
        "training_seat": training_seat,
    }


def calibrate_value(
    checkpoint_path: Path,
    output_path: Path,
    profile: str,
    games: int,
    validation_games: int,
    seed: int,
    device_name: str,
    width: int,
    height: int,
    action_limit: int,
    sample_stride: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    exploration_probability: float = 0.15,
    exploration_top_k: int = 4,
    generator: str = "symmetric_duel_v1",
    players: int = 2,
    land_density_per_million: int = 650_000,
    starting_province_size: int = 5,
    starting_money: int = 10,
    tree_density_per_million: int = 150_000,
    neutral_tower_density_per_million: int = 20_000,
    neutral_capital_density_per_million: int = 10_000,
    grave_density_per_million: int = 15_000,
    training_seat: int | None = None,
) -> dict[str, object]:
    if games < 2 or validation_games < 1 or validation_games >= games:
        raise ValueError("validation games must be a non-empty strict subset")
    if generator not in ("symmetric_duel_v1", "procedural_v1"):
        raise ValueError("unsupported value calibration map generator")
    if players < 2 or players > 8:
        raise ValueError("value calibration player count must be between two and eight")
    if generator == "symmetric_duel_v1" and players != 2:
        raise ValueError("symmetric duel value calibration requires two players")
    if training_seat is not None and not 0 <= training_seat < players:
        raise ValueError("value calibration training seat is outside the player range")
    device = torch.device(device_name)
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    config = dict(checkpoint["config"])
    descriptor = {
        "width": width,
        "height": height,
        "players": players,
        "action_limit": action_limit,
        "fog": config["fog"],
        "diplomacy": config.get("diplomacy", False),
        "initial_relation": config.get("initial_relation", "neutral"),
    }
    if generator == "procedural_v1":
        descriptor.update(
            {
                "land_density_per_million": land_density_per_million,
                "starting_province_size": starting_province_size,
                "starting_money": starting_money,
                "tree_density_per_million": tree_density_per_million,
                "neutral_tower_density_per_million": neutral_tower_density_per_million,
                "neutral_capital_density_per_million": neutral_capital_density_per_million,
                "grave_density_per_million": grave_density_per_million,
            }
        )
    evaluation_domain = domain_key(generator, descriptor)
    models: dict[str, UniversalPolicy] = {}
    selected_experts: list[str] = []
    selected_config: dict[str, object] | None = None
    for seat in range(players):
        state, seat_config = select_policy_state(
            checkpoint,
            profile,
            generator,
            players,
            seat,
            evaluation_domain,
        )
        expert = str(seat_config["selected_expert"])
        if expert not in models:
            models[expert] = instantiate_policy(state, seat_config, device)
        selected_experts.append(expert)
        selected_config = seat_config
    if training_seat is None and len(set(selected_experts)) != 1:
        raise ValueError("all-seat value calibration requires one shared expert")
    target_seat = 0 if training_seat is None else training_seat
    target_expert = selected_experts[target_seat]
    environment_arguments = {
        "action_limit": action_limit,
        "profile": profile,
        "fog": bool(config["fog"]),
        "diplomacy": bool(config.get("diplomacy", False)),
        "initial_relation": str(config.get("initial_relation", "neutral")),
    }
    if generator == "procedural_v1":
        procedural_config = ProceduralConfig(
            width=width,
            height=height,
            players=players,
            seed=seed,
            land_density_per_million=land_density_per_million,
            starting_province_size=starting_province_size,
            starting_money=starting_money,
            tree_density_per_million=tree_density_per_million,
            neutral_tower_density_per_million=neutral_tower_density_per_million,
            neutral_capital_density_per_million=neutral_capital_density_per_million,
            grave_density_per_million=grave_density_per_million,
        )
        environment = VectorEnv.procedural(
            games, procedural_config, **environment_arguments
        )
    else:
        environment = VectorEnv(
            games,
            width=width,
            height=height,
            seed=seed,
            **environment_arguments,
        )
    rules = encode_rules_batch(environment.rules_jsons(), device)
    game_samples, collection = collect_self_play_samples(
        environment,
        RoutedPolicy(models, selected_experts),
        rules,
        seed,
        sample_stride,
        exploration_probability,
        exploration_top_k,
        players,
        training_seat,
    )
    training_games = games - validation_games
    training = [sample for game in game_samples[:training_games] for sample in game]
    validation = [sample for game in game_samples[training_games:] for sample in game]
    rule_features = encode_rules(environment.rules_jsons()[0], device)
    model = models[target_expert]
    calibration = train_value_head(
        model,
        training,
        validation,
        rule_features,
        epochs,
        batch_size,
        learning_rate,
        seed,
    )
    output_config = dict(selected_config or config)
    output_config.pop("selected_expert", None)
    output_config.pop("policy_kind", None)
    source_sha256 = digest(checkpoint_path)
    report = {
        "schema_version": 1,
        "kind": "monte_carlo_value_head_calibration",
        "source": {
            "path": str(checkpoint_path),
            "sha256": source_sha256,
            "expert": target_expert,
            "seat_experts": selected_experts,
        },
        "profile": profile,
        "generator": generator,
        "domain": evaluation_domain,
        "domain_descriptor": descriptor,
        "seed": seed,
        "training_games": training_games,
        "validation_games": validation_games,
        "sample_stride": sample_stride,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_seat": training_seat,
        "exploration_probability": exploration_probability,
        "exploration_top_k": exploration_top_k,
        "collection": collection,
        "calibration": calibration,
        "policy_parameters_frozen": True,
    }
    output = {
        "model": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "checkpoint_version": checkpoint["checkpoint_version"],
        "observation_version": checkpoint["observation_version"],
        "rule_features": checkpoint["rule_features"],
        "config": output_config,
        "summary": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(output, temporary)
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
    parser.add_argument(
        "--generator",
        choices=("symmetric_duel_v1", "procedural_v1"),
        default="symmetric_duel_v1",
    )
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--validation-games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=600_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
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
    parser.add_argument("--training-seat", type=int)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--exploration-probability", type=float, default=0.15)
    parser.add_argument("--exploration-top-k", type=int, default=4)
    arguments = parser.parse_args()
    report = calibrate_value(
        arguments.checkpoint,
        arguments.output,
        arguments.profile,
        arguments.games,
        arguments.validation_games,
        arguments.seed,
        arguments.device,
        arguments.width,
        arguments.height,
        arguments.action_limit,
        arguments.sample_stride,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.exploration_probability,
        arguments.exploration_top_k,
        arguments.generator,
        arguments.players,
        arguments.land_density_per_million,
        arguments.starting_province_size,
        arguments.starting_money,
        arguments.tree_density_per_million,
        arguments.neutral_tower_density_per_million,
        arguments.neutral_capital_density_per_million,
        arguments.grave_density_per_million,
        arguments.training_seat,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
