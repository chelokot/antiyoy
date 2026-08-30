from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from antiyoy_rl import (
    OBSERVATION_VERSION,
    ProceduralConfig,
    ScenarioObjective,
    VectorEnv,
)
from antiyoy_rl.model import (
    ACTION_KIND_NAMES,
    RULE_FEATURES,
    UniversalPolicy,
    action_distribution,
    concatenate_observations,
    encode_rules_batch,
    load_policy_state,
    rotate_observation_180,
    select_environments,
)

try:
    from .build_bundle import BUNDLE_KIND, SUPPORTED_BUNDLE_VERSIONS
except ImportError:
    from build_bundle import BUNDLE_KIND, SUPPORTED_BUNDLE_VERSIONS
try:
    from .routes import select_bundle_expert
except ImportError:
    from routes import select_bundle_expert


CHECKPOINT_VERSION = 5


@dataclass(frozen=True)
class TrainingConfig:
    environments: int
    updates: int
    procedural: bool
    width: int
    height: int
    players: int
    players_schedule: list[int] | None
    map_size_schedule: list[tuple[int, int]] | None
    seed: int
    land_density_per_million: int
    land_density_schedule_per_million: list[int] | None
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
    imitation_reset_interval: int
    imitation_teacher: str
    imitation_search_replan: bool
    imitation_rollin: str
    imitation_symmetry_augmentation: bool
    imitation_reference_weight: float
    imitation_slice_weights: list[str]
    imitation_action_weights: list[str]
    imitation_policy_rollin_slices: list[str]
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
    fixed_opponent: str | None
    learner_seat: int
    opponent_minibatch: int
    opponent_reference_weight: float
    opponent_counterfactual_baseline: bool
    profile: str | None
    profiles: list[str] | None
    fog: bool
    diplomacy: bool
    initial_relation: str
    device: str
    initialize: Path | None
    initialize_profile: str | None
    initialize_generator: str | None
    initialize_players: int | None
    initialize_seat: int | None
    initialize_domain: str | None
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


@dataclass(frozen=True)
class EpisodeDecision:
    observation: dict[str, np.ndarray]
    action: int
    environment: int
    log_probability: float
    value: float


@dataclass(frozen=True)
class EpisodeTrajectory:
    decisions: tuple[EpisodeDecision, ...]
    outcome: float
    terminal_outcome: float
    baseline_outcome: float | None


def reward_tensor(
    result: dict[str, np.ndarray], config: TrainingConfig, device: torch.device
) -> Tensor:
    outcome = torch.as_tensor(result["outcomes"], dtype=torch.float32, device=device)
    territory = torch.as_tensor(
        result["territory_delta"], dtype=torch.float32, device=device
    )
    treasury = torch.as_tensor(
        result["treasury_delta"], dtype=torch.float32, device=device
    )
    units = torch.as_tensor(
        result["unit_strength_delta"], dtype=torch.float32, device=device
    )
    return (
        outcome
        + config.territory_weight * territory
        + config.treasury_weight * treasury
        + config.unit_weight * units
    )


def procedural_config(
    config: TrainingConfig,
    seed: int,
    players: int,
    width: int,
    height: int,
    land_density_per_million: int,
) -> ProceduralConfig:
    return ProceduralConfig(
        width=width,
        height=height,
        players=players,
        seed=seed,
        land_density_per_million=land_density_per_million,
        starting_province_size=config.starting_province_size,
        starting_money=config.starting_money,
        tree_density_per_million=config.tree_density_per_million,
        neutral_tower_density_per_million=config.neutral_tower_density_per_million,
        neutral_capital_density_per_million=config.neutral_capital_density_per_million,
        grave_density_per_million=config.grave_density_per_million,
    )


def make_environment(config: TrainingConfig) -> VectorEnv:
    generator = procedural_config(
        config,
        config.seed,
        config.players,
        config.width,
        config.height,
        config.land_density_per_million,
    )
    objective = (
        None
        if config.objective_json is None
        else ScenarioObjective.from_json(config.objective_json)
    )
    schedule = profile_schedule(config)
    scheduled_domains = (
        config.players_schedule is not None
        or config.map_size_schedule is not None
        or config.land_density_schedule_per_million is not None
    )
    if config.procedural and scheduled_domains:
        players = (
            [config.players]
            if config.players_schedule is None
            else config.players_schedule
        )
        densities = (
            [config.land_density_per_million]
            if config.land_density_schedule_per_million is None
            else config.land_density_schedule_per_million
        )
        map_sizes = (
            [(config.width, config.height)]
            if config.map_size_schedule is None
            else config.map_size_schedule
        )
        generators = [
            procedural_config(
                config,
                config.seed + index,
                players[index % len(players)],
                *map_sizes[index % len(map_sizes)],
                densities[index % len(densities)],
            )
            for index in range(config.environments)
        ]
        return VectorEnv.procedural_domains(
            schedule,
            generators,
            action_limit=config.action_limit,
            fog=config.fog,
            diplomacy=config.diplomacy,
            initial_relation=config.initial_relation,
            objective=objective,
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


def profile_schedule(config: TrainingConfig) -> list[str]:
    if config.profiles is None:
        return [cast(str, config.profile)] * config.environments
    return [
        config.profiles[index % len(config.profiles)]
        for index in range(config.environments)
    ]


def maximum_players(config: TrainingConfig) -> int:
    if config.players_schedule is None:
        return config.players
    return max(config.players_schedule)


def parsed_slice_weights(config: TrainingConfig) -> dict[tuple[str, int], float]:
    scheduled = set(profile_schedule(config))
    parsed: dict[tuple[str, int], float] = {}
    for specification in config.imitation_slice_weights:
        parts = specification.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError("imitation slice weights use PROFILE:SEAT:WEIGHT")
        profile, seat_text, weight_text = parts
        try:
            seat = int(seat_text)
            weight = float(weight_text)
        except ValueError as error:
            raise ValueError(
                "imitation slice weights use PROFILE:SEAT:WEIGHT"
            ) from error
        if profile not in scheduled:
            raise ValueError(f"imitation slice profile is not scheduled: {profile}")
        if seat < 0 or seat >= maximum_players(config):
            raise ValueError(f"imitation slice seat is out of range: {seat}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("imitation slice weight must be finite and positive")
        key = (profile, seat)
        if key in parsed:
            raise ValueError(f"duplicate imitation slice weight: {profile}:{seat}")
        parsed[key] = weight
    return parsed


def parsed_action_weights(config: TrainingConfig) -> dict[int, float]:
    action_kinds = {name: index for index, name in enumerate(ACTION_KIND_NAMES)}
    parsed: dict[int, float] = {}
    for specification in config.imitation_action_weights:
        kind_name, separator, weight_text = specification.rpartition(":")
        if not separator or kind_name not in action_kinds:
            raise ValueError("imitation action weights use ACTION_KIND:WEIGHT")
        try:
            weight = float(weight_text)
        except ValueError as error:
            raise ValueError(
                "imitation action weights use ACTION_KIND:WEIGHT"
            ) from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("imitation action weight must be finite and positive")
        kind = action_kinds[kind_name]
        if kind in parsed:
            raise ValueError(f"duplicate imitation action weight: {kind_name}")
        parsed[kind] = weight
    return parsed


def parsed_policy_rollin_slices(config: TrainingConfig) -> set[tuple[str, int]]:
    scheduled = set(profile_schedule(config))
    parsed: set[tuple[str, int]] = set()
    for specification in config.imitation_policy_rollin_slices:
        parts = specification.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError("imitation policy rollin slices use PROFILE:SEAT")
        profile, seat_text = parts
        try:
            seat = int(seat_text)
        except ValueError as error:
            raise ValueError(
                "imitation policy rollin slices use PROFILE:SEAT"
            ) from error
        if profile not in scheduled:
            raise ValueError(
                f"imitation policy rollin profile is not scheduled: {profile}"
            )
        if seat < 0 or seat >= maximum_players(config):
            raise ValueError(f"imitation policy rollin seat is out of range: {seat}")
        key = (profile, seat)
        if key in parsed:
            raise ValueError(
                f"duplicate imitation policy rollin slice: {profile}:{seat}"
            )
        parsed.add(key)
    return parsed


def imitation_weights(
    config: TrainingConfig,
    observation: dict[str, np.ndarray],
    device: torch.device,
) -> Tensor:
    configured = parsed_slice_weights(config)
    active_players = np.asarray(observation["active_players"], dtype=np.int64)
    weights = [
        configured.get((profile, int(active_player)), 1.0)
        for profile, active_player in zip(
            profile_schedule(config), active_players, strict=True
        )
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def imitation_action_weights(
    config: TrainingConfig,
    observation: dict[str, np.ndarray],
    actions: np.ndarray,
    device: torch.device,
) -> Tensor:
    configured = parsed_action_weights(config)
    offsets = np.asarray(observation["action_offsets"][:-1], dtype=np.int64)
    indices = offsets + np.asarray(actions, dtype=np.int64)
    kinds = np.asarray(observation["action_kinds"], dtype=np.uint8)[indices]
    weights = [configured.get(int(kind), 1.0) for kind in kinds]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def policy_rollin_mask(
    config: TrainingConfig,
    observation: dict[str, np.ndarray],
    device: torch.device,
) -> Tensor:
    configured = parsed_policy_rollin_slices(config)
    active_players = np.asarray(observation["active_players"], dtype=np.int64)
    selected = [
        (profile, int(active_player)) in configured
        for profile, active_player in zip(
            profile_schedule(config), active_players, strict=True
        )
    ]
    return torch.tensor(selected, dtype=torch.bool, device=device)


def validate_config(config: TrainingConfig) -> None:
    positive = {
        "environments": config.environments,
        "rollout_steps": config.rollout_steps,
        "epochs": config.epochs,
        "opponent_minibatch": config.opponent_minibatch,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"positive training values required: {', '.join(invalid)}")
    if config.clip_ratio <= 0:
        raise ValueError("clip_ratio must be positive")
    if config.imitation_updates < 0:
        raise ValueError("imitation_updates must not be negative")
    if config.imitation_reset_interval < 0:
        raise ValueError("imitation_reset_interval must not be negative")
    if config.imitation_reference_weight < 0:
        raise ValueError("imitation_reference_weight must not be negative")
    if config.opponent_reference_weight < 0:
        raise ValueError("opponent_reference_weight must not be negative")
    if config.fixed_opponent not in {None, "greedy", "search"}:
        raise ValueError("fixed_opponent must be greedy, search, or omitted")
    if config.opponent_counterfactual_baseline:
        if config.fixed_opponent is None:
            raise ValueError(
                "opponent_counterfactual_baseline requires a fixed opponent"
            )
        if config.environments < 2 or config.environments % 2 != 0:
            raise ValueError(
                "counterfactual fixed-opponent training requires an even environment count"
            )
    if config.profiles is not None and not config.profiles:
        raise ValueError("profiles must not be empty")
    if config.procedural and (config.players < 2 or config.players > 8):
        raise ValueError("procedural maps require between two and eight players")
    if config.players_schedule is not None:
        if not config.procedural:
            raise ValueError("players schedule requires procedural maps")
        if not config.players_schedule:
            raise ValueError("players schedule must not be empty")
        if any(players < 2 or players > 8 for players in config.players_schedule):
            raise ValueError("players schedule values must be between two and eight")
    player_counts = config.players_schedule or [config.players]
    if config.learner_seat < 0 or config.learner_seat >= min(player_counts):
        raise ValueError("learner_seat must exist in every scheduled domain")
    if config.map_size_schedule is not None:
        if not config.procedural:
            raise ValueError("map size schedule requires procedural maps")
        if not config.map_size_schedule:
            raise ValueError("map size schedule must not be empty")
        if any(width < 1 or height < 1 for width, height in config.map_size_schedule):
            raise ValueError("map size schedule dimensions must be positive")
    parsed_slice_weights(config)
    parsed_action_weights(config)
    policy_slices = parsed_policy_rollin_slices(config)
    if policy_slices and config.imitation_rollin != "teacher":
        raise ValueError("asymmetric policy rollin requires teacher rollin as its base")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must not be negative")
    if config.checkpoint_every > 0 and config.checkpoint is None:
        raise ValueError("checkpoint_every requires a checkpoint path")
    if config.updates < 0:
        raise ValueError("updates must not be negative")
    if config.updates == 0 and config.imitation_updates == 0:
        raise ValueError("at least one PPO or imitation update is required")
    if config.fixed_opponent is not None and config.updates == 0:
        raise ValueError("fixed-opponent training requires at least one update")
    if config.imitation_teacher not in {"greedy", "search"}:
        raise ValueError("imitation_teacher must be greedy or search")
    if config.imitation_search_replan and config.imitation_teacher != "search":
        raise ValueError("imitation_search_replan requires the search teacher")
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
    if config.initialize is not None and config.resume is not None:
        raise ValueError("initialize and resume are mutually exclusive")
    if config.initialize_profile is not None and config.initialize is None:
        raise ValueError("initialize_profile requires an initialization checkpoint")
    initialization_context = (
        config.initialize_generator,
        config.initialize_players,
        config.initialize_seat,
        config.initialize_domain,
    )
    if any(value is not None for value in initialization_context):
        if config.initialize is None:
            raise ValueError(
                "initialize route selectors require an initialization checkpoint"
            )
        if config.initialize_profile is None:
            raise ValueError("initialize route selectors require initialize_profile")
    if (config.initialize_generator is None) != (config.initialize_players is None):
        raise ValueError(
            "initialize_generator and initialize_players must be used together"
        )
    if config.initialize_seat is not None and config.initialize_players is None:
        raise ValueError("initialize_seat requires generator and player selectors")
    if config.initialize_domain is not None and config.initialize_seat is None:
        raise ValueError("initialize_domain requires initialize_seat")
    if (
        config.initialize_players is not None
        and not 2 <= config.initialize_players <= 8
    ):
        raise ValueError("initialize_players must be between two and eight")
    if config.initialize_seat is not None:
        selected_players = cast(int, config.initialize_players)
        if config.initialize_seat < 0 or config.initialize_seat >= selected_players:
            raise ValueError("initialize_seat must be in the selected player range")
    if config.land_density_schedule_per_million is not None:
        if not config.procedural:
            raise ValueError("land density schedule requires procedural maps")
        if not config.land_density_schedule_per_million:
            raise ValueError("land density schedule must not be empty")
    land_densities = (
        [config.land_density_per_million]
        if config.land_density_schedule_per_million is None
        else config.land_density_schedule_per_million
    )
    densities = [
        *land_densities,
        config.tree_density_per_million,
        config.neutral_tower_density_per_million,
        config.neutral_capital_density_per_million,
        config.grave_density_per_million,
    ]
    if config.procedural and any(
        density < 0 or density > 1_000_000 for density in densities
    ):
        raise ValueError("procedural densities must be between zero and one million")
    if config.procedural and sum(densities[len(land_densities) :]) > 1_000_000:
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


def collect_fixed_opponent_episodes(
    environment: VectorEnv,
    model: UniversalPolicy,
    rules: Tensor,
    config: TrainingConfig,
    device: torch.device,
    reset_seed: int,
    reference_model: UniversalPolicy | None = None,
) -> tuple[list[EpisodeTrajectory], int, int]:
    paired_baseline = config.opponent_counterfactual_baseline
    learner_environments = (
        config.environments // 2 if paired_baseline else config.environments
    )
    if paired_baseline and reference_model is None:
        raise ValueError("counterfactual collection requires a reference model")
    for learner_environment in range(learner_environments):
        environment.reset(learner_environment, reset_seed)
        if paired_baseline:
            environment.reset(learner_environment + learner_environments, reset_seed)
        reset_seed += 1
    finished = np.zeros(config.environments, dtype=np.bool_)
    decisions: list[list[EpisodeDecision]] = [[] for _ in range(learner_environments)]
    outcomes = np.zeros(config.environments, dtype=np.float32)
    environment_indices = np.arange(config.environments)
    environment_steps = 0
    while not bool(np.all(finished)):
        observation = environment.observe()
        active_players = np.asarray(observation["active_players"], dtype=np.int64)
        learner_active = active_players == config.learner_seat
        with torch.no_grad():
            logits, values = model(observation, rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            selected_actions = distribution.sample()
            log_probabilities = distribution.log_prob(selected_actions)
            policy_actions = selected_actions.cpu().numpy().astype(np.uint64)
            if paired_baseline:
                reference_logits, _ = reference_model(observation, rules)
                reference_distribution = action_distribution(
                    reference_logits, observation["action_offsets"]
                )
                reference_actions = (
                    reference_distribution.logits.argmax(dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.uint64)
                )
        if config.fixed_opponent == "search":
            opponent_actions = np.asarray(
                environment.search_actions(
                    node_budget=config.search_nodes,
                    beam_width=config.search_beam_width,
                    branch_width=config.search_branch_width,
                    maximum_actions_per_turn=config.search_maximum_actions_per_turn,
                    active_mask=np.asarray(
                        np.logical_and(~learner_active, ~finished), dtype=np.uint8
                    ),
                ),
                dtype=np.uint64,
            )
        else:
            opponent_actions = np.asarray(environment.greedy_actions(), dtype=np.uint64)
        actions = np.where(learner_active, policy_actions, opponent_actions)
        if paired_baseline:
            reference_active = np.logical_and(
                learner_active, environment_indices >= learner_environments
            )
            actions[reference_active] = reference_actions[reference_active]
        learner_policy_active = np.logical_and.reduce(
            (
                learner_active,
                ~finished,
                environment_indices < learner_environments,
            )
        )
        for environment_index in np.flatnonzero(learner_policy_active):
            decisions[environment_index].append(
                EpisodeDecision(
                    observation=select_environments(
                        observation, [int(environment_index)]
                    ),
                    action=int(policy_actions[environment_index]),
                    environment=int(environment_index),
                    log_probability=float(log_probabilities[environment_index].item()),
                    value=float(values[environment_index].item()),
                )
            )
        result = environment.step(actions)
        environment_steps += config.environments
        done = np.logical_or(result["terminal"], result["truncated"])
        for environment_index in np.flatnonzero(done):
            if not finished[environment_index]:
                winner = int(result["winners"][environment_index])
                if winner == 255:
                    winner = int(result["adjudicated_winners"][environment_index])
                if winner == 255:
                    outcomes[environment_index] = 0.0
                elif winner == config.learner_seat:
                    outcomes[environment_index] = 1.0
                else:
                    outcomes[environment_index] = -1.0
                finished[environment_index] = True
            environment.reset(int(environment_index), reset_seed)
            reset_seed += 1
    episodes = []
    for learner_environment, trajectory in enumerate(decisions):
        terminal_outcome = float(outcomes[learner_environment])
        baseline_outcome = (
            float(outcomes[learner_environment + learner_environments])
            if paired_baseline
            else None
        )
        credit = (
            (terminal_outcome - baseline_outcome) / 2
            if baseline_outcome is not None
            else terminal_outcome
        )
        episodes.append(
            EpisodeTrajectory(
                tuple(trajectory), credit, terminal_outcome, baseline_outcome
            )
        )
    return episodes, reset_seed, environment_steps


def fixed_opponent_advantages(
    returns: Tensor,
    behavior_values: Tensor,
    counterfactual_baseline: bool,
) -> Tensor:
    advantages = (
        returns.clone() if counterfactual_baseline else returns - behavior_values
    )
    if not counterfactual_baseline:
        advantages -= advantages.mean()
    return advantages / advantages.std(correction=0).clamp_min(1e-6)


def optimize_fixed_opponent_episodes(
    model: UniversalPolicy,
    reference_model: UniversalPolicy | None,
    optimizer: torch.optim.Optimizer,
    rules: Tensor,
    episodes: list[EpisodeTrajectory],
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float | int]:
    samples: list[tuple[EpisodeDecision, float]] = []
    for episode in episodes:
        decision_count = len(episode.decisions)
        for decision_index, decision in enumerate(episode.decisions):
            remaining = decision_count - decision_index - 1
            samples.append((decision, episode.outcome * config.gamma**remaining))
    if not samples:
        raise ValueError("fixed-opponent episodes contain no learner decisions")
    returns = torch.tensor(
        [sample_return for _, sample_return in samples],
        dtype=torch.float32,
        device=device,
    )
    behavior_values = torch.tensor(
        [decision.value for decision, _ in samples],
        dtype=torch.float32,
        device=device,
    )
    advantages = fixed_opponent_advantages(
        returns, behavior_values, config.opponent_counterfactual_baseline
    )
    loss_sum = 0.0
    entropy_sum = 0.0
    retention_sum = 0.0
    optimizer_steps = 0
    for _ in range(config.epochs):
        ordering = torch.randperm(len(samples)).tolist()
        for start in range(0, len(ordering), config.opponent_minibatch):
            indices = ordering[start : start + config.opponent_minibatch]
            batch = [samples[index][0] for index in indices]
            observation = concatenate_observations(
                [decision.observation for decision in batch]
            )
            batch_rules = torch.stack(
                [rules[decision.environment] for decision in batch]
            )
            logits, values = model(observation, batch_rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            actions = torch.tensor(
                [decision.action for decision in batch],
                dtype=torch.long,
                device=device,
            )
            log_probabilities = distribution.log_prob(actions)
            previous_log_probabilities = torch.tensor(
                [decision.log_probability for decision in batch],
                dtype=torch.float32,
                device=device,
            )
            ratio = torch.exp(log_probabilities - previous_log_probabilities)
            unclipped = ratio * advantages[indices]
            clipped = (
                torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
                * advantages[indices]
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (
                torch.zeros((), device=device)
                if config.opponent_counterfactual_baseline
                else (values - returns[indices]).square().mean()
            )
            entropy = distribution.entropy().mean()
            retention = torch.zeros((), device=device)
            if reference_model is not None and config.opponent_reference_weight > 0:
                with torch.no_grad():
                    reference_logits, _ = reference_model(observation, batch_rules)
                    reference_distribution = action_distribution(
                        reference_logits, observation["action_offsets"]
                    )
                retention = torch.distributions.kl_divergence(
                    reference_distribution, distribution
                ).mean()
            loss = (
                policy_loss
                + config.value_weight * value_loss
                - config.entropy_weight * entropy
                + config.opponent_reference_weight * retention
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.item())
            entropy_sum += float(entropy.item())
            retention_sum += float(retention.item())
    credits = np.asarray([episode.outcome for episode in episodes], dtype=np.float32)
    terminal_outcomes = np.asarray(
        [episode.terminal_outcome for episode in episodes], dtype=np.float32
    )
    baseline_outcomes = np.asarray(
        [
            episode.baseline_outcome
            for episode in episodes
            if episode.baseline_outcome is not None
        ],
        dtype=np.float32,
    )
    return {
        "episodes": len(episodes),
        "decisions": len(samples),
        "wins": int(np.count_nonzero(terminal_outcomes == 1)),
        "draws": int(np.count_nonzero(terminal_outcomes == 0)),
        "losses": int(np.count_nonzero(terminal_outcomes == -1)),
        "mean_outcome": float(terminal_outcomes.mean()),
        "mean_credit": float(credits.mean()),
        "baseline_episodes": len(baseline_outcomes),
        "baseline_wins": int(np.count_nonzero(baseline_outcomes == 1)),
        "baseline_draws": int(np.count_nonzero(baseline_outcomes == 0)),
        "baseline_losses": int(np.count_nonzero(baseline_outcomes == -1)),
        "baseline_mean_outcome": (
            float(baseline_outcomes.mean()) if len(baseline_outcomes) > 0 else 0.0
        ),
        "loss": loss_sum / optimizer_steps,
        "entropy": entropy_sum / optimizer_steps,
        "retention_kl": retention_sum / optimizer_steps,
        "optimizer_steps": optimizer_steps,
    }


def pretrain_teacher(
    environment: VectorEnv,
    model: UniversalPolicy,
    optimizer: torch.optim.Optimizer,
    rules: Tensor,
    config: TrainingConfig,
    reset_seed: int,
    device: torch.device,
    reference_model: UniversalPolicy | None = None,
    checkpoint_callback: Callable[[int, float, float], None] | None = None,
) -> tuple[int, float, float]:
    loss_average = 0.0
    accuracy_average = 0.0
    for update in range(1, config.imitation_updates + 1):
        observation = environment.observe()
        if config.imitation_teacher == "greedy":
            selected = environment.greedy_actions()
        elif config.imitation_search_replan:
            selected = environment.search_actions_replanned(
                node_budget=config.search_nodes,
                beam_width=config.search_beam_width,
                branch_width=config.search_branch_width,
                maximum_actions_per_turn=config.search_maximum_actions_per_turn,
            )
        else:
            selected = environment.search_actions(
                node_budget=config.search_nodes,
                beam_width=config.search_beam_width,
                branch_width=config.search_branch_width,
                maximum_actions_per_turn=config.search_maximum_actions_per_turn,
            )
        targets = torch.as_tensor(selected, dtype=torch.long, device=device)
        model_observation = observation
        if config.imitation_symmetry_augmentation:
            rotation_mask = (
                np.arange(config.environments, dtype=np.uint64) + update
            ) % 2 == 0
            model_observation = rotate_observation_180(observation, rotation_mask)
        logits, _ = model(model_observation, rules)
        distribution = action_distribution(logits, model_observation["action_offsets"])
        weights = imitation_weights(
            config, observation, device
        ) * imitation_action_weights(config, observation, selected, device)
        weight_sum = weights.sum()
        teacher_losses = -distribution.log_prob(targets)
        loss = (teacher_losses * weights).sum() / weight_sum
        retention_kl = 0.0
        if reference_model is not None:
            with torch.no_grad():
                reference_logits, _ = reference_model(model_observation, rules)
                reference_distribution = action_distribution(
                    reference_logits, model_observation["action_offsets"]
                )
            retention = torch.distributions.kl_divergence(
                reference_distribution, distribution
            )
            retention = (retention * weights).sum() / weight_sum
            loss = loss + config.imitation_reference_weight * retention
            retention_kl = float(retention.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        policy_actions = distribution.logits.argmax(dim=1)
        accuracy = float((policy_actions == targets).float().mean().item())
        loss_average += (float(loss.item()) - loss_average) / update
        accuracy_average += (accuracy - accuracy_average) / update
        if config.imitation_policy_rollin_slices:
            use_policy = policy_rollin_mask(config, observation, device)
            rollin_actions = torch.where(use_policy, policy_actions, targets)
        else:
            rollin_actions = (
                targets if config.imitation_rollin == "teacher" else policy_actions
            )
        result = environment.step(rollin_actions.cpu().numpy().astype(np.uint64))
        done = np.logical_or(result["terminal"], result["truncated"])
        reset_all = (
            config.imitation_reset_interval > 0
            and update % config.imitation_reset_interval == 0
        )
        reset_indices = (
            range(config.environments) if reset_all else np.flatnonzero(done)
        )
        for index in reset_indices:
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
                        "retention_kl": retention_kl,
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
    advantages = (advantages - advantages.mean()) / advantages.std(
        correction=0
    ).clamp_min(1e-6)
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
            clipped = (
                torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
                * advantages[step]
            )
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
    profile: str | None,
    generator: str | None,
    players: int | None,
    seat: int | None,
    domain: str | None,
) -> str:
    checkpoint = load_training_checkpoint(path, device, compatible=True)
    state, selected_expert = initialization_state(
        checkpoint, profile, generator, players, seat, domain
    )
    load_policy_state(model, state)
    return selected_expert


def initialization_state(
    checkpoint: dict[str, object],
    profile: str | None,
    generator: str | None = None,
    players: int | None = None,
    seat: int | None = None,
    domain: str | None = None,
) -> tuple[dict[str, Tensor], str]:
    state = checkpoint.get("model")
    if state is not None:
        selectors = (profile, generator, players, seat, domain)
        if any(value is not None for value in selectors):
            raise ValueError(
                "initialize route selectors are only valid for a policy bundle"
            )
        return state, "single"
    if checkpoint.get("kind") != BUNDLE_KIND:
        raise ValueError("initialization checkpoint has no policy weights")
    if checkpoint.get("bundle_version") not in SUPPORTED_BUNDLE_VERSIONS:
        raise ValueError("initialization policy bundle version does not match")
    if profile is None:
        raise ValueError("policy bundle initialization requires initialize_profile")
    selected_expert = select_bundle_expert(
        checkpoint, profile, generator, players, seat, domain
    )
    return checkpoint["experts"][selected_expert], selected_expert


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
    initialized_expert = ""
    if config.initialize is not None:
        initialized_expert = initialize_checkpoint(
            config.initialize,
            model,
            device,
            config.initialize_profile,
            config.initialize_generator,
            config.initialize_players,
            config.initialize_seat,
            config.initialize_domain,
        )
    elif config.resume is not None:
        restore_checkpoint(config.resume, model, optimizer, device)
    reference_model = None
    if (
        config.imitation_reference_weight > 0
        or config.opponent_reference_weight > 0
        or config.opponent_counterfactual_baseline
    ):
        reference_model = copy.deepcopy(model).eval()
        reference_model.requires_grad_(False)
    rules = encode_rules_batch(environment.rules_jsons(), device)
    reset_seed = config.seed + config.environments
    imitation_reset_seed = reset_seed
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
            reference_model,
            save_imitation_recovery,
        )
    imitation_environment_resets = reset_seed - imitation_reset_seed
    imitation_seconds = time.perf_counter() - imitation_started
    imitation_transitions = config.imitation_updates * config.environments
    reward_average = 0.0
    loss_average = 0.0
    training_transitions = 0
    training_optimizer_steps = 0
    opponent_episodes = 0
    opponent_wins = 0
    opponent_draws = 0
    opponent_losses = 0
    opponent_baseline_wins = 0
    opponent_baseline_draws = 0
    opponent_baseline_losses = 0
    for update in range(1, config.updates + 1):
        if config.fixed_opponent is None:
            rollout, reset_seed = collect_rollout(
                environment, model, rules, config, device, reset_seed
            )
            advantages, returns = rollout_targets(rollout, config)
            loss, entropy = optimize_rollout(
                model, optimizer, rules, rollout, advantages, returns, config
            )
            reward = float(rollout.rewards.mean().item())
            training_transitions += config.rollout_steps * config.environments
            training_optimizer_steps += config.rollout_steps * config.epochs
            progress = {
                "mean_value": float(rollout.values.mean().item()),
                "entropy": entropy,
            }
        else:
            episodes, reset_seed, environment_steps = collect_fixed_opponent_episodes(
                environment,
                model,
                rules,
                config,
                device,
                reset_seed,
                reference_model,
            )
            metrics = optimize_fixed_opponent_episodes(
                model,
                reference_model,
                optimizer,
                rules,
                episodes,
                config,
                device,
            )
            loss = float(metrics["loss"])
            reward = float(metrics["mean_credit"])
            training_transitions += environment_steps
            training_optimizer_steps += int(metrics["optimizer_steps"])
            opponent_episodes += int(metrics["episodes"])
            opponent_wins += int(metrics["wins"])
            opponent_draws += int(metrics["draws"])
            opponent_losses += int(metrics["losses"])
            opponent_baseline_wins += int(metrics["baseline_wins"])
            opponent_baseline_draws += int(metrics["baseline_draws"])
            opponent_baseline_losses += int(metrics["baseline_losses"])
            progress = {
                "episodes": metrics["episodes"],
                "decisions": metrics["decisions"],
                "wins": metrics["wins"],
                "draws": metrics["draws"],
                "losses": metrics["losses"],
                "mean_outcome": metrics["mean_outcome"],
                "mean_credit": metrics["mean_credit"],
                "baseline_wins": metrics["baseline_wins"],
                "baseline_draws": metrics["baseline_draws"],
                "baseline_losses": metrics["baseline_losses"],
                "entropy": metrics["entropy"],
                "retention_kl": metrics["retention_kl"],
            }
        reward_average += (reward - reward_average) / update
        loss_average += (loss - loss_average) / update
        if (
            config.fixed_opponent is not None
            or update == 1
            or update % 10 == 0
            or update == config.updates
        ):
            print(
                json.dumps(
                    {
                        "update": update,
                        "transitions": training_transitions,
                        "mean_reward": reward,
                        "loss": loss,
                        **progress,
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
                    "stage": (
                        "fixed_opponent" if config.fixed_opponent is not None else "ppo"
                    ),
                    "ppo_update": update,
                    "mean_reward": reward_average,
                    "mean_loss": loss_average,
                },
            )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    algorithm = (
        f"fixed_{config.fixed_opponent}_opponent_"
        f"{'counterfactual_' if config.opponent_counterfactual_baseline else ''}terminal_ppo"
        if config.fixed_opponent is not None
        else "perspective_ppo_gae"
    )
    if config.imitation_updates > 0:
        algorithm = (
            f"{config.imitation_teacher}_distilled_{config.imitation_rollin}_rollin"
        )
        if config.imitation_symmetry_augmentation:
            algorithm += "_rot180_augmented"
        if config.imitation_search_replan:
            algorithm += "_replanned_labels"
        if config.imitation_reset_interval > 0:
            algorithm += "_periodic_map_resets"
        if config.imitation_reference_weight > 0:
            algorithm += "_reference_regularized"
        if config.imitation_slice_weights:
            algorithm += "_slice_weighted"
        if config.imitation_action_weights:
            algorithm += "_action_weighted"
        if config.imitation_policy_rollin_slices:
            algorithm += "_asymmetric_dagger"
        if config.fixed_opponent is not None:
            algorithm += f"_fixed_{config.fixed_opponent}_opponent_"
            if config.opponent_counterfactual_baseline:
                algorithm += "counterfactual_"
            algorithm += "terminal_ppo"
        elif config.updates > 0:
            algorithm += "_perspective_ppo_gae"
    summary: dict[str, float | int | str] = {
        "algorithm": algorithm,
        "updates": config.updates,
        "environments": config.environments,
        "map_generator": "procedural_v1" if config.procedural else "symmetric_duel_v1",
        "players_schedule": ",".join(
            str(players) for players in config.players_schedule or []
        ),
        "map_size_schedule": ",".join(
            f"{width}x{height}" for width, height in config.map_size_schedule or []
        ),
        "land_density_schedule_per_million": ",".join(
            str(density) for density in config.land_density_schedule_per_million or []
        ),
        "parameters": parameters,
        "transitions": training_transitions,
        "optimizer_steps": training_optimizer_steps,
        "fixed_opponent": config.fixed_opponent or "",
        "learner_seat": config.learner_seat,
        "opponent_reference_weight": config.opponent_reference_weight,
        "opponent_counterfactual_baseline": config.opponent_counterfactual_baseline,
        "opponent_episodes": opponent_episodes,
        "opponent_wins": opponent_wins,
        "opponent_draws": opponent_draws,
        "opponent_losses": opponent_losses,
        "opponent_baseline_wins": opponent_baseline_wins,
        "opponent_baseline_draws": opponent_baseline_draws,
        "opponent_baseline_losses": opponent_baseline_losses,
        "imitation_updates": config.imitation_updates,
        "imitation_reset_interval": config.imitation_reset_interval,
        "imitation_environment_resets": imitation_environment_resets,
        "imitation_teacher": config.imitation_teacher,
        "imitation_search_replan": config.imitation_search_replan,
        "imitation_rollin": config.imitation_rollin,
        "imitation_symmetry_augmentation": config.imitation_symmetry_augmentation,
        "imitation_reference_weight": config.imitation_reference_weight,
        "imitation_slice_weights": ",".join(config.imitation_slice_weights),
        "imitation_action_weights": ",".join(config.imitation_action_weights),
        "imitation_policy_rollin_slices": ",".join(
            config.imitation_policy_rollin_slices
        ),
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
        "initialized_from": str(config.initialize)
        if config.initialize is not None
        else "",
        "initialized_expert": initialized_expert,
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


def parse_map_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x")
        width = int(width_text)
        height = int(height_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("map sizes use WIDTHxHEIGHT") from error
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("map size dimensions must be positive")
    return width, height


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--procedural", action="store_true")
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--players-schedule", type=int, nargs="+")
    parser.add_argument("--map-size-schedule", type=parse_map_size, nargs="+")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--land-density-per-million", type=int, default=650_000)
    parser.add_argument("--land-density-schedule-per-million", type=int, nargs="+")
    parser.add_argument("--starting-province-size", type=int, default=5)
    parser.add_argument("--starting-money", type=int, default=10)
    parser.add_argument("--tree-density-per-million", type=int, default=150_000)
    parser.add_argument("--neutral-tower-density-per-million", type=int, default=20_000)
    parser.add_argument(
        "--neutral-capital-density-per-million", type=int, default=10_000
    )
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
    parser.add_argument("--imitation-reset-interval", type=int, default=0)
    parser.add_argument(
        "--imitation-teacher", choices=("greedy", "search"), default="greedy"
    )
    parser.add_argument("--imitation-search-replan", action="store_true")
    parser.add_argument(
        "--imitation-rollin", choices=("teacher", "policy"), default="teacher"
    )
    parser.add_argument("--imitation-symmetry-augmentation", action="store_true")
    parser.add_argument("--imitation-reference-weight", type=float, default=0.0)
    parser.add_argument(
        "--imitation-slice-weight",
        action="append",
        dest="imitation_slice_weights",
        default=[],
        metavar="PROFILE:SEAT:WEIGHT",
    )
    parser.add_argument(
        "--imitation-action-weight",
        action="append",
        dest="imitation_action_weights",
        default=[],
        metavar="ACTION_KIND:WEIGHT",
    )
    parser.add_argument(
        "--imitation-policy-rollin-slice",
        action="append",
        dest="imitation_policy_rollin_slices",
        default=[],
        metavar="PROFILE:SEAT",
    )
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
    parser.add_argument("--fixed-opponent", choices=("greedy", "search"))
    parser.add_argument("--learner-seat", type=int, default=0)
    parser.add_argument("--opponent-minibatch", type=int, default=256)
    parser.add_argument("--opponent-reference-weight", type=float, default=0.0)
    parser.add_argument("--opponent-counterfactual-baseline", action="store_true")
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
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--initialize", type=Path)
    continuation.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-profile")
    parser.add_argument("--initialize-generator")
    parser.add_argument("--initialize-players", type=int)
    parser.add_argument("--initialize-seat", type=int)
    parser.add_argument("--initialize-domain")
    parser.add_argument("--checkpoint", type=Path)
    arguments = vars(parser.parse_args())
    if arguments["profile"] is None and arguments["profiles"] is None:
        arguments["profile"] = "classic_generic_2022"
    return TrainingConfig(**arguments)


if __name__ == "__main__":
    configuration = parse_args()
    print(json.dumps(asdict(configuration), default=str, sort_keys=True), flush=True)
    print(json.dumps(train(configuration), sort_keys=True), flush=True)
