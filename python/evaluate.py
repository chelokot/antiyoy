from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from antiyoy_rl import OBSERVATION_VERSION, ProceduralConfig, VectorEnv
from antiyoy_rl.model import (
    ACTION_KIND_NAMES,
    RULE_FEATURES,
    UniversalPolicy,
    domain_key,
    encode_rules_batch,
    load_policy_state,
)
from antiyoy_rl.puct import (
    OpponentHorizon,
    PolicySearchConfig,
    SearchObjective,
    ValuePerspective,
    policy_search_actions,
)
from antiyoy_rl.routed import RoutedPolicy
from antiyoy_rl.vector_value import load_relative_value_head

try:
    from .build_bundle import BUNDLE_KIND, SUPPORTED_BUNDLE_VERSIONS, digest
except ImportError:
    from build_bundle import BUNDLE_KIND, SUPPORTED_BUNDLE_VERSIONS, digest
try:
    from .routes import select_bundle_expert
except ImportError:
    from routes import select_bundle_expert
try:
    from .train import CHECKPOINT_VERSION
except ImportError:
    from train import CHECKPOINT_VERSION


PAIRING_SCHEME = "adjacent_same_seed_opposite_seat_v1"
SEAT_ROTATION_SCHEME = "adjacent_same_seed_all_seats_v1"
FIXED_SEAT_SCHEME = "unique_seed_fixed_seat_v1"


class BaselineSelfPlay(TypedDict):
    games: int
    wins_by_seat: list[int]
    draws: int
    terminal_draws: int
    truncations: int
    winners: list[int]


def paired_elo(score: float, games: int) -> float:
    return relative_skill_delta(score, games, 2)


def relative_skill_delta(score: float, games: int, players: int) -> float:
    clipped_score = min(max(score, 0.5 / games), 1 - 0.5 / games)
    return 400 * math.log10(clipped_score * (players - 1) / (1 - clipped_score))


def baseline_adjusted_elo_delta(
    score: float, baseline_score: float, games: int
) -> float:
    minimum_score = 0.5 / games
    candidate = min(max(score, minimum_score), 1 - minimum_score)
    baseline = min(max(baseline_score, minimum_score), 1 - minimum_score)
    candidate_odds = candidate / (1 - candidate)
    baseline_odds = baseline / (1 - baseline)
    return 400 * math.log10(candidate_odds / baseline_odds)


def winner_score(winner: int, seat: int) -> float:
    if winner == 255:
        return 0.5
    return 1.0 if winner == seat else 0.0


def paired_method_comparison(
    candidate_scores: np.ndarray, baseline_scores: np.ndarray
) -> dict[str, float | int]:
    if candidate_scores.shape != baseline_scores.shape or candidate_scores.ndim != 1:
        raise ValueError("paired method comparison requires equal score vectors")
    candidate_better = int(np.count_nonzero(candidate_scores > baseline_scores))
    baseline_better = int(np.count_nonzero(candidate_scores < baseline_scores))
    same = int(candidate_scores.size - candidate_better - baseline_better)
    return paired_comparison_summary(candidate_better, baseline_better, same)


def paired_comparison_summary(
    candidate_better: int, baseline_better: int, same: int
) -> dict[str, float | int]:
    if candidate_better < 0 or baseline_better < 0 or same < 0:
        raise ValueError("paired comparison counts must be non-negative")
    discordant = candidate_better + baseline_better
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, outcome)
            for outcome in range(min(candidate_better, baseline_better) + 1)
        )
        p_value = min(1.0, 2 * tail / (2**discordant))
    return {
        "candidate_better": candidate_better,
        "baseline_better": baseline_better,
        "same": same,
        "discordant": discordant,
        "net_improvements": candidate_better - baseline_better,
        "exact_two_sided_sign_test_p": p_value,
    }


def paired_seeds(games: int, seed: int) -> np.ndarray:
    if games < 2 or games % 2 != 0:
        raise ValueError("paired evaluation requires a positive even number of games")
    return seed + np.arange(games, dtype=np.uint64) // 2


def seat_rotation_seeds(games: int, seed: int, players: int) -> np.ndarray:
    if players < 2:
        raise ValueError("seat rotation requires at least two players")
    if games < players or games % players != 0:
        raise ValueError("games must be a positive multiple of players")
    return seed + np.arange(games, dtype=np.uint64) // players


def evaluation_schedule(
    games: int,
    seed: int,
    players: int,
    model_seat: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if model_seat is None:
        return (
            seat_rotation_seeds(games, seed, players),
            np.arange(games, dtype=np.uint8) % players,
        )
    if games < 1:
        raise ValueError("fixed-seat evaluation requires at least one game")
    if model_seat < 0 or model_seat >= players:
        raise ValueError("model seat must belong to the player range")
    return (
        seed + np.arange(games, dtype=np.uint64),
        np.full(games, model_seat, dtype=np.uint8),
    )


def outcome_summary(
    games: int,
    wins: int,
    draws: int,
    losses: int,
    terminal_draws: int,
    truncations: int,
    players: int = 2,
    adjudications: int = 0,
) -> dict[str, float | int]:
    score = (wins + 0.5 * draws) / games
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "terminal_draws": terminal_draws,
        "truncations": truncations,
        "adjudications": adjudications,
        "losses": losses,
        "score": score,
        "elo_delta": relative_skill_delta(score, games, players),
    }


def reference_adjusted_outcome(
    outcome: dict[str, float | int],
    baseline_games: int,
    baseline_wins: int,
    baseline_draws: int,
    baseline_truncations: int,
) -> dict[str, float | int]:
    if baseline_games != int(outcome["games"]):
        raise ValueError("baseline and policy outcome counts must match")
    baseline_score = (baseline_wins + 0.5 * baseline_draws) / baseline_games
    baseline_truncation_rate = baseline_truncations / baseline_games
    return {
        **outcome,
        "baseline_wins": baseline_wins,
        "baseline_draws": baseline_draws,
        "baseline_truncations": baseline_truncations,
        "baseline_score": baseline_score,
        "score_delta": float(outcome["score"]) - baseline_score,
        "baseline_adjusted_elo_delta": baseline_adjusted_elo_delta(
            float(outcome["score"]), baseline_score, baseline_games
        ),
        "baseline_truncation_rate": baseline_truncation_rate,
        "truncation_rate_delta": (
            int(outcome["truncations"]) / baseline_games - baseline_truncation_rate
        ),
    }


def selected_action_kinds(
    observation: dict[str, np.ndarray], actions: np.ndarray
) -> np.ndarray:
    offsets = np.asarray(observation["action_offsets"][:-1], dtype=np.int64)
    indices = offsets + actions.astype(np.int64, copy=False)
    return np.asarray(observation["action_kinds"])[indices]


def named_action_counts(counts: np.ndarray) -> dict[str, int]:
    return {name: int(counts[index]) for index, name in enumerate(ACTION_KIND_NAMES)}


def choose_baseline_actions(
    environment: VectorEnv,
    observation: dict[str, np.ndarray],
    baseline: str,
    finished: np.ndarray,
    random: np.random.Generator,
    search_nodes: int,
    search_beam_width: int,
    search_branch_width: int,
    search_maximum_actions_per_turn: int,
) -> np.ndarray:
    if baseline == "search":
        return np.asarray(
            environment.search_actions(
                node_budget=search_nodes,
                beam_width=search_beam_width,
                branch_width=search_branch_width,
                maximum_actions_per_turn=search_maximum_actions_per_turn,
                active_mask=np.logical_not(finished).astype(np.uint8),
            ),
            dtype=np.uint64,
        )
    if baseline == "greedy":
        return np.asarray(environment.greedy_actions(), dtype=np.uint64)
    counts = np.diff(observation["action_offsets"])
    return np.array([random.integers(0, count) for count in counts], dtype=np.uint64)


def load_policy_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["checkpoint_version"] not in (4, CHECKPOINT_VERSION):
        raise ValueError("checkpoint format version does not match this evaluator")
    if checkpoint["observation_version"] not in (6, OBSERVATION_VERSION):
        raise ValueError(
            "checkpoint observation version does not match the native environment"
        )
    if checkpoint["rule_features"] not in (42, RULE_FEATURES):
        raise ValueError("checkpoint rule feature width does not match the policy")
    return checkpoint


def select_policy_state(
    checkpoint: dict[str, object],
    profile: str | None = None,
    generator: str | None = None,
    players: int | None = None,
    seat: int | None = None,
    domain: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    config = dict(checkpoint["config"])
    state = checkpoint.get("model")
    selected_expert = "single"
    if checkpoint.get("kind") == BUNDLE_KIND:
        if checkpoint.get("bundle_version") not in SUPPORTED_BUNDLE_VERSIONS:
            raise ValueError("policy bundle version does not match this evaluator")
        selected_profile = profile or config["profile"] or config["profiles"][0]
        selected_expert = select_bundle_expert(
            checkpoint, selected_profile, generator, players, seat, domain
        )
        state = checkpoint["experts"][selected_expert]
    if state is None:
        raise ValueError("checkpoint has no policy weights")
    config["policy_kind"] = checkpoint.get("kind", "single_policy")
    config["selected_expert"] = selected_expert
    return state, config


def instantiate_policy(
    state: dict[str, torch.Tensor],
    config: dict[str, object],
    device: torch.device,
) -> UniversalPolicy:
    model = UniversalPolicy(config["hidden"], config["layers"]).to(device)
    load_policy_state(model, state)
    model.eval()
    return model


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
    profile: str | None = None,
    generator: str | None = None,
    players: int | None = None,
    seat: int | None = None,
    domain: str | None = None,
) -> tuple[UniversalPolicy, dict[str, object]]:
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    state, config = select_policy_state(
        checkpoint, profile, generator, players, seat, domain
    )
    model = instantiate_policy(state, config, device)
    return model, config


def evaluate(
    checkpoint_path: Path,
    games: int,
    seed: int,
    device_name: str,
    baseline: str,
    profile: str | None,
    search_nodes: int,
    search_beam_width: int,
    search_branch_width: int,
    search_maximum_actions_per_turn: int,
    width: int | None,
    height: int | None,
    action_limit: int | None,
    procedural: bool = False,
    players: int = 2,
    land_density_per_million: int = 650_000,
    starting_province_size: int = 5,
    starting_money: int = 10,
    tree_density_per_million: int = 150_000,
    neutral_tower_density_per_million: int = 20_000,
    neutral_capital_density_per_million: int = 10_000,
    grave_density_per_million: int = 15_000,
    model_seat: int | None = None,
    model_agent: str = "policy",
    puct_nodes: int = 256,
    puct_exploration: float = 1.5,
    puct_virtual_loss: float = 1.0,
    puct_maximum_depth: int = 128,
    puct_root_value_weight: float | None = None,
    puct_leaf_batch_size: int = 512,
    baseline_checkpoint_path: Path | None = None,
    puct_value_perspective: ValuePerspective = "active",
    puct_opponent_horizon: OpponentHorizon = "search",
    puct_objective: SearchObjective = "scalar",
    maxn_value_head_path: Path | None = None,
) -> dict[str, object]:
    if model_agent not in ("policy", "puct"):
        raise ValueError(f"unsupported model agent: {model_agent}")
    if baseline_checkpoint_path is not None and baseline != "policy":
        raise ValueError("a baseline checkpoint requires the policy baseline")
    if maxn_value_head_path is not None and (
        model_agent != "puct" or puct_objective != "maxn"
    ):
        raise ValueError("a MaxN value head requires the MaxN PUCT agent")
    evaluation_seeds, model_seats = evaluation_schedule(
        games, seed, players, model_seat
    )
    device = torch.device(device_name)
    generator_name = "procedural_v1" if procedural else "symmetric_duel_v1"
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    base_config = dict(checkpoint["config"])
    evaluation_profile = profile or base_config["profile"] or base_config["profiles"][0]
    evaluation_width = base_config["width"] if width is None else width
    evaluation_height = base_config["height"] if height is None else height
    evaluation_action_limit = (
        base_config["action_limit"] if action_limit is None else action_limit
    )
    domain_descriptor: dict[str, object] = {
        "width": evaluation_width,
        "height": evaluation_height,
        "players": players,
        "action_limit": evaluation_action_limit,
        "fog": base_config["fog"],
        "diplomacy": base_config.get("diplomacy", False),
        "initial_relation": base_config.get("initial_relation", "neutral"),
    }
    if procedural:
        domain_descriptor.update(
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
    evaluation_domain = domain_key(generator_name, domain_descriptor)
    models: dict[str, UniversalPolicy] = {}
    selected_experts: list[str] = []
    configs: list[dict[str, object]] = []
    for seat in range(players):
        state, seat_config = select_policy_state(
            checkpoint,
            profile,
            generator_name,
            players,
            seat,
            evaluation_domain,
        )
        selected_expert = str(seat_config["selected_expert"])
        if selected_expert not in models:
            models[selected_expert] = instantiate_policy(state, seat_config, device)
        selected_experts.append(selected_expert)
        configs.append(seat_config)
    config = configs[0]
    environment_arguments = {
        "action_limit": evaluation_action_limit,
        "profile": evaluation_profile,
        "fog": config["fog"],
        "diplomacy": config.get("diplomacy", False),
        "initial_relation": config.get("initial_relation", "neutral"),
    }

    def create_environment(environments: int) -> VectorEnv:
        if procedural:
            generator = ProceduralConfig(
                width=evaluation_width,
                height=evaluation_height,
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
            return VectorEnv.procedural(
                environments, generator, **environment_arguments
            )
        if players != 2:
            raise ValueError("symmetric duel evaluation requires exactly two players")
        return VectorEnv(
            environments,
            width=evaluation_width,
            height=evaluation_height,
            seed=seed,
            **environment_arguments,
        )

    environment = create_environment(games)
    for index, evaluation_seed in enumerate(evaluation_seeds):
        environment.reset(index, int(evaluation_seed))
    rules = encode_rules_batch(environment.rules_jsons(), device)

    routed_policy = RoutedPolicy(models, selected_experts)
    maxn_value_head = None
    if maxn_value_head_path is not None:
        artifact = torch.load(
            maxn_value_head_path, map_location=device, weights_only=False
        )
        source = artifact.get("source")
        if not isinstance(source, dict) or source.get("sha256") != digest(
            checkpoint_path
        ):
            raise ValueError("vector-value head was trained for another checkpoint")
        maxn_value_head = load_relative_value_head(
            artifact, int(config["hidden"]), device
        )
    fast_maxn_evaluator = (
        None
        if maxn_value_head is None
        else lambda selected_observation, selected_rules: routed_policy.maxn(
            selected_observation, selected_rules, maxn_value_head
        )
    )
    baseline_policy = routed_policy
    baseline_selected_experts = selected_experts
    if baseline_checkpoint_path is not None:
        baseline_checkpoint = load_policy_checkpoint(baseline_checkpoint_path, device)
        baseline_base_config = dict(baseline_checkpoint["config"])
        environment_defaults = {
            "fog": False,
            "diplomacy": False,
            "initial_relation": "neutral",
        }
        for field, default in environment_defaults.items():
            if baseline_base_config.get(field, default) != base_config.get(
                field, default
            ):
                raise ValueError(
                    f"baseline checkpoint environment field does not match: {field}"
                )
        baseline_models: dict[str, UniversalPolicy] = {}
        baseline_selected_experts = []
        for seat in range(players):
            baseline_state, baseline_config = select_policy_state(
                baseline_checkpoint,
                evaluation_profile,
                generator_name,
                players,
                seat,
                evaluation_domain,
            )
            baseline_expert = str(baseline_config["selected_expert"])
            if baseline_expert not in baseline_models:
                baseline_models[baseline_expert] = instantiate_policy(
                    baseline_state, baseline_config, device
                )
            baseline_selected_experts.append(baseline_expert)
        baseline_policy = RoutedPolicy(baseline_models, baseline_selected_experts)

    def evaluate_baseline_reference() -> BaselineSelfPlay:
        reference_games = games // players if model_seat is None else games
        reference_environment = create_environment(reference_games)
        for index in range(reference_games):
            reference_environment.reset(index, seed + index)
        reference_finished = np.zeros(reference_games, dtype=np.bool_)
        reference_wins = np.zeros(players, dtype=np.int64)
        reference_winners = np.full(reference_games, 255, dtype=np.uint8)
        reference_draws = 0
        reference_terminal_draws = 0
        reference_truncations = 0
        reference_reset_seed = seed + reference_games
        reference_random = np.random.default_rng(seed ^ 0xA11CE)
        reference_rules = encode_rules_batch(
            reference_environment.rules_jsons(), device
        )
        while not bool(reference_finished.all()):
            reference_observation = reference_environment.observe()
            reference_actions = (
                baseline_policy.actions(reference_observation, reference_rules)
                if baseline == "policy"
                else choose_baseline_actions(
                    reference_environment,
                    reference_observation,
                    baseline,
                    reference_finished,
                    reference_random,
                    search_nodes,
                    search_beam_width,
                    search_branch_width,
                    search_maximum_actions_per_turn,
                )
            )
            reference_result = reference_environment.step(reference_actions)
            reference_done = np.logical_or(
                reference_result["terminal"], reference_result["truncated"]
            )
            for index in np.flatnonzero(reference_done):
                if not reference_finished[index]:
                    winner = int(reference_result["winners"][index])
                    truncated = bool(reference_result["truncated"][index])
                    if truncated:
                        reference_truncations += 1
                        winner = int(reference_result["adjudicated_winners"][index])
                    if winner == 255:
                        reference_draws += 1
                        if not truncated:
                            reference_terminal_draws += 1
                    else:
                        reference_wins[winner] += 1
                    reference_winners[index] = winner
                    reference_finished[index] = True
                reference_environment.reset(int(index), reference_reset_seed)
                reference_reset_seed += 1
        return {
            "games": reference_games,
            "wins_by_seat": reference_wins.tolist(),
            "draws": reference_draws,
            "terminal_draws": reference_terminal_draws,
            "truncations": reference_truncations,
            "winners": reference_winners.tolist(),
        }

    baseline_reference = evaluate_baseline_reference()
    finished = np.zeros(games, dtype=np.bool_)
    seat_wins = np.zeros(players, dtype=np.int64)
    seat_draws = np.zeros(players, dtype=np.int64)
    seat_terminal_draws = np.zeros(players, dtype=np.int64)
    seat_truncations = np.zeros(players, dtype=np.int64)
    seat_adjudications = np.zeros(players, dtype=np.int64)
    seat_losses = np.zeros(players, dtype=np.int64)
    game_scores = np.zeros(games, dtype=np.float64)
    game_winners = np.full(games, 255, dtype=np.uint8)
    reset_seed = seed + games // players
    random = np.random.default_rng(seed)
    transitions = 0
    model_action_counts = np.zeros(len(ACTION_KIND_NAMES), dtype=np.int64)
    baseline_action_counts = np.zeros(len(ACTION_KIND_NAMES), dtype=np.int64)
    puct_decisions = 0
    puct_evaluated_leaves = 0
    puct_leaf_batches = 0
    puct_total_nodes = 0
    puct_total_root_visits = 0
    puct_maximum_reached_depth = 0
    puct_config = PolicySearchConfig(
        node_budget=puct_nodes,
        exploration=puct_exploration,
        virtual_loss=puct_virtual_loss,
        maximum_depth=puct_maximum_depth,
        root_value_weight=puct_root_value_weight,
        leaf_batch_size=puct_leaf_batch_size,
        value_perspective=puct_value_perspective,
        opponent_horizon=puct_opponent_horizon,
        objective=puct_objective,
    )
    while not bool(finished.all()):
        observation = environment.observe()
        active_players = observation["active_players"]
        active = np.logical_not(finished)
        model_turns = np.logical_and(active, active_players == model_seats)
        if model_agent == "puct":
            model_actions, puct_metrics = policy_search_actions(
                environment,
                routed_policy,
                rules,
                model_turns,
                puct_config,
                maxn_evaluator=fast_maxn_evaluator,
            )
            puct_decisions += int(model_turns.sum())
            puct_evaluated_leaves += puct_metrics["evaluated_leaves"]
            puct_leaf_batches += puct_metrics["leaf_batches"]
            puct_total_nodes += int(puct_metrics["nodes"].sum())
            puct_total_root_visits += int(puct_metrics["root_visits"].sum())
            puct_maximum_reached_depth = max(
                puct_maximum_reached_depth,
                int(puct_metrics["maximum_depth"].max(initial=0)),
            )
        else:
            model_actions = routed_policy.actions(observation, rules)
        if baseline == "policy":
            baseline_actions = (
                model_actions
                if model_agent == "policy" and baseline_checkpoint_path is None
                else baseline_policy.actions(observation, rules)
            )
        else:
            baseline_actions = choose_baseline_actions(
                environment,
                observation,
                baseline,
                finished,
                random,
                search_nodes,
                search_beam_width,
                search_branch_width,
                search_maximum_actions_per_turn,
            )
        actions = np.where(
            active_players == model_seats, model_actions, baseline_actions
        )
        action_kinds = selected_action_kinds(observation, actions)
        baseline_turns = np.logical_and(active, active_players != model_seats)
        model_action_counts += np.bincount(
            action_kinds[model_turns], minlength=len(ACTION_KIND_NAMES)
        )
        baseline_action_counts += np.bincount(
            action_kinds[baseline_turns], minlength=len(ACTION_KIND_NAMES)
        )
        result = environment.step(actions)
        transitions += games
        done = np.logical_or(result["terminal"], result["truncated"])
        for index in np.flatnonzero(done):
            if not finished[index]:
                game_model_seat = int(model_seats[index])
                winner = int(result["winners"][index])
                truncated = bool(result["truncated"][index])
                if truncated:
                    seat_truncations[game_model_seat] += 1
                    seat_adjudications[game_model_seat] += 1
                    winner = int(result["adjudicated_winners"][index])
                if winner == 255:
                    seat_draws[game_model_seat] += 1
                    if not truncated:
                        seat_terminal_draws[game_model_seat] += 1
                elif winner == game_model_seat:
                    seat_wins[game_model_seat] += 1
                else:
                    seat_losses[game_model_seat] += 1
                game_scores[index] = winner_score(winner, game_model_seat)
                game_winners[index] = winner
                finished[index] = True
            environment.reset(int(index), reset_seed)
            reset_seed += 1
    reference_wins = (
        sum(int(wins) for wins in baseline_reference["wins_by_seat"])
        if model_seat is None
        else int(baseline_reference["wins_by_seat"][model_seat])
    )
    reference_replication = players if model_seat is None else 1
    summary = reference_adjusted_outcome(
        outcome_summary(
            games,
            int(seat_wins.sum()),
            int(seat_draws.sum()),
            int(seat_losses.sum()),
            int(seat_terminal_draws.sum()),
            int(seat_truncations.sum()),
            players,
            int(seat_adjudications.sum()),
        ),
        games,
        reference_wins,
        int(baseline_reference["draws"]) * reference_replication,
        int(baseline_reference["truncations"]) * reference_replication,
    )
    baseline_scores = np.asarray(
        [
            winner_score(
                int(
                    baseline_reference["winners"][
                        index if model_seat is not None else index // players
                    ]
                ),
                int(model_seats[index]),
            )
            for index in range(games)
        ],
        dtype=np.float64,
    )
    reported_seats = range(players) if model_seat is None else (model_seat,)
    games_per_reported_seat = games // players if model_seat is None else games
    seats = []
    for seat in reported_seats:
        seat_mask = model_seats == seat
        seats.append(
            {
                "seat": seat,
                **reference_adjusted_outcome(
                    outcome_summary(
                        games_per_reported_seat,
                        int(seat_wins[seat]),
                        int(seat_draws[seat]),
                        int(seat_losses[seat]),
                        int(seat_terminal_draws[seat]),
                        int(seat_truncations[seat]),
                        players,
                        int(seat_adjudications[seat]),
                    ),
                    games_per_reported_seat,
                    int(baseline_reference["wins_by_seat"][seat]),
                    int(baseline_reference["draws"]),
                    int(baseline_reference["truncations"]),
                ),
                "paired_method_comparison": paired_method_comparison(
                    game_scores[seat_mask], baseline_scores[seat_mask]
                ),
            }
        )
    return {
        "checkpoint": str(checkpoint_path),
        "baseline": baseline,
        "baseline_checkpoint": (
            str(baseline_checkpoint_path)
            if baseline_checkpoint_path is not None
            else None
        ),
        **summary,
        "pairing": {
            "scheme": (
                FIXED_SEAT_SCHEME
                if model_seat is not None
                else PAIRING_SCHEME
                if players == 2
                else SEAT_ROTATION_SCHEME
            ),
            "unique_seeds": games if model_seat is not None else games // players,
            "first_seed": seed,
            "last_seed": seed
            + (games if model_seat is not None else games // players)
            - 1,
        },
        "seats": seats,
        "transitions": transitions,
        "device": str(device),
        "profile": evaluation_profile,
        "policy_kind": config["policy_kind"],
        "selected_expert": (
            selected_experts[model_seat]
            if model_seat is not None
            else selected_experts[0]
            if len(set(selected_experts)) == 1
            else "seat_routed"
        ),
        "selected_experts": selected_experts,
        "baseline_selected_experts": (
            baseline_selected_experts if baseline == "policy" else []
        ),
        "model_seat": model_seat,
        "game_seeds": evaluation_seeds.tolist(),
        "model_seats": model_seats.tolist(),
        "winners": game_winners.tolist(),
        "baseline_self_play": baseline_reference,
        "paired_method_comparison": paired_method_comparison(
            game_scores, baseline_scores
        ),
        "seed": seed,
        "generator": generator_name,
        "domain": evaluation_domain,
        "domain_descriptor": domain_descriptor,
        "players": players,
        "arena_width": evaluation_width,
        "arena_height": evaluation_height,
        "action_limit": evaluation_action_limit,
        "generator_config": (
            {
                "land_density_per_million": land_density_per_million,
                "starting_province_size": starting_province_size,
                "starting_money": starting_money,
                "tree_density_per_million": tree_density_per_million,
                "neutral_tower_density_per_million": neutral_tower_density_per_million,
                "neutral_capital_density_per_million": neutral_capital_density_per_million,
                "grave_density_per_million": grave_density_per_million,
            }
            if procedural
            else None
        ),
        "model_action_counts": named_action_counts(model_action_counts),
        "baseline_action_counts": named_action_counts(baseline_action_counts),
        "model_agent": model_agent,
        "policy_search": {
            "node_budget": puct_nodes if model_agent == "puct" else 0,
            "exploration": puct_exploration if model_agent == "puct" else 0.0,
            "virtual_loss": puct_virtual_loss if model_agent == "puct" else 0.0,
            "maximum_depth": puct_maximum_depth if model_agent == "puct" else 0,
            "root_value_weight": (
                puct_root_value_weight if model_agent == "puct" else None
            ),
            "leaf_batch_size": puct_leaf_batch_size if model_agent == "puct" else 0,
            "value_perspective": (
                puct_value_perspective if model_agent == "puct" else None
            ),
            "opponent_horizon": (
                puct_opponent_horizon if model_agent == "puct" else None
            ),
            "objective": puct_objective if model_agent == "puct" else None,
            "vector_value_head": (
                str(maxn_value_head_path) if maxn_value_head_path is not None else None
            ),
            "decisions": puct_decisions,
            "evaluated_leaves": puct_evaluated_leaves,
            "leaf_batches": puct_leaf_batches,
            "total_nodes": puct_total_nodes,
            "total_root_visits": puct_total_root_visits,
            "maximum_reached_depth": puct_maximum_reached_depth,
        },
        "search_nodes": search_nodes if baseline == "search" else 0,
        "search_beam_width": search_beam_width if baseline == "search" else 0,
        "search_branch_width": search_branch_width if baseline == "search" else 0,
        "search_maximum_actions_per_turn": (
            search_maximum_actions_per_turn if baseline == "search" else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--baseline",
        choices=("policy", "search", "greedy", "random"),
        default="greedy",
    )
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--search-nodes", type=int, default=2048)
    parser.add_argument("--search-beam-width", type=int, default=32)
    parser.add_argument("--search-branch-width", type=int, default=48)
    parser.add_argument("--search-maximum-actions-per-turn", type=int, default=24)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--action-limit", type=int)
    parser.add_argument("--procedural", action="store_true")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--model-seat", type=int)
    parser.add_argument("--model-agent", choices=("policy", "puct"), default="policy")
    parser.add_argument("--puct-nodes", type=int, default=256)
    parser.add_argument("--puct-exploration", type=float, default=1.5)
    parser.add_argument("--puct-virtual-loss", type=float, default=1.0)
    parser.add_argument("--puct-maximum-depth", type=int, default=128)
    parser.add_argument("--puct-root-value-weight", type=float)
    parser.add_argument("--puct-leaf-batch-size", type=int, default=512)
    parser.add_argument(
        "--puct-value-perspective", choices=("active", "root"), default="active"
    )
    parser.add_argument(
        "--puct-opponent-horizon", choices=("search", "leaf"), default="search"
    )
    parser.add_argument(
        "--puct-objective", choices=("scalar", "maxn"), default="scalar"
    )
    parser.add_argument("--maxn-value-head", type=Path)
    parser.add_argument("--land-density-per-million", type=int, default=650_000)
    parser.add_argument("--starting-province-size", type=int, default=5)
    parser.add_argument("--starting-money", type=int, default=10)
    parser.add_argument("--tree-density-per-million", type=int, default=150_000)
    parser.add_argument("--neutral-tower-density-per-million", type=int, default=20_000)
    parser.add_argument(
        "--neutral-capital-density-per-million", type=int, default=10_000
    )
    parser.add_argument("--grave-density-per-million", type=int, default=15_000)
    arguments = parser.parse_args()
    if arguments.players < 2:
        parser.error("players must be at least two")
    if arguments.model_seat is None and (
        arguments.games < arguments.players or arguments.games % arguments.players != 0
    ):
        parser.error("games must be a positive multiple of players")
    if arguments.model_seat is not None and arguments.games < 1:
        parser.error("fixed-seat evaluation requires at least one game")
    if arguments.model_seat is not None and not (
        0 <= arguments.model_seat < arguments.players
    ):
        parser.error("model seat must belong to the player range")
    if not arguments.procedural and arguments.players != 2:
        parser.error("symmetric duel evaluation requires exactly two players")
    if arguments.baseline_checkpoint is not None and arguments.baseline != "policy":
        parser.error("--baseline-checkpoint requires --baseline policy")
    print(
        json.dumps(
            evaluate(
                arguments.checkpoint,
                arguments.games,
                arguments.seed,
                arguments.device,
                arguments.baseline,
                arguments.profile,
                arguments.search_nodes,
                arguments.search_beam_width,
                arguments.search_branch_width,
                arguments.search_maximum_actions_per_turn,
                arguments.width,
                arguments.height,
                arguments.action_limit,
                arguments.procedural,
                arguments.players,
                arguments.land_density_per_million,
                arguments.starting_province_size,
                arguments.starting_money,
                arguments.tree_density_per_million,
                arguments.neutral_tower_density_per_million,
                arguments.neutral_capital_density_per_million,
                arguments.grave_density_per_million,
                arguments.model_seat,
                arguments.model_agent,
                arguments.puct_nodes,
                arguments.puct_exploration,
                arguments.puct_virtual_loss,
                arguments.puct_maximum_depth,
                arguments.puct_root_value_weight,
                arguments.puct_leaf_batch_size,
                arguments.baseline_checkpoint,
                arguments.puct_value_perspective,
                arguments.puct_opponent_horizon,
                arguments.puct_objective,
                arguments.maxn_value_head,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
