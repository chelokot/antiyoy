from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import OBSERVATION_VERSION, VectorEnv
from antiyoy_rl.model import (
    RULE_FEATURES,
    UniversalPolicy,
    action_distribution,
    encode_rules_batch,
    load_policy_state,
)
try:
    from .build_bundle import BUNDLE_KIND, BUNDLE_VERSION
except ImportError:
    from build_bundle import BUNDLE_KIND, BUNDLE_VERSION
try:
    from .train import CHECKPOINT_VERSION
except ImportError:
    from train import CHECKPOINT_VERSION


ACTION_KIND_NAMES = ("end_turn", "move", "recruit", "build", "plant_tree", "diplomacy")
PAIRING_SCHEME = "adjacent_same_seed_opposite_seat_v1"


def paired_elo(score: float, games: int) -> float:
    clipped_score = min(max(score, 0.5 / games), 1 - 0.5 / games)
    return 400 * math.log10(clipped_score / (1 - clipped_score))


def paired_seeds(games: int, seed: int) -> np.ndarray:
    if games < 2 or games % 2 != 0:
        raise ValueError("paired evaluation requires a positive even number of games")
    return seed + np.arange(games, dtype=np.uint64) // 2


def outcome_summary(
    games: int,
    wins: int,
    draws: int,
    losses: int,
    terminal_draws: int,
    truncations: int,
) -> dict[str, float | int]:
    score = (wins + 0.5 * draws) / games
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "terminal_draws": terminal_draws,
        "truncations": truncations,
        "losses": losses,
        "score": score,
        "elo_delta": paired_elo(score, games),
    }


def selected_action_kinds(
    observation: dict[str, np.ndarray], actions: np.ndarray
) -> np.ndarray:
    offsets = np.asarray(observation["action_offsets"][:-1], dtype=np.int64)
    indices = offsets + actions.astype(np.int64, copy=False)
    return np.asarray(observation["action_kinds"])[indices]


def named_action_counts(counts: np.ndarray) -> dict[str, int]:
    return {
        name: int(counts[index])
        for index, name in enumerate(ACTION_KIND_NAMES)
    }


def load_policy(
    checkpoint_path: Path, device: torch.device, profile: str | None = None
) -> tuple[UniversalPolicy, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["checkpoint_version"] not in (4, CHECKPOINT_VERSION):
        raise ValueError("checkpoint format version does not match this evaluator")
    if checkpoint["observation_version"] not in (6, OBSERVATION_VERSION):
        raise ValueError("checkpoint observation version does not match the native environment")
    if checkpoint["rule_features"] not in (42, RULE_FEATURES):
        raise ValueError("checkpoint rule feature width does not match the policy")
    config = dict(checkpoint["config"])
    state = checkpoint.get("model")
    selected_expert = "single"
    if checkpoint.get("kind") == BUNDLE_KIND:
        if checkpoint.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError("policy bundle version does not match this evaluator")
        selected_profile = profile or config["profile"] or config["profiles"][0]
        selected_expert = checkpoint["routes"].get(selected_profile)
        if selected_expert is None:
            raise ValueError(f"policy bundle has no route for profile: {selected_profile}")
        state = checkpoint["experts"][selected_expert]
    if state is None:
        raise ValueError("checkpoint has no policy weights")
    model = UniversalPolicy(config["hidden"], config["layers"]).to(device)
    load_policy_state(model, state)
    model.eval()
    config["policy_kind"] = checkpoint.get("kind", "single_policy")
    config["selected_expert"] = selected_expert
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
) -> dict[str, object]:
    evaluation_seeds = paired_seeds(games, seed)
    device = torch.device(device_name)
    model, config = load_policy(checkpoint_path, device, profile)
    evaluation_profile = profile or config["profile"] or config["profiles"][0]
    evaluation_width = config["width"] if width is None else width
    evaluation_height = config["height"] if height is None else height
    evaluation_action_limit = (
        config["action_limit"] if action_limit is None else action_limit
    )
    environment = VectorEnv(
        games,
        width=evaluation_width,
        height=evaluation_height,
        seed=seed,
        action_limit=evaluation_action_limit,
        profile=evaluation_profile,
        fog=config["fog"],
        diplomacy=config.get("diplomacy", False),
        initial_relation=config.get("initial_relation", "neutral"),
    )
    for index, evaluation_seed in enumerate(evaluation_seeds):
        environment.reset(index, int(evaluation_seed))
    rules = encode_rules_batch(environment.rules_jsons(), device)
    model_seats = np.arange(games, dtype=np.uint8) % 2
    finished = np.zeros(games, dtype=np.bool_)
    seat_wins = np.zeros(2, dtype=np.int64)
    seat_draws = np.zeros(2, dtype=np.int64)
    seat_terminal_draws = np.zeros(2, dtype=np.int64)
    seat_truncations = np.zeros(2, dtype=np.int64)
    seat_losses = np.zeros(2, dtype=np.int64)
    reset_seed = seed + games // 2
    random = np.random.default_rng(seed)
    transitions = 0
    model_action_counts = np.zeros(len(ACTION_KIND_NAMES), dtype=np.int64)
    baseline_action_counts = np.zeros(len(ACTION_KIND_NAMES), dtype=np.int64)
    while not bool(finished.all()):
        observation = environment.observe()
        with torch.no_grad():
            logits, _ = model(observation, rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            model_actions = distribution.logits.argmax(dim=1).cpu().numpy().astype(np.uint64)
        if baseline == "search":
            baseline_actions = np.asarray(
                environment.search_actions(
                    node_budget=search_nodes,
                    beam_width=search_beam_width,
                    branch_width=search_branch_width,
                    maximum_actions_per_turn=search_maximum_actions_per_turn,
                    active_mask=np.logical_not(finished).astype(np.uint8),
                ),
                dtype=np.uint64,
            )
        elif baseline == "greedy":
            baseline_actions = np.asarray(environment.greedy_actions(), dtype=np.uint64)
        else:
            counts = np.diff(observation["action_offsets"])
            baseline_actions = np.array(
                [random.integers(0, count) for count in counts], dtype=np.uint64
            )
        active_players = observation["active_players"]
        actions = np.where(active_players == model_seats, model_actions, baseline_actions)
        action_kinds = selected_action_kinds(observation, actions)
        active = np.logical_not(finished)
        model_turns = np.logical_and(active, active_players == model_seats)
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
                model_seat = int(model_seats[index])
                winner = int(result["winners"][index])
                truncated = bool(result["truncated"][index])
                if truncated:
                    seat_truncations[model_seat] += 1
                if winner == 255:
                    seat_draws[model_seat] += 1
                    if not truncated:
                        seat_terminal_draws[model_seat] += 1
                elif winner == model_seat:
                    seat_wins[model_seat] += 1
                else:
                    seat_losses[model_seat] += 1
                finished[index] = True
            environment.reset(int(index), reset_seed)
            reset_seed += 1
    summary = outcome_summary(
        games,
        int(seat_wins.sum()),
        int(seat_draws.sum()),
        int(seat_losses.sum()),
        int(seat_terminal_draws.sum()),
        int(seat_truncations.sum()),
    )
    seats = [
        {
            "seat": seat,
            **outcome_summary(
                games // 2,
                int(seat_wins[seat]),
                int(seat_draws[seat]),
                int(seat_losses[seat]),
                int(seat_terminal_draws[seat]),
                int(seat_truncations[seat]),
            ),
        }
        for seat in range(2)
    ]
    return {
        "checkpoint": str(checkpoint_path),
        "baseline": baseline,
        **summary,
        "pairing": {
            "scheme": PAIRING_SCHEME,
            "unique_seeds": games // 2,
            "first_seed": seed,
            "last_seed": seed + games // 2 - 1,
        },
        "seats": seats,
        "transitions": transitions,
        "device": str(device),
        "profile": evaluation_profile,
        "policy_kind": config["policy_kind"],
        "selected_expert": config["selected_expert"],
        "seed": seed,
        "arena_width": evaluation_width,
        "arena_height": evaluation_height,
        "action_limit": evaluation_action_limit,
        "model_action_counts": named_action_counts(model_action_counts),
        "baseline_action_counts": named_action_counts(baseline_action_counts),
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--baseline", choices=("search", "greedy", "random"), default="greedy"
    )
    parser.add_argument("--profile")
    parser.add_argument("--search-nodes", type=int, default=2048)
    parser.add_argument("--search-beam-width", type=int, default=32)
    parser.add_argument("--search-branch-width", type=int, default=48)
    parser.add_argument("--search-maximum-actions-per-turn", type=int, default=24)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--action-limit", type=int)
    arguments = parser.parse_args()
    if arguments.games < 2 or arguments.games % 2 != 0:
        parser.error("games must be a positive even number for paired evaluation")
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
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
