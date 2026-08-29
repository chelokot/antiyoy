from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import UniversalPolicy, action_distribution, encode_rules


def evaluate(
    checkpoint_path: Path,
    games: int,
    seed: int,
    device_name: str,
    baseline: str,
) -> dict[str, float | int | str]:
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = UniversalPolicy(config["hidden"], config["layers"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    environment = VectorEnv(
        games,
        width=config["width"],
        height=config["height"],
        seed=seed,
        action_limit=config["action_limit"],
    )
    rules = encode_rules(environment.rules_json(), device)
    model_seats = np.arange(games, dtype=np.uint8) % 2
    finished = np.zeros(games, dtype=np.bool_)
    wins = 0
    draws = 0
    losses = 0
    reset_seed = seed + games
    random = np.random.default_rng(seed)
    transitions = 0
    while not bool(finished.all()):
        observation = environment.observe()
        with torch.no_grad():
            logits, _ = model(observation, rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            model_actions = distribution.logits.argmax(dim=1).cpu().numpy().astype(np.uint64)
        if baseline == "greedy":
            baseline_actions = np.asarray(environment.greedy_actions(), dtype=np.uint64)
        else:
            counts = np.diff(observation["action_offsets"])
            baseline_actions = np.array(
                [random.integers(0, count) for count in counts], dtype=np.uint64
            )
        active_players = observation["active_players"]
        actions = np.where(active_players == model_seats, model_actions, baseline_actions)
        result = environment.step(actions)
        transitions += games
        done = np.logical_or(result["terminal"], result["truncated"])
        for index in np.flatnonzero(done):
            if not finished[index]:
                winner = int(result["winners"][index])
                if winner == 255:
                    draws += 1
                elif winner == int(model_seats[index]):
                    wins += 1
                else:
                    losses += 1
                finished[index] = True
            environment.reset(int(index), reset_seed)
            reset_seed += 1
    score = (wins + 0.5 * draws) / games
    clipped_score = min(max(score, 0.5 / games), 1 - 0.5 / games)
    elo_delta = 400 * math.log10(clipped_score / (1 - clipped_score))
    return {
        "checkpoint": str(checkpoint_path),
        "baseline": baseline,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_delta": elo_delta,
        "transitions": transitions,
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline", choices=("greedy", "random"), default="greedy")
    arguments = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                arguments.checkpoint,
                arguments.games,
                arguments.seed,
                arguments.device,
                arguments.baseline,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
