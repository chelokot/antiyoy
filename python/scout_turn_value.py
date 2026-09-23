from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from antiyoy_rl.model import UniversalPolicy, encode_rules_batch
from antiyoy_rl.turn_credit import TurnCreditPosition, load_turn_credit_positions

from .build_bundle import digest
from .evaluate import load_policy, paired_comparison_summary


@dataclass(frozen=True)
class EmbeddedPosition:
    seed: int
    seat: int
    features: Tensor
    outcomes: np.ndarray
    search_index: int


def embed_position(position: TurnCreditPosition, model: UniversalPolicy) -> EmbeddedPosition:
    if not np.all(position.post_turn["player_counts"] == 5) or any(
        json.loads(rule)["profile"] != "ClassicGeneric"
        for rule in position.post_turn_rules_json
    ):
        raise ValueError("the scout requires five-player Classic Generic observations")
    rules = encode_rules_batch(list(position.post_turn_rules_json), torch.device("cpu"))
    with torch.inference_mode():
        _, _, features = model.forward_with_value_features(position.post_turn, rules)
    static_score = torch.as_tensor(position.static_scores, dtype=torch.float32)
    features = torch.cat((features.cpu(), static_score[:, None] / 1000), dim=1)
    return EmbeddedPosition(
        seed=position.seed,
        seat=position.seat,
        features=features,
        outcomes=position.outcome_scores,
        search_index=position.search_index,
    )


def load_embeddings(paths: list[Path], model: UniversalPolicy) -> list[EmbeddedPosition]:
    positions = []
    for path in paths:
        positions.extend(
            embed_position(position, model)
            for position in load_turn_credit_positions(path)
        )
    return positions


def pairwise_examples(positions: list[EmbeddedPosition]) -> tuple[Tensor, Tensor]:
    position_pairs = []
    for position in positions:
        differences = []
        complete = np.flatnonzero(position.outcomes >= 0)
        for left in complete:
            for right in complete:
                if position.outcomes[left] > position.outcomes[right]:
                    differences.append(position.features[left] - position.features[right])
        if differences:
            position_pairs.append((position.seed, torch.stack(differences)))
    if not position_pairs:
        raise ValueError("training data has no outcome-diverse completed slates")
    informative_maps = Counter(seed for seed, _ in position_pairs)
    examples = torch.cat([pairs for _, pairs in position_pairs])
    weights = torch.cat(
        [
            torch.full((len(pairs),), 1 / (informative_maps[seed] * len(pairs)))
            for seed, pairs in position_pairs
        ]
    )
    return examples, weights


def train_head(
    differences: Tensor, example_weights: Tensor, steps: int = 200
) -> tuple[Tensor, Tensor]:
    scales = differences.std(dim=0, unbiased=False).clamp_min(0.01)
    normalized = differences / scales
    weight = torch.nn.Parameter(torch.zeros(normalized.shape[1]))
    optimizer = torch.optim.Adam([weight], lr=0.03)
    for _ in range(steps):
        logits = normalized @ weight
        pair_loss = (functional.softplus(-logits) * example_weights).sum()
        loss = pair_loss / example_weights.sum() + 0.1 * weight.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return weight.detach(), scales


def evaluate_head(
    positions: list[EmbeddedPosition],
    weight: Tensor,
    scales: Tensor,
    minimum_margin: float = 0.0,
) -> dict[str, object]:
    better = 0
    worse = 0
    same = 0
    censored = 0
    changed = 0
    by_seat: dict[int, dict[str, int]] = {}
    map_deltas: dict[int, int] = {}
    censored_maps = set()
    for position in positions:
        values = (position.features / scales) @ weight
        selected = int(values.argmax())
        if float(values[selected] - values[position.search_index]) <= minimum_margin:
            selected = position.search_index
        changed += int(selected != position.search_index)
        baseline = int(position.outcomes[position.search_index])
        candidate = int(position.outcomes[selected])
        seat_counts = by_seat.setdefault(
            position.seat, {"better": 0, "worse": 0, "same": 0, "censored": 0}
        )
        if baseline < 0 or candidate < 0:
            censored += 1
            seat_counts["censored"] += 1
            censored_maps.add(position.seed)
            continue
        difference = candidate - baseline
        map_deltas[position.seed] = map_deltas.get(position.seed, 0) + difference
        if difference > 0:
            better += 1
            seat_counts["better"] += 1
        elif difference < 0:
            worse += 1
            seat_counts["worse"] += 1
        else:
            same += 1
            seat_counts["same"] += 1
    independent_maps = paired_comparison_summary(
        sum(value > 0 for seed, value in map_deltas.items() if seed not in censored_maps),
        sum(value < 0 for seed, value in map_deltas.items() if seed not in censored_maps),
        sum(value == 0 for seed, value in map_deltas.items() if seed not in censored_maps),
    )
    independent_maps["censored"] = len(censored_maps)
    return {
        "positions": len(positions),
        "changed_choices": changed,
        "better": better,
        "worse": worse,
        "same": same,
        "censored": censored,
        "by_seat": by_seat,
        "independent_maps": independent_maps,
    }


def scout(
    checkpoint: Path,
    training_paths: list[Path],
    validation_paths: list[Path],
) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.manual_seed(72031)
    model, config = load_policy(
        checkpoint,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=5,
    )
    training = load_embeddings(training_paths, model)
    validation = load_embeddings(validation_paths, model)
    training_seeds = {position.seed for position in training}
    validation_seeds = {position.seed for position in validation}
    if training_seeds & validation_seeds:
        raise ValueError("training and validation maps overlap")
    differences, example_weights = pairwise_examples(training)
    weight, scales = train_head(differences, example_weights)
    return {
        "kind": "whole_turn_outcome_value_scout",
        "source_model_sha256": digest(checkpoint),
        "source_expert": config["selected_expert"],
        "training_files": [
            {"name": path.name, "sha256": digest(path)} for path in training_paths
        ],
        "validation_files": [
            {"name": path.name, "sha256": digest(path)} for path in validation_paths
        ],
        "training_maps_with_samples": len(training_seeds),
        "validation_maps_with_samples": len(validation_seeds),
        "training_pairs": differences.shape[0],
        "training": evaluate_head(training, weight, scales),
        "validation": evaluate_head(validation, weight, scales),
        "training_conservative": evaluate_head(training, weight, scales, 1.0),
        "validation_conservative": evaluate_head(validation, weight, scales, 1.0),
        "qualification": "offline exact greedy continuation only; no policy promotion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train", required=True, action="append", type=Path)
    parser.add_argument("--validation", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = scout(arguments.checkpoint, arguments.train, arguments.validation)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
