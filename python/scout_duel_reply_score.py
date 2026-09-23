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

from antiyoy_rl.slate_dataset import load_teacher_slates

from .build_bundle import digest
from .evaluate import load_policy, paired_comparison_summary
from .scout_duel_nonlinear_choice import TurnScorer, choices
from .scout_duel_teacher_choice import agreement_for_choices, embed_spatial
from .scout_duel_turn_value import DuelEmbedding


@dataclass(frozen=True)
class ScoredTurn:
    embedding: DuelEmbedding
    reply_scores: np.ndarray
    target: Tensor
    terminal_magnitude_scores: int
    opponent_actions: tuple[tuple[str, ...], ...] | None = None
    static_scores: np.ndarray | None = None


def load_scored_turns(paths: list[Path], encoder: torch.nn.Module) -> list[ScoredTurn]:
    turns = []
    for path in paths:
        for source in load_teacher_slates(path):
            scores = source.opponent_reply_scores
            turns.append(
                ScoredTurn(
                    embedding=embed_spatial(source, encoder, "next_active"),
                    reply_scores=scores,
                    target=torch.as_tensor(
                        np.arcsinh(scores / 1000), dtype=torch.float32
                    ),
                    terminal_magnitude_scores=int((np.abs(scores) >= 1e9).sum()),
                    opponent_actions=source.opponent_actions,
                    static_scores=source.static_scores,
                )
            )
    return turns


def training_pairs(
    turns: list[ScoredTurn],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    positions_by_map = Counter(turn.embedding.position.seed for turn in turns)
    baselines = []
    alternatives = []
    targets = []
    weights = []
    for turn in turns:
        features = turn.embedding.position.features
        competing = len(features) - 1
        if competing == 0:
            continue
        position_weight = 1 / (
            len(positions_by_map)
            * positions_by_map[turn.embedding.position.seed]
            * competing
        )
        for index in range(1, len(features)):
            baselines.append(features[0])
            alternatives.append(features[index])
            targets.append(turn.target[index] - turn.target[0])
            weights.append(position_weight)
    if not baselines:
        raise ValueError("reply-score training requires competing turn candidates")
    return (
        torch.stack(baselines),
        torch.stack(alternatives),
        torch.stack(targets),
        torch.as_tensor(weights, dtype=torch.float32),
    )


def train_scorer(
    turns: list[ScoredTurn], epochs: int = 20
) -> tuple[TurnScorer, Tensor, Tensor, list[float], int]:
    torch.manual_seed(72031)
    features = torch.cat([turn.embedding.position.features for turn in turns])
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(0.01)
    baselines, alternatives, targets, weights = training_pairs(turns)
    baselines = (baselines - mean) / scale
    alternatives = (alternatives - mean) / scale
    scorer = TurnScorer(features.shape[1])
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=0.001, weight_decay=0.01)
    generator = torch.Generator().manual_seed(72031)
    losses = []
    for _ in range(epochs):
        weighted_loss = 0.0
        total_weight = 0.0
        for indices in torch.randperm(len(weights), generator=generator).split(2048):
            batch_weights = weights[indices]
            prediction = scorer(alternatives[indices]) - scorer(baselines[indices])
            per_pair = functional.smooth_l1_loss(
                prediction, targets[indices], beta=1.0, reduction="none"
            )
            loss = (per_pair * batch_weights).sum() / batch_weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_weight = batch_weights.sum().item()
            weighted_loss += loss.detach().item() * batch_weight
            total_weight += batch_weight
        losses.append(weighted_loss / total_weight)
    return scorer, mean, scale, losses, len(weights)


def score_metrics(
    turns: list[ScoredTurn], scorer: TurnScorer, mean: Tensor, scale: Tensor
) -> dict[str, float | int]:
    error = 0.0
    correctly_ordered = 0
    ordered_pairs = 0
    actual_positive = 0
    predicted_positive = 0
    true_positive = 0
    pairs = 0
    with torch.inference_mode():
        for turn in turns:
            features = turn.embedding.position.features
            prediction = scorer((features - mean) / scale)
            predicted_differences = prediction[1:] - prediction[0]
            target_differences = turn.target[1:] - turn.target[0]
            error += (predicted_differences - target_differences).abs().sum().item()
            informative = target_differences != 0
            correctly_ordered += int(
                (
                    (predicted_differences[informative] > 0)
                    == (target_differences[informative] > 0)
                )
                .sum()
                .item()
            )
            ordered_pairs += int(informative.sum().item())
            actual_positive += int((target_differences > 0).sum().item())
            predicted_positive += int((predicted_differences > 0).sum().item())
            true_positive += int(
                ((target_differences > 0) & (predicted_differences > 0)).sum().item()
            )
            pairs += len(target_differences)
    return {
        "relative_score_pairs": pairs,
        "transformed_relative_mae": error / pairs,
        "sign_correct": correctly_ordered,
        "sign_informative_pairs": ordered_pairs,
        "actual_positive_pairs": actual_positive,
        "predicted_positive_pairs": predicted_positive,
        "true_positive_pairs": true_positive,
    }


def native_reply_comparison(
    turns: list[ScoredTurn], selected: list[int]
) -> dict[str, object]:
    better = worse = same = 0
    map_deltas: dict[int, int] = {}
    by_seat: dict[int, dict[str, int]] = {}
    for turn, index in zip(turns, selected, strict=True):
        score = turn.reply_scores[index]
        baseline = turn.reply_scores[0]
        if score > baseline:
            result = "better"
        elif score < baseline:
            result = "worse"
        else:
            result = "same"
        better += result == "better"
        worse += result == "worse"
        same += result == "same"
        seat = turn.embedding.position.seat
        seat_counts = by_seat.setdefault(seat, {"better": 0, "worse": 0, "same": 0})
        seat_counts[result] += 1
        map_seed = turn.embedding.position.seed
        map_deltas[map_seed] = (
            map_deltas.get(map_seed, 0) + int(score > baseline) - int(score < baseline)
        )
    grouped = paired_comparison_summary(
        sum(delta > 0 for delta in map_deltas.values()),
        sum(delta < 0 for delta in map_deltas.values()),
        sum(delta == 0 for delta in map_deltas.values()),
    )
    return {
        "positions": len(turns),
        "better": better,
        "worse": worse,
        "same": same,
        "by_seat": by_seat,
        "independent_maps": grouped,
    }


def scout(
    fit_paths: list[Path],
    validation_paths: list[Path],
    encoder_path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    torch.set_num_threads(1)
    encoder, config = load_policy(
        encoder_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    fit = load_scored_turns(fit_paths, encoder)
    validation = load_scored_turns(validation_paths, encoder)
    fit_maps = {turn.embedding.position.seed for turn in fit}
    validation_maps = {turn.embedding.position.seed for turn in validation}
    if fit_maps & validation_maps:
        raise ValueError("fit and validation maps overlap")
    scorer, mean, scale, losses, pair_count = train_scorer(fit)
    torch.save(
        {
            "state_dict": scorer.state_dict(),
            "mean": mean,
            "scale": scale,
            "encoder_sha256": digest(encoder_path),
            "spatial_perspective": "next_active",
            "target": "asinh(reply_score/1000)",
        },
        checkpoint_path,
    )
    fit_embeddings = [turn.embedding for turn in fit]
    validation_embeddings = [turn.embedding for turn in validation]
    fit_choices = choices(fit_embeddings, scorer, mean, scale)
    validation_choices = choices(validation_embeddings, scorer, mean, scale)
    return {
        "kind": "procedural_duel_reply_score_distillation_scout",
        "fit_files": [
            {"name": path.name, "sha256": digest(path)} for path in fit_paths
        ],
        "validation_files": [
            {"name": path.name, "sha256": digest(path)} for path in validation_paths
        ],
        "encoder_sha256": digest(encoder_path),
        "selected_expert": config["selected_expert"],
        "fit_maps": len(fit_maps),
        "validation_maps": len(validation_maps),
        "fit_positions": len(fit),
        "validation_positions": len(validation),
        "fit_terminal_magnitude_scores": sum(
            turn.terminal_magnitude_scores for turn in fit
        ),
        "validation_terminal_magnitude_scores": sum(
            turn.terminal_magnitude_scores for turn in validation
        ),
        "fit_pairs": pair_count,
        "epochs": len(losses),
        "first_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "checkpoint_sha256": digest(checkpoint_path),
        "fit_score_metrics": score_metrics(fit, scorer, mean, scale),
        "validation_score_metrics": score_metrics(validation, scorer, mean, scale),
        "fit_agreement": agreement_for_choices(fit_embeddings, fit_choices),
        "validation_agreement": agreement_for_choices(
            validation_embeddings, validation_choices
        ),
        "fit_native_reply_vs_static": native_reply_comparison(fit, fit_choices),
        "validation_native_reply_vs_static": native_reply_comparison(
            validation, validation_choices
        ),
        "qualification": "Offline native opponent-reply-score prediction only; no complete-game or Elo claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", required=True, action="append", type=Path)
    parser.add_argument("--validation", required=True, action="append", type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--checkpoint-out", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            scout(
                arguments.fit,
                arguments.validation,
                arguments.encoder,
                arguments.checkpoint_out,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
