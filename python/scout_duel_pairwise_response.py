from __future__ import annotations

import argparse
import json
from collections import Counter
from math import log
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from .build_bundle import digest
from .evaluate import load_policy
from .scout_duel_nonlinear_choice import TurnScorer
from .scout_duel_reply_score import (
    ScoredTurn,
    load_scored_turns,
    native_reply_comparison,
)
from .scout_duel_teacher_choice import agreement_for_choices


THRESHOLDS = (0.5, 0.7, 0.9)


def transformed_features(turn: ScoredTurn) -> Tensor:
    features = turn.embedding.position.features.clone()
    features[:, 0] = torch.asinh(features[:, 0])
    return features


def pair_features(baseline: Tensor, alternatives: Tensor) -> Tensor:
    return torch.cat((baseline, alternatives, alternatives - baseline), dim=1)


def examples(
    turns: list[ScoredTurn], mean: Tensor, scale: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    positions_by_map = Counter(turn.embedding.position.seed for turn in turns)
    batches = []
    targets = []
    weights = []
    for turn in turns:
        features = (transformed_features(turn) - mean) / scale
        competitors = len(features) - 1
        if competitors == 0:
            continue
        batches.append(pair_features(features[0].expand(competitors, -1), features[1:]))
        targets.append(
            torch.as_tensor(
                turn.reply_scores[1:] > turn.reply_scores[0], dtype=torch.float32
            )
        )
        weight = 1 / (
            len(positions_by_map)
            * positions_by_map[turn.embedding.position.seed]
            * competitors
        )
        weights.append(torch.full((competitors,), weight))
    if not batches:
        raise ValueError("pairwise response training requires competing turns")
    return torch.cat(batches), torch.cat(targets), torch.cat(weights)


def train(
    turns: list[ScoredTurn], epochs: int = 20
) -> tuple[TurnScorer, Tensor, Tensor, list[float], int]:
    torch.manual_seed(72037)
    features = torch.cat([transformed_features(turn) for turn in turns])
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(0.01)
    inputs, targets, weights = examples(turns, mean, scale)
    scorer = TurnScorer(inputs.shape[1])
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=0.001, weight_decay=0.01)
    generator = torch.Generator().manual_seed(72037)
    positive_weight = torch.tensor(4.0)
    losses = []
    for _ in range(epochs):
        weighted_loss = 0.0
        total_weight = 0.0
        for indices in torch.randperm(len(weights), generator=generator).split(2048):
            batch_weights = weights[indices]
            prediction = scorer(inputs[indices])
            per_pair = functional.binary_cross_entropy_with_logits(
                prediction,
                targets[indices],
                pos_weight=positive_weight,
                reduction="none",
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


def candidate_logits(
    turn: ScoredTurn, scorer: nn.Module, mean: Tensor, scale: Tensor
) -> Tensor:
    features = (transformed_features(turn) - mean) / scale
    baseline = features[0].expand(len(features) - 1, -1)
    return scorer(pair_features(baseline, features[1:]))


def select_candidate(logits: Tensor, threshold: float) -> int:
    if logits.numel() == 0:
        return 0
    best = int(logits.argmax())
    return best + 1 if logits[best].item() > log(threshold / (1 - threshold)) else 0


def evaluate_threshold(
    turns: list[ScoredTurn], logits: list[Tensor], threshold: float
) -> dict[str, object]:
    selected = [select_candidate(row, threshold) for row in logits]
    return {
        "threshold": threshold,
        "overrides": sum(index != 0 for index in selected),
        "agreement": agreement_for_choices(
            [turn.embedding for turn in turns], selected
        ),
        "native_reply_vs_static": native_reply_comparison(turns, selected),
    }


def qualifying_threshold(results: list[dict[str, object]]) -> float | None:
    eligible = []
    for report in results:
        comparison = report["native_reply_vs_static"]
        assert isinstance(comparison, dict)
        if report["overrides"] < 20 or comparison["better"] <= comparison["worse"]:
            continue
        if any(
            seat["better"] < seat["worse"] for seat in comparison["by_seat"].values()
        ):
            continue
        eligible.append(report)
    if not eligible:
        return None
    chosen = max(
        eligible,
        key=lambda report: (
            report["native_reply_vs_static"]["better"]
            - report["native_reply_vs_static"]["worse"],
            report["threshold"],
        ),
    )
    return float(chosen["threshold"])


def pair_metrics(turns: list[ScoredTurn], logits: list[Tensor]) -> dict[str, int]:
    positive = predicted = true_positive = pairs = 0
    for turn, row in zip(turns, logits, strict=True):
        target = torch.as_tensor(turn.reply_scores[1:] > turn.reply_scores[0])
        prediction = row > 0
        pairs += len(row)
        positive += int(target.sum())
        predicted += int(prediction.sum())
        true_positive += int((target & prediction).sum())
    return {
        "pairs": pairs,
        "actual_positive": positive,
        "predicted_positive_at_0_5": predicted,
        "true_positive_at_0_5": true_positive,
    }


def scout(
    dataset_paths: list[Path], encoder_path: Path, checkpoint_path: Path
) -> dict[str, object]:
    torch.set_num_threads(1)
    encoder, config = load_policy(
        encoder_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    turns = load_scored_turns(dataset_paths, encoder)
    fit = [turn for turn in turns if 6280000 <= turn.embedding.position.seed < 6280640]
    calibration = [
        turn for turn in turns if 6280640 <= turn.embedding.position.seed < 6280768
    ]
    if len({turn.embedding.position.seed for turn in fit}) != 640:
        raise ValueError("pairwise response fit map count differs from protocol")
    if len({turn.embedding.position.seed for turn in calibration}) != 128:
        raise ValueError(
            "pairwise response calibration map count differs from protocol"
        )
    if len(fit) + len(calibration) != len(turns):
        raise ValueError("pairwise response datasets extend beyond protocol windows")
    scorer, mean, scale, losses, pairs = train(fit)
    with torch.inference_mode():
        fit_logits = [candidate_logits(turn, scorer, mean, scale) for turn in fit]
        calibration_logits = [
            candidate_logits(turn, scorer, mean, scale) for turn in calibration
        ]
    calibration_results = [
        evaluate_threshold(calibration, calibration_logits, threshold)
        for threshold in THRESHOLDS
    ]
    threshold = qualifying_threshold(calibration_results)
    torch.save(
        {
            "state_dict": scorer.state_dict(),
            "mean": mean,
            "scale": scale,
            "encoder_sha256": digest(encoder_path),
            "selected_threshold": threshold,
            "feature_transform": "asinh_static_score",
            "target": "candidate_reply_score_strictly_exceeds_static",
        },
        checkpoint_path,
    )
    return {
        "kind": "procedural_duel_pairwise_response_scout",
        "protocol": "benchmarks/protocols/2026-09-23-duel-pairwise-response-v1.json",
        "dataset_files": [
            {"name": path.name, "sha256": digest(path)} for path in dataset_paths
        ],
        "encoder_sha256": digest(encoder_path),
        "selected_expert": config["selected_expert"],
        "checkpoint_sha256": digest(checkpoint_path),
        "fit_maps": 640,
        "calibration_maps": 128,
        "fit_positions": len(fit),
        "calibration_positions": len(calibration),
        "fit_pairs": pairs,
        "fit_first_epoch_loss": losses[0],
        "fit_final_epoch_loss": losses[-1],
        "fit_pair_metrics": pair_metrics(fit, fit_logits),
        "calibration_pair_metrics": pair_metrics(calibration, calibration_logits),
        "calibration_thresholds": calibration_results,
        "selected_threshold": threshold,
        "qualification": (
            "Calibration on existing training maps only. No independent offline, "
            "complete-game, Elo, rated-agent or browser strength claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, action="append", type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--checkpoint-out", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            scout(arguments.dataset, arguments.encoder, arguments.checkpoint_out),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
