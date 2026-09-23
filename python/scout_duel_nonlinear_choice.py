from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from antiyoy_rl.slate_dataset import load_teacher_slates

from .build_bundle import digest
from .evaluate import load_policy
from .scout_duel_teacher_choice import (
    agreement_for_choices,
    embed_spatial,
    teacher_choice_pairs,
)
from .scout_duel_turn_value import DuelEmbedding


class TurnScorer(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features).flatten()


def choices(
    positions: list[DuelEmbedding], scorer: TurnScorer, mean: Tensor, scale: Tensor
) -> list[int]:
    selected = []
    with torch.inference_mode():
        for position in positions:
            values = scorer((position.position.features - mean) / scale)
            selected.append(int(values.argmax()))
    return selected


def train_scorer(
    positions: list[DuelEmbedding], epochs: int = 20
) -> tuple[TurnScorer, Tensor, Tensor, list[float], int]:
    torch.manual_seed(72031)
    features = torch.cat([position.position.features for position in positions])
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(0.01)
    chosen, alternatives, weights = teacher_choice_pairs(positions)
    chosen = (chosen - mean) / scale
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
            difference = scorer(chosen[indices]) - scorer(alternatives[indices])
            loss = (functional.softplus(-difference) * batch_weights).sum()
            loss = loss / batch_weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_weight = batch_weights.sum().item()
            weighted_loss += loss.detach().item() * batch_weight
            total_weight += batch_weight
        losses.append(weighted_loss / total_weight)
    return scorer, mean, scale, losses, len(weights)


def scout(
    fit_paths: list[Path],
    holdout_paths: list[Path],
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
    fit = [
        embed_spatial(position, encoder)
        for path in fit_paths
        for position in load_teacher_slates(path)
    ]
    holdout = [
        embed_spatial(position, encoder)
        for path in holdout_paths
        for position in load_teacher_slates(path)
    ]
    fit_maps = {position.position.seed for position in fit}
    holdout_maps = {position.position.seed for position in holdout}
    if fit_maps & holdout_maps:
        raise ValueError("fit and holdout maps overlap")
    scorer, mean, scale, losses, pair_count = train_scorer(fit)
    torch.save(
        {
            "state_dict": scorer.state_dict(),
            "mean": mean,
            "scale": scale,
            "encoder_sha256": digest(encoder_path),
        },
        checkpoint_path,
    )
    return {
        "kind": "procedural_duel_nonlinear_teacher_choice_diagnostic",
        "fit_files": [
            {"name": path.name, "sha256": digest(path)} for path in fit_paths
        ],
        "holdout_files": [
            {"name": path.name, "sha256": digest(path)} for path in holdout_paths
        ],
        "encoder_sha256": digest(encoder_path),
        "selected_expert": config["selected_expert"],
        "fit_maps": len(fit_maps),
        "holdout_maps": len(holdout_maps),
        "fit_pairs": pair_count,
        "epochs": len(losses),
        "first_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "checkpoint_sha256": digest(checkpoint_path),
        "fit_agreement": agreement_for_choices(fit, choices(fit, scorer, mean, scale)),
        "holdout_agreement": agreement_for_choices(
            holdout, choices(holdout, scorer, mean, scale)
        ),
        "qualification": "Offline hard teacher-choice imitation diagnostic only; no complete-game or Elo claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", required=True, action="append", type=Path)
    parser.add_argument("--holdout", required=True, action="append", type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--checkpoint-out", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            scout(
                arguments.fit,
                arguments.holdout,
                arguments.encoder,
                arguments.checkpoint_out,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
