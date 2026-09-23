from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import torch

from .build_bundle import digest
from .evaluate import load_policy
from .scout_duel_nonlinear_choice import choices
from .scout_duel_reply_score import (
    ScoredTurn,
    load_scored_turns,
    native_reply_comparison,
    score_metrics,
    train_scorer,
)
from .scout_duel_teacher_choice import agreement_for_choices


def stage_metrics(fit: list[ScoredTurn], validation: list[ScoredTurn]) -> dict[str, object]:
    scorer, mean, scale, losses, pair_count = train_scorer(fit)
    fit_choices = choices([turn.embedding for turn in fit], scorer, mean, scale)
    validation_choices = choices(
        [turn.embedding for turn in validation], scorer, mean, scale
    )
    return {
        "fit_pairs": pair_count,
        "epochs": len(losses),
        "first_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "fit_score_metrics": score_metrics(fit, scorer, mean, scale),
        "validation_score_metrics": score_metrics(validation, scorer, mean, scale),
        "fit_teacher_agreement": agreement_for_choices(
            [turn.embedding for turn in fit], fit_choices
        ),
        "validation_teacher_agreement": agreement_for_choices(
            [turn.embedding for turn in validation], validation_choices
        ),
        "fit_native_reply_vs_static": native_reply_comparison(fit, fit_choices),
        "validation_native_reply_vs_static": native_reply_comparison(
            validation, validation_choices
        ),
    }


def audit(
    fit_path: Path, validation_path: Path, encoder_path: Path
) -> dict[str, object]:
    torch.set_num_threads(1)
    encoder, config = load_policy(
        encoder_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    stages: dict[str, dict[str, object]] = {}
    fit_maps: set[int] | None = None
    validation_maps: set[int] | None = None
    for stage in ("post_turn", "post_reply"):
        selected_stage: Literal["post_turn", "post_reply"] = stage
        fit = load_scored_turns([fit_path], encoder, selected_stage, "root")
        validation = load_scored_turns(
            [validation_path], encoder, selected_stage, "root"
        )
        stage_fit_maps = {turn.embedding.position.seed for turn in fit}
        stage_validation_maps = {
            turn.embedding.position.seed for turn in validation
        }
        if stage_fit_maps & stage_validation_maps:
            raise ValueError("fit and validation maps overlap")
        if fit_maps is not None and (
            stage_fit_maps != fit_maps or stage_validation_maps != validation_maps
        ):
            raise ValueError("observation stages do not contain identical maps")
        fit_maps = stage_fit_maps
        validation_maps = stage_validation_maps
        stages[stage] = {
            "fit_positions": len(fit),
            "validation_positions": len(validation),
            "fit_terminal_magnitude_candidates": sum(
                turn.terminal_magnitude_scores for turn in fit
            ),
            "validation_terminal_magnitude_candidates": sum(
                turn.terminal_magnitude_scores for turn in validation
            ),
            **stage_metrics(fit, validation),
        }
    return {
        "kind": "procedural_duel_opponent_response_state_boundary",
        "fit_file": {"name": fit_path.name, "sha256": digest(fit_path)},
        "validation_file": {
            "name": validation_path.name,
            "sha256": digest(validation_path),
        },
        "encoder_sha256": digest(encoder_path),
        "selected_expert": config["selected_expert"],
        "spatial_perspective": "root",
        "fit_maps": len(fit_maps or ()),
        "validation_maps": len(validation_maps or ()),
        "stages": stages,
        "qualification": "Frozen-encoder offline information-boundary diagnostic; post-reply state is not available to an instantaneous online student",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.fit, arguments.validation, arguments.encoder)))


if __name__ == "__main__":
    main()
