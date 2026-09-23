from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from .build_bundle import digest
from .evaluate import load_policy
from .scout_duel_nonlinear_choice import TurnScorer
from .scout_duel_pairwise_response import (
    candidate_logits,
    evaluate_threshold,
    pair_metrics,
)
from .scout_duel_reply_score import load_scored_turns


def load_frozen(
    checkpoint_path: Path, encoder_path: Path
) -> tuple[TurnScorer, Tensor, Tensor, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["encoder_sha256"] != digest(encoder_path):
        raise ValueError("pairwise checkpoint and spatial encoder differ")
    if checkpoint["feature_transform"] != "asinh_static_score":
        raise ValueError("pairwise checkpoint feature transform differs")
    if checkpoint["target"] != "candidate_reply_score_strictly_exceeds_static":
        raise ValueError("pairwise checkpoint target differs")
    mean = cast(Tensor, checkpoint["mean"])
    scale = cast(Tensor, checkpoint["scale"])
    scorer = TurnScorer(3 * len(mean))
    scorer.load_state_dict(checkpoint["state_dict"])
    scorer.eval()
    return scorer, mean, scale, cast(float, checkpoint["selected_threshold"])


def passes_offline_gate(result: dict[str, object]) -> bool:
    comparison = cast(dict[str, object], result["native_reply_vs_static"])
    seats = cast(dict[int, dict[str, int]], comparison["by_seat"])
    grouped = cast(dict[str, float | int], comparison["independent_maps"])
    return (
        cast(int, result["overrides"]) >= 20
        and cast(int, comparison["better"]) > cast(int, comparison["worse"])
        and set(seats) == {0, 1}
        and all(seat["better"] > seat["worse"] for seat in seats.values())
        and grouped["exact_two_sided_sign_test_p"] < 0.05
    )


def validate(
    dataset_path: Path, encoder_path: Path, checkpoint_path: Path
) -> dict[str, object]:
    torch.set_num_threads(1)
    scorer, mean, scale, threshold = load_frozen(checkpoint_path, encoder_path)
    if threshold != 0.9:
        raise ValueError("predeclared calibration did not select threshold 0.9")
    encoder, config = load_policy(
        encoder_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    turns = load_scored_turns([dataset_path], encoder)
    map_seeds = {turn.embedding.position.seed for turn in turns}
    if len(map_seeds) != 256 or min(map_seeds) < 6318000 or max(map_seeds) >= 6318256:
        raise ValueError("pairwise response validation map window differs")
    with torch.inference_mode():
        logits = [candidate_logits(turn, scorer, mean, scale) for turn in turns]
    result = evaluate_threshold(turns, logits, threshold)
    return {
        "kind": "procedural_duel_pairwise_response_fresh_validation",
        "protocol": "benchmarks/protocols/2026-09-23-duel-pairwise-response-v1.json",
        "dataset_name": dataset_path.name,
        "dataset_sha256": digest(dataset_path),
        "encoder_sha256": digest(encoder_path),
        "selected_expert": config["selected_expert"],
        "checkpoint_sha256": digest(checkpoint_path),
        "independent_maps_with_positions": len(map_seeds),
        "positions": len(turns),
        "pair_metrics_at_0_5": pair_metrics(turns, logits),
        "frozen_threshold_result": result,
        "passes_predeclared_offline_gate": passes_offline_gate(result),
        "qualification": (
            "Fresh conditional native reply-score validation only. Not complete "
            "games, Elo, rated-agent or browser strength."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            validate(arguments.dataset, arguments.encoder, arguments.checkpoint),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
