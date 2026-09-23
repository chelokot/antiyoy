from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .build_bundle import digest
from .evaluate import load_policy, paired_comparison_summary
from .scout_duel_nonlinear_choice import TurnScorer
from .scout_duel_reply_score import load_scored_turns, native_reply_comparison
from .scout_duel_teacher_choice import agreement_for_choices


@dataclass(frozen=True)
class Override:
    margin: float
    seed: int
    seat: int
    result: int


def summarize_prefix(overrides: list[Override]) -> dict[str, object]:
    better = sum(override.result > 0 for override in overrides)
    worse = sum(override.result < 0 for override in overrides)
    by_seat: dict[int, dict[str, int]] = {}
    map_deltas: dict[int, int] = {}
    for override in overrides:
        seat = by_seat.setdefault(override.seat, {"better": 0, "worse": 0, "same": 0})
        if override.result > 0:
            seat["better"] += 1
        elif override.result < 0:
            seat["worse"] += 1
        else:
            seat["same"] += 1
        map_deltas[override.seed] = map_deltas.get(override.seed, 0) + override.result
    grouped = paired_comparison_summary(
        sum(delta > 0 for delta in map_deltas.values()),
        sum(delta < 0 for delta in map_deltas.values()),
        sum(delta == 0 for delta in map_deltas.values()),
    )
    return {
        "positions": len(overrides),
        "minimum_predicted_margin": overrides[-1].margin,
        "better": better,
        "worse": worse,
        "same": len(overrides) - better - worse,
        "by_seat": by_seat,
        "independent_maps": grouped,
    }


def margin_prefixes(overrides: list[Override]) -> dict[str, dict[str, object]]:
    if not overrides:
        return {}
    ordered = sorted(
        overrides, key=lambda record: (-record.margin, record.seed, record.seat)
    )
    return {
        str(percent): summarize_prefix(
            ordered[: math.ceil(len(ordered) * percent / 100)]
        )
        for percent in (10, 25, 50, 75, 100)
    }


def gated_choice(predictions: torch.Tensor, minimum_margin: float) -> int:
    selected = int(predictions.argmax())
    if float(predictions[selected] - predictions[0]) < minimum_margin:
        return 0
    return selected


def audit(
    dataset_paths: list[Path],
    encoder_path: Path,
    checkpoint_path: Path,
    minimum_margin: float | None = None,
) -> dict[str, object]:
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder_sha256 = digest(encoder_path)
    if (
        checkpoint["encoder_sha256"] != encoder_sha256
        or checkpoint["spatial_perspective"] != "next_active"
        or checkpoint["target"] != "asinh(reply_score/1000)"
    ):
        raise ValueError("checkpoint does not match reply-score audit contract")
    mean = checkpoint["mean"]
    scale = checkpoint["scale"]
    scorer = TurnScorer(len(mean))
    scorer.load_state_dict(checkpoint["state_dict"])
    scorer.eval()
    encoder, config = load_policy(
        encoder_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    turns = load_scored_turns(dataset_paths, encoder)
    overrides = []
    selected_choices = []
    with torch.inference_mode():
        for turn in turns:
            features = turn.embedding.position.features
            predictions = scorer((features - mean) / scale)
            selected = int(predictions.argmax())
            if minimum_margin is not None:
                selected_choices.append(gated_choice(predictions, minimum_margin))
            if selected == 0:
                continue
            difference = turn.reply_scores[selected] - turn.reply_scores[0]
            overrides.append(
                Override(
                    margin=float(predictions[selected] - predictions[0]),
                    seed=turn.embedding.position.seed,
                    seat=turn.embedding.position.seat,
                    result=int(difference > 0) - int(difference < 0),
                )
            )
    report = {
        "kind": "procedural_duel_fit_only_reply_margin_audit",
        "dataset_files": [
            {"name": path.name, "sha256": digest(path)} for path in dataset_paths
        ],
        "encoder_sha256": encoder_sha256,
        "selected_expert": config["selected_expert"],
        "checkpoint_sha256": digest(checkpoint_path),
        "maps": len({turn.embedding.position.seed for turn in turns}),
        "positions": len(turns),
        "model_overrides": len(overrides),
        "qualification": "In-sample read-only diagnostic, not a policy, complete-game, or Elo result",
    }
    if minimum_margin is None:
        report["ranked_prefixes_percent_of_overrides"] = margin_prefixes(overrides)
    else:
        embeddings = [turn.embedding for turn in turns]
        report["fixed_gate"] = {
            "minimum_margin": minimum_margin,
            "selected_overrides": sum(choice != 0 for choice in selected_choices),
            "native_reply_vs_static": native_reply_comparison(turns, selected_choices),
            "teacher_agreement": agreement_for_choices(embeddings, selected_choices),
        }
        report["qualification"] = (
            "Offline fixed-threshold conditional reply-score comparison only; "
            "not complete-game or Elo strength"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, action="append", type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--minimum-margin", type=float)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(
                arguments.dataset,
                arguments.encoder,
                arguments.checkpoint,
                arguments.minimum_margin,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
