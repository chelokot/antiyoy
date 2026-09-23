from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch

from .audit_duel_reply_margins import load_frozen_reply_model
from .build_bundle import digest
from .evaluate import paired_comparison_summary
from .scout_duel_reply_score import load_scored_turns


@dataclass(frozen=True)
class ReplyDecomposition:
    seed: int
    seat: int
    same_opponent_actions: bool
    root_static_gap: int
    reply_gap: int
    response_swing: int
    predicted_margin: float
    terminal_magnitude: bool


def summarize(rows: list[ReplyDecomposition]) -> dict[str, int | float | None]:
    ordinary = [row for row in rows if not row.terminal_magnitude]
    return {
        "positions": len(rows),
        "better": sum(row.reply_gap > 0 for row in rows),
        "worse": sum(row.reply_gap < 0 for row in rows),
        "same": sum(row.reply_gap == 0 for row in rows),
        "root_static_score_ties": sum(row.root_static_gap == 0 for row in rows),
        "terminal_magnitude_positions": len(rows) - len(ordinary),
        "ordinary_root_static_gap_median": (
            median(row.root_static_gap for row in ordinary) if ordinary else None
        ),
        "ordinary_reply_gap_median": (
            median(row.reply_gap for row in ordinary) if ordinary else None
        ),
        "ordinary_response_swing_median": (
            median(row.response_swing for row in ordinary) if ordinary else None
        ),
        "ordinary_positive_response_swings": sum(
            row.response_swing > 0 for row in ordinary
        ),
        "ordinary_negative_response_swings": sum(
            row.response_swing < 0 for row in ordinary
        ),
        "ordinary_zero_response_swings": sum(
            row.response_swing == 0 for row in ordinary
        ),
        "predicted_margin_median": (
            median(row.predicted_margin for row in rows) if rows else None
        ),
    }


def decompose(
    dataset_paths: list[Path], encoder_path: Path, checkpoint_path: Path
) -> dict[str, object]:
    frozen = load_frozen_reply_model(encoder_path, checkpoint_path)
    turns = load_scored_turns(dataset_paths, frozen.encoder)
    rows = []
    teacher_identical_actions = 0
    teacher_changed_actions = 0
    map_deltas = dict.fromkeys((turn.embedding.position.seed for turn in turns), 0)
    with torch.inference_mode():
        for turn in turns:
            if turn.static_scores is None or turn.opponent_actions is None:
                raise ValueError(
                    "reply decomposition requires scores and opponent actions"
                )
            teacher = turn.embedding.reply_index
            if teacher != 0:
                teacher_identical_actions += (
                    turn.opponent_actions[teacher] == turn.opponent_actions[0]
                )
                teacher_changed_actions += (
                    turn.opponent_actions[teacher] != turn.opponent_actions[0]
                )
            features = turn.embedding.position.features
            predictions = frozen.scorer((features - frozen.mean) / frozen.scale)
            selected = int(predictions.argmax())
            if selected == 0:
                continue
            root_static_gap = int(turn.static_scores[selected] - turn.static_scores[0])
            if root_static_gap > 0:
                raise ValueError("rank zero must have the highest static score")
            reply_gap = int(turn.reply_scores[selected] - turn.reply_scores[0])
            row = ReplyDecomposition(
                seed=turn.embedding.position.seed,
                seat=turn.embedding.position.seat,
                same_opponent_actions=(
                    turn.opponent_actions[selected] == turn.opponent_actions[0]
                ),
                root_static_gap=root_static_gap,
                reply_gap=reply_gap,
                response_swing=reply_gap - root_static_gap,
                predicted_margin=float(predictions[selected] - predictions[0]),
                terminal_magnitude=(
                    abs(turn.reply_scores[selected]) >= 1e9
                    or abs(turn.reply_scores[0]) >= 1e9
                ),
            )
            rows.append(row)
            map_deltas[row.seed] += int(reply_gap > 0) - int(reply_gap < 0)
    return {
        "kind": "procedural_duel_reply_score_decomposition",
        "dataset_files": [
            {"name": path.name, "sha256": digest(path)} for path in dataset_paths
        ],
        "encoder_sha256": frozen.encoder_sha256,
        "selected_expert": frozen.selected_expert,
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "maps": len(map_deltas),
        "positions": len(turns),
        "model_overrides": summarize(rows),
        "by_seat": {
            str(seat): summarize([row for row in rows if row.seat == seat])
            for seat in sorted({turn.embedding.position.seat for turn in turns})
        },
        "by_opponent_action_sequence": {
            "identical": summarize([row for row in rows if row.same_opponent_actions]),
            "changed": summarize(
                [row for row in rows if not row.same_opponent_actions]
            ),
        },
        "teacher_nonstatic_choices": {
            "identical_opponent_actions": teacher_identical_actions,
            "changed_opponent_actions": teacher_changed_actions,
        },
        "independent_maps": paired_comparison_summary(
            sum(delta > 0 for delta in map_deltas.values()),
            sum(delta < 0 for delta in map_deltas.values()),
            sum(delta == 0 for delta in map_deltas.values()),
        ),
        "qualification": (
            "Post hoc decomposition of already-inspected conditional native reply "
            "scores; not a policy, full-game, or Elo result"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, action="append", type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            decompose(arguments.dataset, arguments.encoder, arguments.checkpoint),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
