from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from antiyoy_rl.slate_dataset import load_teacher_slates

from .build_bundle import digest


@dataclass(frozen=True)
class PairTarget:
    seed: int
    seat: int
    static_gap: int
    reply_gap: int
    response_swing: int
    terminal_magnitude: bool


def summarize_pairs(pairs: list[PairTarget]) -> dict[str, int | float | None]:
    ordinary = [pair for pair in pairs if not pair.terminal_magnitude]
    beneficial = [pair for pair in ordinary if pair.reply_gap > 0]
    return {
        "pairs": len(pairs),
        "reply_better": sum(pair.reply_gap > 0 for pair in pairs),
        "reply_worse": sum(pair.reply_gap < 0 for pair in pairs),
        "reply_same": sum(pair.reply_gap == 0 for pair in pairs),
        "static_ties": sum(pair.static_gap == 0 for pair in pairs),
        "positive_response_swings": sum(pair.response_swing > 0 for pair in pairs),
        "negative_response_swings": sum(pair.response_swing < 0 for pair in pairs),
        "zero_response_swings": sum(pair.response_swing == 0 for pair in pairs),
        "terminal_magnitude_pairs": len(pairs) - len(ordinary),
        "ordinary_positive_static_gap_median": (
            median(-pair.static_gap for pair in beneficial) if beneficial else None
        ),
        "ordinary_positive_response_swing_median": (
            median(pair.response_swing for pair in beneficial) if beneficial else None
        ),
    }


def audit(paths: list[Path]) -> dict[str, object]:
    pairs = []
    positions = 0
    maps = set()
    maps_with_better = set()
    teacher_overrides = 0
    for path in paths:
        for position in load_teacher_slates(path):
            positions += 1
            maps.add(position.seed)
            selected = max(
                range(len(position.static_scores)),
                key=lambda index: (
                    position.opponent_reply_scores[index],
                    position.static_scores[index],
                    -index,
                ),
            )
            if selected != 0:
                teacher_overrides += 1
            for index in range(1, len(position.static_scores)):
                static_gap = int(
                    position.static_scores[index] - position.static_scores[0]
                )
                reply_gap = int(
                    position.opponent_reply_scores[index]
                    - position.opponent_reply_scores[0]
                )
                if static_gap > 0:
                    raise ValueError("rank zero must have the highest static score")
                if reply_gap > 0:
                    maps_with_better.add(position.seed)
                pairs.append(
                    PairTarget(
                        seed=position.seed,
                        seat=position.seat,
                        static_gap=static_gap,
                        reply_gap=reply_gap,
                        response_swing=reply_gap - static_gap,
                        terminal_magnitude=(
                            abs(position.opponent_reply_scores[index]) >= 1e9
                            or abs(position.opponent_reply_scores[0]) >= 1e9
                        ),
                    )
                )
    return {
        "kind": "procedural_duel_pair_target_prevalence",
        "datasets": [{"name": path.name, "sha256": digest(path)} for path in paths],
        "maps": len(maps),
        "positions": positions,
        "maps_with_better_alternative": len(maps_with_better),
        "teacher_overrides": teacher_overrides,
        "all_pairs": summarize_pairs(pairs),
        "by_seat": {
            str(seat): summarize_pairs([pair for pair in pairs if pair.seat == seat])
            for seat in sorted({pair.seat for pair in pairs})
        },
        "qualification": "Read-only label prevalence on already-used training maps, not a policy or strength result",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, action="append", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset), sort_keys=True))


if __name__ == "__main__":
    main()
