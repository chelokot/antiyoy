from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from antiyoy_rl.turn_credit import TurnCreditPosition, load_turn_credit_positions

from .build_bundle import digest


def preference(candidate: int, baseline: int) -> int:
    return (candidate > baseline) - (candidate < baseline)


def summarize(positions: list[TurnCreditPosition]) -> dict[str, object]:
    if not positions:
        raise ValueError("audit requires sampled positions")
    probe_nodes = {position.opponent_search_nodes for position in positions}
    if None in probe_nodes or len(probe_nodes) != 1:
        raise ValueError("audit requires one opponent search budget in every position")
    state_counts: Counter[str] = Counter()
    comparison_counts: Counter[str] = Counter()
    affected_state_maps = set()
    affected_preference_maps = set()
    strict_reversal_maps = set()
    greedy_opportunity_maps = set()
    probe_opportunity_maps = set()
    seat_counts: dict[int, Counter[str]] = {seat: Counter() for seat in range(5)}
    for position in positions:
        probe = position.opponent_search_scores
        if probe is None or len(probe) != len(position.outcome_scores):
            raise ValueError("opponent search labels must align with completed states")
        greedy = position.outcome_scores
        seat_counts[position.seat]["positions"] += 1
        state_changed = False
        preference_changed = False
        strict_reversal = False
        for greedy_score, probe_score in zip(greedy, probe, strict=True):
            if greedy_score < 0 or probe_score < 0:
                state_counts["censored"] += 1
            elif greedy_score == probe_score:
                state_counts["same"] += 1
            else:
                state_counts["changed"] += 1
                state_changed = True
        baseline = position.search_index
        if greedy[baseline] >= 0 and np.any(greedy > greedy[baseline]):
            greedy_opportunity_maps.add(position.seed)
        if probe[baseline] >= 0 and np.any(probe > probe[baseline]):
            probe_opportunity_maps.add(position.seed)
        for index in range(len(greedy)):
            if index == baseline:
                continue
            if min(greedy[index], greedy[baseline], probe[index], probe[baseline]) < 0:
                comparison_counts["censored"] += 1
                continue
            comparison_counts["complete"] += 1
            greedy_preference = preference(int(greedy[index]), int(greedy[baseline]))
            probe_preference = preference(int(probe[index]), int(probe[baseline]))
            if greedy_preference != 0 or probe_preference != 0:
                comparison_counts["informative"] += 1
            if greedy_preference == probe_preference:
                comparison_counts["same"] += 1
            else:
                comparison_counts["changed"] += 1
                preference_changed = True
            if greedy_preference * probe_preference < 0:
                comparison_counts["strict_reversals"] += 1
                strict_reversal = True
            if greedy_preference > 0 and probe_preference <= 0:
                comparison_counts["greedy_better_probe_not"] += 1
            if probe_preference > 0 and greedy_preference <= 0:
                comparison_counts["probe_better_greedy_not"] += 1
        if state_changed:
            affected_state_maps.add(position.seed)
            seat_counts[position.seat]["changed_outcome_positions"] += 1
        if preference_changed:
            affected_preference_maps.add(position.seed)
            seat_counts[position.seat]["changed_preference_positions"] += 1
        if strict_reversal:
            strict_reversal_maps.add(position.seed)
    return {
        "opponent_search_nodes": probe_nodes.pop(),
        "positions": len(positions),
        "independent_maps": len({position.seed for position in positions}),
        "distinct_states": sum(len(position.outcome_scores) for position in positions),
        "paired_state_outcomes": dict(state_counts),
        "candidate_vs_search": dict(comparison_counts),
        "maps_with_changed_state_outcome": len(affected_state_maps),
        "maps_with_changed_candidate_preference": len(affected_preference_maps),
        "maps_with_strict_preference_reversal": len(strict_reversal_maps),
        "maps_with_better_candidate": {
            "greedy_continuation": len(greedy_opportunity_maps),
            "search_opponents": len(probe_opportunity_maps),
        },
        "by_seat": {str(seat): dict(counts) for seat, counts in seat_counts.items()},
    }


def audit(paths: list[Path]) -> dict[str, object]:
    positions = [
        position
        for path in paths
        for position in load_turn_credit_positions(path)
    ]
    return {
        "kind": "whole_turn_opponent_continuation_label_stability",
        "source_files": [{"name": path.name, "sha256": digest(path)} for path in paths],
        "comparison": "greedy root and opponents versus greedy root with searched opponents",
        **summarize(positions),
        "qualification": "conditional terminal labels only; no policy promotion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.write_text(json.dumps(audit(arguments.input), indent=2) + "\n")


if __name__ == "__main__":
    main()
