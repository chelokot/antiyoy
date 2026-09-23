from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from antiyoy_rl.slate_dataset import load_teacher_slates

from .audit_native_score_observation import INVALID_PROVINCE
from .build_bundle import digest
from .scout_duel_turn_value import side_features


UNIT_SCORES = (28, 82, 210, 430)


def aggregate_score(
    observation: dict[str, np.ndarray],
    state: int,
    root_seat: int,
    economy: dict[str, object],
) -> int:
    score = 0
    unit_upkeep = cast(list[int], economy["unit_upkeep"])
    for owner in (root_seat, 1 - root_seat):
        features = [int(value) for value in side_features(observation, state, owner)]
        territory, *units = features[:5]
        capital, farm, tower, strong_tower, tree = features[6:11]
        defense, money, _, provinces = features[11:15]
        income = (
            int(economy["clear_hex_income"]) * (territory - farm - tree)
            + int(economy["farm_hex_income"]) * farm
        )
        upkeep = (
            sum(
                count * unit_upkeep[strength]
                for strength, count in enumerate(units, start=1)
            )
            + tower * int(economy["tower_upkeep"])
            + strong_tower * int(economy["strong_tower_upkeep"])
        )
        side_score = (
            128 * territory
            + sum(
                count * value for count, value in zip(units, UNIT_SCORES, strict=True)
            )
            + 42 * capital
            + 58 * farm
            + 46 * tower
            + 96 * strong_tower
            - 10 * tree
            + 5 * defense
            + money
            + 11 * income
            - 15 * upkeep
            - 14 * provinces
        )
        score += side_score if owner == root_seat else -side_score
    return score


def omitted_score(
    observation: dict[str, np.ndarray],
    state: int,
    root_seat: int,
    economy: dict[str, object],
) -> tuple[int, int, int, int]:
    start, end = (
        int(value) for value in observation["cell_offsets"][state : state + 2]
    )
    grave_delta = orphan_economy_delta = grave_count = orphan_count = 0
    unit_upkeep = cast(list[int], economy["unit_upkeep"])
    for cell in range(start, end):
        owner = int(observation["owners"][cell])
        if owner == 255:
            continue
        direction = 1 if owner == root_seat else -1
        obj = int(observation["objects"][cell])
        strength = int(observation["unit_strengths"][cell])
        if obj == 7:
            grave_delta += direction
            grave_count += 1
        if int(observation["province_ids"][cell]) != INVALID_PROVINCE:
            continue
        orphan_count += 1
        income = (
            0
            if obj in (5, 6)
            else int(economy["farm_hex_income"])
            if obj == 2
            else int(economy["clear_hex_income"])
        )
        upkeep = (
            unit_upkeep[strength]
            if strength > 0
            else int(economy["tower_upkeep"])
            if obj == 3
            else int(economy["strong_tower_upkeep"])
            if obj == 4
            else 0
        )
        orphan_economy_delta += direction * (11 * income - 15 * upkeep)
    return -16 * grave_delta, -orphan_economy_delta, grave_count, orphan_count


def audit(paths: list[Path]) -> dict[str, object]:
    files = {}
    totals = {
        "nonterminal_candidates": 0,
        "terminal_candidates_excluded": 0,
        "aggregate_exact": 0,
        "aggregate_mismatches": 0,
        "grave_candidates": 0,
        "orphan_candidates": 0,
        "corrected_exact": 0,
        "maximum_aggregate_error": 0,
        "nonterminal_slates": 0,
        "terminal_slates_excluded": 0,
        "aggregate_teacher_matches": 0,
        "static_teacher_matches": 0,
        "aggregate_better_than_static": 0,
        "aggregate_worse_than_static": 0,
    }
    for path in paths:
        positions = load_teacher_slates(path)
        file_totals = dict.fromkeys(totals, 0)
        for position in positions:
            if position.post_reply is None:
                raise ValueError("aggregate audit requires post-reply observations")
            scores = []
            for state, target in enumerate(position.opponent_reply_scores):
                if abs(target) >= 1_000_000_000:
                    file_totals["terminal_candidates_excluded"] += 1
                    scores.append(None)
                    continue
                economy = cast(
                    dict[str, object],
                    json.loads(position.post_turn_rules_json[state])["economy"],
                )
                predicted = aggregate_score(
                    position.post_reply, state, position.seat, economy
                )
                grave, orphan, grave_count, orphan_count = omitted_score(
                    position.post_reply, state, position.seat, economy
                )
                error = abs(predicted - int(target))
                file_totals["nonterminal_candidates"] += 1
                file_totals["aggregate_exact"] += error == 0
                file_totals["aggregate_mismatches"] += error != 0
                file_totals["grave_candidates"] += grave_count > 0
                file_totals["orphan_candidates"] += orphan_count > 0
                file_totals["corrected_exact"] += int(
                    predicted + grave + orphan == target
                )
                file_totals["maximum_aggregate_error"] = max(
                    file_totals["maximum_aggregate_error"], error
                )
                scores.append(predicted)
            if any(score is None for score in scores):
                file_totals["terminal_slates_excluded"] += 1
                continue
            file_totals["nonterminal_slates"] += 1
            choice = max(
                range(len(scores)),
                key=lambda index: (
                    scores[index],
                    position.static_scores[index],
                    -index,
                ),
            )
            teacher = max(
                range(len(scores)),
                key=lambda index: (
                    position.opponent_reply_scores[index],
                    position.static_scores[index],
                    -index,
                ),
            )
            file_totals["aggregate_teacher_matches"] += choice == teacher
            file_totals["static_teacher_matches"] += teacher == 0
            file_totals["aggregate_better_than_static"] += int(
                position.opponent_reply_scores[choice]
                > position.opponent_reply_scores[0]
            )
            file_totals["aggregate_worse_than_static"] += int(
                position.opponent_reply_scores[choice]
                < position.opponent_reply_scores[0]
            )
        for name, count in file_totals.items():
            if name == "maximum_aggregate_error":
                totals[name] = max(totals[name], count)
            else:
                totals[name] += count
        files[path.name] = {
            "sha256": digest(path),
            "positions": len(positions),
            **file_totals,
        }
    return {
        "kind": "procedural_duel_aggregate_response_score_sufficiency",
        "files": files,
        "totals": totals,
        "qualification": "Read-only post hoc audit on already inspected exact post-reply states; the opponent successor is unavailable to an instantaneous online student",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="+")
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset), sort_keys=True))


if __name__ == "__main__":
    main()
